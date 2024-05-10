import numpy as np
import torch
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, PNDMScheduler #UNet2DConditionModel, AutoencoderKL,
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast
import os
import random
from PIL import Image

resolution=(256,256)
random_flip=True
center_crop=False

transform_train = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
# data prep for test set
transform_test = transforms.Compose([
    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    #normalize
    transforms.Normalize([0.5], [0.5])
    ])

# Load the dataset
dataset = load_dataset("nlphuji/flickr30k")
split = dataset['test'].train_test_split(test_size=0.2, seed=42)
test_data=split['test']
split = split['train'].train_test_split(test_size=0.2, seed=42)
# val_data=split['test']
train_data=split['train']
# print(train_data)
# print(test_data)
# print(val_data)

# Preprocess dataset with text captions
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def preprocess_train(examples):
    image = [transform_train(example.convert("RGB")) for example in examples["image"]]
    caption = tokenizer([example[0] for example in examples["caption"]], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    return {"pixel_values": image, "caption_ids": caption['input_ids'], "caption_attention_mask": caption['attention_mask']}

def preprocess_test(examples):
    image = [transform_test(example.convert("RGB")) for example in examples["image"]]
    caption = [example[0] for example in examples["caption"]]
    return {"pixel_values": image, "prompt": caption}

# Apply preprocessing to each dataset
train_data.set_transform(preprocess_train, output_all_columns=True)
# train_data.set_format(type="torch", columns=["pixel_values", "caption_ids", "caption_attention_mask"], output_all_columns=True)
# val_data.set_transform(preprocess_test, output_all_columns=True)
# val_data.set_format(type="torch", columns=["pixel_values"])
test_data.set_transform(preprocess_test, output_all_columns=True)
# test_data.set_format(type="torch", columns=["pixel_values", "caption_ids", "caption_attention_mask"])

# Set up DataLoaders
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
# val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

weight_dtype=torch.float32
# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", dtype=weight_dtype)
# pipe =pip.to('cuda')

# Extract individual components for training
vae = pipe.vae
# tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = pipe.unet

# scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
scheduler = pipe.scheduler

# Optimizer
optimizer = AdamW(unet.parameters(), lr=1e-4)

# Training loop
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze vae and text_encoder and set unet to trainable
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

# Move components to the appropriate device
vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)

# Function to compute text embeddings
def get_text_embeddings(inputs, text_encoder):
    # inputs = {key: value.to(device) for key, value in inputs.items()}
    return text_encoder(inputs["caption_ids"].to(device), inputs["caption_attention_mask"].to(device)).last_hidden_state

def train(data_loader, vae, unet, tokenizer, text_encoder, scheduler, optimizer, device, weight_dtype, num_epochs):
    for epoch in range(num_epochs):
        unet.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # Prepare inputs
            latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor  # Scaling factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            text_embeddings = get_text_embeddings(batch, text_encoder)

            # Get model output
            model_output = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute loss (switch to float32 for stability)
            loss = mse_loss(model_output.float(), noise.float(), reduction="mean")
            train_loss += loss.item()

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"Loss": train_loss / len(progress_bar)})

        print(f"Epoch {epoch+1}: Avg Loss = {train_loss / len(train_dataloader)}")

### Evaluation Function

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM) between two images."""
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    return ssim(img1, img2, multichannel=True)


def resize_generated_image(image, size):
    """Resize generated images to match the size of the real image."""
    pil_image = Image.fromarray(image)  # Convert back to PIL Image
    return np.array(pil_image.resize((size[0], size[1]), Image.Resampling.LANCZOS))
def evaluate_model(data_loader, pipeline, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype, num_inference_steps):
    """Evaluate the Stable Diffusion model on a given dataset."""
    unet.eval()
    pipeline.to("cuda")
    ssim_scores = []
    generator = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            generated_images = []
            with autocast(device_type="cuda", dtype=weight_dtype):
                # Generate images one prompt at a time
                for prompt in batch["prompt"]:
                    image = pipeline(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
                    generated_images.append(image)
            # # Encode the images to latents
            # latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(device)).latent_dist.sample()
            # latents = latents * vae.config.scaling_factor
            #
            # # Generate captions to text embeddings
            # text_embeddings = get_text_embeddings(batch, text_encoder)
            #
            # # Set initial noise
            # noise = torch.randn_like(latents)
            #
            # # Run the denoising loop backward
            # for t in reversed(range(num_inference_steps)):
            #     timestep = torch.tensor([t], device=device).long()
            #     model_output = unet(noise, timestep, encoder_hidden_states=text_embeddings).sample
            #     noise = scheduler.step(model_output, timestep, noise).prev_sample
            #
            # # Decode the latents back to images
            # generated_images = vae.decode(noise / vae.config.scaling_factor).sample
            torch.cuda.empty_cache()

            # Calculate SSIM scores between original and generated images
            for real, generated in zip(batch["pixel_values"], generated_images):
                real = (real * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0) * 255  # Denormalize and convert to uint8
                generated = np.array(generated)
                real_size = (real.shape[1], real.shape[0])
                generated = resize_generated_image(generated, real_size)
                ssim_score = ssim(real, generated, win_size=3, data_range=1.0, multichannel=True)
                ssim_scores.append(ssim_score)
            print(ssim_score)
            break

    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim}")

def validate_with_prompts(pipeline, validation_prompts, num_inference_steps=20, output_dir=""):
    """Generate and display images for a list of validation prompts."""
    unet.eval()
    pipeline.to("cuda")
    images = []
    generator = torch.Generator(device=device).manual_seed(42)
    for prompt in validation_prompts:
        with autocast(device_type="cuda", dtype=weight_dtype):
            image = pipeline(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
            image.save(os.path.join(output_dir, f"validation_output_{prompt}.png"))
        images.append(image)
    return images

# Train and evaluate
train(train_dataloader, vae, unet, tokenizer, text_encoder, scheduler, optimizer, device, weight_dtype, num_epochs)
evaluate_model(test_dataloader, pipe, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype, num_inference_steps=20)

# Example validation prompts
validation_prompts = [
    "A beautiful landscape with mountains",
    "A futuristic cityscape at night",
    "An underwater coral reef scene",
    "A close-up of a robotic hand"
]
validation_images = validate_with_prompts(pipe, validation_prompts, num_inference_steps=20)