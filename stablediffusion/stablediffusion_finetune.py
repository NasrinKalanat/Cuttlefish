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

resolution=68
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

def preprocess(example, transform):
    image = transform(example["image"].convert("RGB"))
    caption = tokenizer(example["caption"][random.randint(0,len(example["caption"])-1)], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    return {"pixel_values": image, "text": caption}

# Apply preprocessing to each dataset
train_data = train_data.map(lambda x: preprocess(x, transform_train), remove_columns=["image", "caption"])
train_data.set_format(type="torch", columns=["pixel_values", "text"], output_all_columns=True)
# val_data = val_data.map(preprocess_test, remove_columns=["image", "caption"])
# val_data.set_format(type="torch", columns=["pixel_values", "text"])
test_data = test_data.map(lambda x: preprocess(x, transform_test), remove_columns=["image", "caption"])
test_data.set_format(type="torch", columns=["pixel_values", "text"])

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
# text_encoder = pipe.text_encoder
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = pipe.unet

scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

# Optimizer
optimizer = AdamW(unet.parameters(), lr=1e-4)

# Training loop
num_epochs = 1#100
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
    return text_encoder(inputs, return_dict=False).last_hidden_state

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
            text_embeddings = get_text_embeddings(batch["text"].to(device), text_encoder)

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

def evaluate_model(data_loader, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype):
    """Evaluate the Stable Diffusion model on a given dataset."""
    unet.eval()
    ssim_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Encode the images to latents
            latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Generate captions to text embeddings
            text_embeddings = get_text_embeddings(batch["text"], text_encoder)

            # Set initial noise
            noise = torch.randn_like(latents)

            # Run the denoising loop backward
            for t in reversed(range(scheduler.config.num_train_timesteps)):
                timestep = torch.tensor([t], device=device).long()
                model_output = unet(noise, timestep, encoder_hidden_states=text_embeddings).sample
                noise = scheduler.step(model_output, timestep, noise).prev_sample

            # Decode the latents back to images
            generated_images = vae.decode(noise / vae.config.scaling_factor).sample

            # Calculate SSIM scores between original and generated images
            for real, generated in zip(batch["pixel_values"], generated_images):
                ssim_score = calculate_ssim(real, generated)
                ssim_scores.append(ssim_score)

    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim}")

def validate_with_prompts(pipeline, validation_prompts, num_inference_steps=20, output_dir="out"):
    """Generate and display images for a list of validation prompts."""
    pipeline.to("cuda")
    images = []
    for prompt in validation_prompts:
        with autocast("cuda"):
            image = pipeline(prompt, num_inference_steps=num_inference_steps).images[0]
            image.save(os.path.join(output_dir, f"validation_output_{validation_prompts}.png"))
        images.append(image)
    return images

# Train and evaluate
train(train_dataloader, vae, unet, tokenizer, text_encoder, scheduler, optimizer, device, weight_dtype, num_epochs)
evaluate_model(test_dataloader, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype)

# Example validation prompts
validation_prompts = [
    "A beautiful landscape with mountains",
    "A futuristic cityscape at night",
    "An underwater coral reef scene",
    "A close-up of a robotic hand"
]
validation_images = validate_with_prompts(pipe, validation_prompts, num_inference_steps=20)