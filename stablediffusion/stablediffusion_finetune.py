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

transform_train = transforms.Compose([
    transforms.RandomCrop(68, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #normalize
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# data prep for test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #normalize
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
def preprocess_train(example):
    image = transform_train(example["image"])
    caption = example["caption"]
    return {"pixel_values": image, "text": caption}
def preprocess_test(example):
    image = transform_test(example["image"])
    caption = example["caption"]
    return {"pixel_values": image, "text": caption}

# Apply preprocessing to each dataset
train_data = train_data.map(preprocess_train, remove_columns=["image", "caption"])
train_data.set_format(type="torch", columns=["pixel_values"], output_all_columns=True)
# val_data = val_data.map(preprocess_test, remove_columns=["image", "caption"])
# val_data.set_format(type="torch", columns=["pixel_values"])
test_data = test_data.map(preprocess_test, remove_columns=["image", "caption"])
test_data.set_format(type="torch", columns=["pixel_values"])

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
# text_encoder = pipe.text_encoder
# tokenizer = pipe.tokenizer
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
unet = pipe.unet

scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

# Optimizer
optimizer = AdamW(unet.parameters(), lr=5e-5)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze vae and text_encoder and set unet to trainable
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

# Move components to the appropriate device
vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)

# Scheduler adjustment for diffusion
def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

# Function to compute text embeddings
# def get_text_embeddings(texts, tokenizer, text_encoder):
#     inputs = tokenizer(texts, padding="max_length", return_tensors="pt", max_length=77, truncation=True)
#     return text_encoder(**inputs.to(device)).last_hidden_state
def get_text_embeddings(texts, tokenizer, text_encoder):
    inputs = tokenizer(texts, padding="max_length", return_tensors="pt", max_length=77, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return text_encoder(**inputs).last_hidden_state

def train(data_loader, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype):
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
            text_embeddings = get_text_embeddings(batch["text"][0], tokenizer, text_encoder)

            # Get model output
            timestep_embeddings = get_timestep_embedding(timesteps, unet.config.in_channels)
            model_output = unet(noisy_latents, timestep_embeddings, encoder_hidden_states=text_embeddings).sample

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
            text_embeddings = get_text_embeddings(batch["text"][0], tokenizer, text_encoder)

            # Set initial noise
            noise = torch.randn_like(latents)

            # Run the denoising loop backward
            for t in reversed(range(scheduler.config.num_train_timesteps)):
                timestep = torch.tensor([t], device=device).long()
                timestep_embeddings = get_timestep_embedding(timestep, unet.config.in_channels)
                model_output = unet(noise, timestep_embeddings, encoder_hidden_states=text_embeddings).sample
                noise = scheduler.step(model_output, timestep, noise).prev_sample

            # Decode the latents back to images
            generated_images = vae.decode(noise / vae.config.scaling_factor).sample

            # Calculate SSIM scores between original and generated images
            for real, generated in zip(batch["pixel_values"], generated_images):
                ssim_score = calculate_ssim(real, generated)
                ssim_scores.append(ssim_score)

    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim}")

# Train and evaluate
train(train_dataloader, vae, unet, tokenizer, text_encoder, scheduler, optimizer, device, weight_dtype, num_epochs)
evaluate_model(test_dataloader, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype)