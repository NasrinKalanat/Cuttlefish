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
#
# weight_dtype=torch.float32
# # Load Stable Diffusion pipeline
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", dtype=weight_dtype)
# # pipe =pip.to('cuda')
#
# tokenizer = pipe.tokenizer
# # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#
# resolution=(256,256)
# random_flip=True
# center_crop=False
#
# transform_train = transforms.Compose(
#     [
#         transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
#         transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )
# # data prep for test set
# transform_test = transforms.Compose([
#     transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#     transforms.ToTensor(),
#     #normalize
#     transforms.Normalize([0.5], [0.5])
#     ])
#
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
#
# # Preprocess dataset with text captions
# def preprocess_train(examples):
#     image = [transform_train(example.convert("RGB")) for example in examples["image"]]
#     caption = tokenizer([example[0] for example in examples["caption"]], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
#     return {"pixel_values": image, "caption_ids": caption['input_ids'], "caption_attention_mask": caption['attention_mask']}
#
# def preprocess_test(examples):
#     image = [transform_test(example.convert("RGB")) for example in examples["image"]]
#     caption = [example[0] for example in examples["caption"]]
#     return {"pixel_values": image, "prompt": caption}
#
# # Apply preprocessing to each dataset
# train_data.set_transform(preprocess_train, output_all_columns=True)
# # train_data.set_format(type="torch", columns=["pixel_values", "caption_ids", "caption_attention_mask"], output_all_columns=True)
# # val_data.set_transform(preprocess_test, output_all_columns=True)
# # val_data.set_format(type="torch", columns=["pixel_values"])
# test_data.set_transform(preprocess_test, output_all_columns=True)
# # test_data.set_format(type="torch", columns=["pixel_values", "caption_ids", "caption_attention_mask"])
#
# # Set up DataLoaders
# train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
# # val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False)
# test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)
#
# # Extract individual components for training
# vae = pipe.vae
# # tokenizer = pipe.tokenizer
# text_encoder = pipe.text_encoder
# # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
# unet = pipe.unet
#
# # scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
# scheduler = pipe.scheduler
#
# # Optimizer
# optimizer = AdamW(unet.parameters(), lr=1e-4)
#
# # Training loop
# num_epochs = 100
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Freeze vae and text_encoder and set unet to trainable
# vae.requires_grad_(False)
# text_encoder.requires_grad_(False)
#
# # Move components to the appropriate device
# vae.to(device, dtype=weight_dtype)
# text_encoder.to(device, dtype=weight_dtype)
# unet.to(device, dtype=weight_dtype)
#
# # Function to compute text embeddings
# def get_text_embeddings(inputs, text_encoder):
#     # inputs = {key: value.to(device) for key, value in inputs.items()}
#     return text_encoder(inputs["caption_ids"].to(device), inputs["caption_attention_mask"].to(device)).last_hidden_state
#
# def train(data_loader, vae, unet, tokenizer, text_encoder, scheduler, optimizer, device, weight_dtype, num_epochs):
#     for epoch in range(num_epochs):
#         unet.requires_grad_(True)
#         unet.train()
#         train_loss = 0
#         progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
#
#         for batch in progress_bar:
#             # Prepare inputs
#             latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(device)).latent_dist.sample()
#             latents = latents * vae.config.scaling_factor  # Scaling factor
#
#             noise = torch.randn_like(latents)
#             timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
#
#             noisy_latents = scheduler.add_noise(latents, noise, timesteps)
#             text_embeddings = get_text_embeddings(batch, text_encoder)
#
#             # Get model output
#             model_output = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
#
#             # Compute loss (switch to float32 for stability)
#             loss = mse_loss(model_output.float(), noise.float(), reduction="mean")
#             train_loss += loss.item()
#
#             # Optimization step
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             progress_bar.set_postfix({"Loss": train_loss / len(progress_bar)})
#
#         print(f"Epoch {epoch+1}: Avg Loss = {train_loss / len(train_dataloader)}")
#
# ### Evaluation Function
#
# def calculate_ssim(img1, img2):
#     """Calculate Structural Similarity Index (SSIM) between two images."""
#     img1 = img1.permute(1, 2, 0).numpy()
#     img2 = img2.permute(1, 2, 0).numpy()
#     return ssim(img1, img2, multichannel=True)
#
#
# def resize_generated_image(image, size):
#     """Resize generated images to match the size of the real image."""
#     pil_image = Image.fromarray(image)  # Convert back to PIL Image
#     return np.array(pil_image.resize((size[0], size[1]), Image.Resampling.LANCZOS))
# def evaluate_model(data_loader, pipeline, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype, num_inference_steps):
#     """Evaluate the Stable Diffusion model on a given dataset."""
#     unet.eval()
#     pipeline.to("cuda")
#     ssim_scores = []
#     generator = torch.Generator(device=device).manual_seed(42)
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Evaluating"):
#             generated_images = []
#             with autocast(device_type="cuda", dtype=weight_dtype):
#                 # Generate images one prompt at a time
#                 for prompt in batch["prompt"]:
#                     image = pipeline(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
#                     generated_images.append(image)
#             # # Encode the images to latents
#             # latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(device)).latent_dist.sample()
#             # latents = latents * vae.config.scaling_factor
#             #
#             # # Generate captions to text embeddings
#             # text_embeddings = get_text_embeddings(batch, text_encoder)
#             #
#             # # Set initial noise
#             # noise = torch.randn_like(latents)
#             #
#             # # Run the denoising loop backward
#             # for t in reversed(range(num_inference_steps)):
#             #     timestep = torch.tensor([t], device=device).long()
#             #     model_output = unet(noise, timestep, encoder_hidden_states=text_embeddings).sample
#             #     noise = scheduler.step(model_output, timestep, noise).prev_sample
#             #
#             # # Decode the latents back to images
#             # generated_images = vae.decode(noise / vae.config.scaling_factor).sample
#             torch.cuda.empty_cache()
#
#             # Calculate SSIM scores between original and generated images
#             for real, generated in zip(batch["pixel_values"], generated_images):
#                 real = (real * 0.5 + 0.5).cpu().numpy().transpose(1, 2, 0) * 255  # Denormalize and convert to uint8
#                 generated = np.array(generated)
#                 real_size = (real.shape[1], real.shape[0])
#                 generated = resize_generated_image(generated, real_size)
#                 ssim_score = ssim(real, generated, win_size=3, data_range=1.0, multichannel=True)
#                 ssim_scores.append(ssim_score)
#             print(ssim_score)
#             break
#
#     avg_ssim = np.mean(ssim_scores)
#     print(f"Average SSIM: {avg_ssim}")
#
# def validate_with_prompts(pipeline, validation_prompts, num_inference_steps=20, output_dir=""):
#     """Generate and display images for a list of validation prompts."""
#     unet.eval()
#     pipeline.to("cuda")
#     images = []
#     generator = torch.Generator(device=device).manual_seed(42)
#     for prompt in validation_prompts:
#         with autocast(device_type="cuda", dtype=weight_dtype):
#             image = pipeline(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
#             image.save(os.path.join(output_dir, f"validation_output_{prompt}.png"))
#         images.append(image)
#     return images
#
# # Train and evaluate
# train(train_dataloader, vae, unet, tokenizer, text_encoder, scheduler, optimizer, device, weight_dtype, num_epochs)
# evaluate_model(test_dataloader, pipe, vae, unet, tokenizer, text_encoder, scheduler, device, weight_dtype, num_inference_steps=20)
#
# # Example validation prompts
# validation_prompts = [
#     "A beautiful landscape with mountains",
#     "A futuristic cityscape at night",
#     "An underwater coral reef scene",
#     "A close-up of a robotic hand"
# ]
# validation_images = validate_with_prompts(pipe, validation_prompts, num_inference_steps=20)

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
import torchvision.transforms as transforms
# import torch.utils.checkpoint
# import transformers
from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
# from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import random

max_train_steps=10
initial_global_step=0
output_dir="out"
logging_dir="log"
gradient_accumulation_steps=4
mixed_precision="fp16"
report_to="tensorboard"
num_train_epochs=100
noise_offset=0
input_perturbation=0
seed=None
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
revision=None
prediction_type=None
dream_training=True
dream_detail_preservation=1.0
snr_gamma=None
gradient_accumulation_steps=1
learning_rate=1e-5
adam_beta1=0.9
adam_beta2=0.999
adam_weight_decay=1e-2
adam_epsilon=1e-08
train_batch_size=16
lr_schedule="constant"
lr_warmup_steps=0
variant=None
non_ema_revision=None
caption_column="caption"
image_column="image"
resolution=32
random_flip=True
center_crop=False
dataloader_num_workers=4
max_train_samples=10
max_grad_norm=1.0


# Disable Tensor Cores
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


dataset = load_dataset("nlphuji/flickr30k")
split = dataset['test'].train_test_split(test_size=0.2, seed=42)
test_data=split['test']
split = split['train'].train_test_split(test_size=0.2, seed=42)
# val_data=split['test']
train_data=split['train']
dataset=split

# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# Preprocessing the datasets.
train_transforms = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples

accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    mixed_precision=mixed_precision,
    log_with=report_to,
    project_config=accelerator_project_config,
)

with accelerator.main_process_first():
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

# DataLoaders creation:
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=train_batch_size,
    num_workers=dataloader_num_workers,
)

optimizer_cls = torch.optim.AdamW

if seed is not None:
    set_seed(seed)

# Load scheduler, tokenizer and models.
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
)


text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
)

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", revision=non_ema_revision
)

# Freeze vae and text_encoder and set unet to trainable
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

# For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
# as these weights are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
    mixed_precision = accelerator.mixed_precision
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
    mixed_precision = accelerator.mixed_precision

# Move text_encode and vae to gpu and cast to weight_dtype
text_encoder.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)
unet.to(accelerator.device, dtype=weight_dtype)

optimizer = optimizer_cls(
    unet.parameters(),
    lr=learning_rate,
    betas=(adam_beta1, adam_beta2),
    weight_decay=adam_weight_decay,
    eps=adam_epsilon,
)

lr_scheduler = get_scheduler(
    lr_schedule,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
    num_training_steps=max_train_steps * accelerator.num_processes,
)

global_step = 0
first_epoch = 0

progress_bar = tqdm(
    range(0, max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)

for epoch in range(first_epoch, num_train_epochs):
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        batch["pixel_values"]=batch["pixel_values"].to('cuda')
        batch["input_ids"]=batch["input_ids"].to('cuda')
        with accelerator.accumulate(unet):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )
            if input_perturbation:
                new_noise = noise + input_perturbation * torch.randn_like(noise)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if input_perturbation:
                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

            # Get the target for loss depending on the prediction type
            if prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # if dream_training:
            #     noisy_latents, target = compute_dream_and_update_latents(
            #         unet,
            #         noise_scheduler,
            #         timesteps,
            #         noise,
            #         noisy_latents,
            #         target,
            #         encoder_hidden_states,
            #         dream_detail_preservation,
            #     )

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            if snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # # Checks if the accelerator has performed an optimization step behind the scenes
        # if accelerator.sync_gradients:
        #     if args.use_ema:
        #         ema_unet.step(unet.parameters())
        #     progress_bar.update(1)
        #     global_step += 1
        #     accelerator.log({"train_loss": train_loss}, step=global_step)
        #     train_loss = 0.0
        #
        #     if global_step % args.checkpointing_steps == 0:
        #         if accelerator.is_main_process:
        #             # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        #             if args.checkpoints_total_limit is not None:
        #                 checkpoints = os.listdir(args.output_dir)
        #                 checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        #                 checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        #
        #                 # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        #                 if len(checkpoints) >= args.checkpoints_total_limit:
        #                     num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
        #                     removing_checkpoints = checkpoints[0:num_to_remove]
        #
        #                     logger.info(
        #                         f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        #                     )
        #                     logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
        #
        #                     for removing_checkpoint in removing_checkpoints:
        #                         removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
        #                         shutil.rmtree(removing_checkpoint)
        #
        #             save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        #             accelerator.save_state(save_path)
        #             logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= max_train_steps:
            break

#     if accelerator.is_main_process:
#         if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
#             if args.use_ema:
#                 # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
#                 ema_unet.store(unet.parameters())
#                 ema_unet.copy_to(unet.parameters())
#             log_validation(
#                 vae,
#                 text_encoder,
#                 tokenizer,
#                 unet,
#                 args,
#                 accelerator,
#                 weight_dtype,
#                 global_step,
#             )
#             if args.use_ema:
#                 # Switch back to the original UNet parameters.
#                 ema_unet.restore(unet.parameters())
#
# # Create the pipeline using the trained modules and save it.
# accelerator.wait_for_everyone()
# if accelerator.is_main_process:
#     unet = unwrap_model(unet)
#     if args.use_ema:
#         ema_unet.copy_to(unet.parameters())
#
#     pipeline = StableDiffusionPipeline.from_pretrained(
#         args.pretrained_model_name_or_path,
#         text_encoder=text_encoder,
#         vae=vae,
#         unet=unet,
#         revision=args.revision,
#         variant=args.variant,
#     )
#     pipeline.save_pretrained(args.output_dir)
#
#     # Run a final round of inference.
#     images = []
#     if args.validation_prompts is not None:
#         logger.info("Running inference for collecting generated images...")
#         pipeline = pipeline.to(accelerator.device)
#         pipeline.torch_dtype = weight_dtype
#         pipeline.set_progress_bar_config(disable=True)
#
#         if args.enable_xformers_memory_efficient_attention:
#             pipeline.enable_xformers_memory_efficient_attention()
#
#         if args.seed is None:
#             generator = None
#         else:
#             generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
#
#         for i in range(len(args.validation_prompts)):
#             with torch.autocast("cuda"):
#                 image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
#             images.append(image)
#
#     if args.push_to_hub:
#         save_model_card(args, repo_id, images, repo_folder=args.output_dir)
#         upload_folder(
#             repo_id=repo_id,
#             folder_path=args.output_dir,
#             commit_message="End of training",
#             ignore_patterns=["step_*", "epoch_*"],
#         )

accelerator.end_training()
