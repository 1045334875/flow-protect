"""
PID (Perturbation-based Image Defense) Protection Implementation
Integrated wrapper combining core PID functionality with interface.
"""

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import AutoencoderKL

try:
    import wandb
except ImportError:
    wandb = None

from ..interfaces import ProtectionMethod


def _load_vae(pretrained_model_name_or_path: str, revision: Optional[str], device: torch.device) -> AutoencoderKL:
    """Load VAE with fallbacks for SD3/FLUX-style repos."""
    use_fp16 = device.type == "cuda"
    weight_dtype = torch.float16 if use_fp16 else torch.float32

    model_id_lower = (pretrained_model_name_or_path or "").lower()
    is_sd3 = "sd3" in model_id_lower or "stable-diffusion-3" in model_id_lower
    is_flux = "flux" in model_id_lower or "black-forest" in model_id_lower

    # Try subfolder="vae" first
    try:
        return AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            torch_dtype=weight_dtype,
        )
    except Exception:
        pass

    # SD3/FLUX fp16 variant
    if is_sd3 or is_flux:
        try:
            return AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="vae",
                revision=revision,
                torch_dtype=weight_dtype,
                variant="fp16",
            )
        except Exception:
            pass

    # Fallback: load without subfolder
    return AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        revision=revision,
        torch_dtype=weight_dtype,
    )


@dataclass
class PIDConfig:
    """Configuration for PID protection."""
    pretrained_model_name_or_path: str
    instance_data_dir: str
    output_dir: str
    revision: Optional[str] = None
    seed: Optional[int] = None
    resolution: int = 512
    center_crop: bool = False
    max_train_steps: int = 100
    dataloader_num_workers: int = 0
    eps: float = 12.75
    step_size: float = 1/255
    attack_type: str = "add"
    # W&B options
    use_wandb: bool = False
    wandb_project: str = "pid"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_log_images: bool = False


class PIDDataset(Dataset):
    """Dataset for PID protection."""

    def __init__(self, instance_data_root: str, size: int = 512, center_crop: bool = False, dtype=torch.float32):
        self.size = size
        self.center_crop = center_crop
        self.dtype = dtype  # Store the target dtype
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example['index'] = index % self.num_instance_images
        pixel_values = self.image_transforms(instance_image)
        # Convert to target dtype
        example['pixel_values'] = pixel_values.to(dtype=self.dtype)
        return example


def run_pid_protection(config: PIDConfig) -> Dict[str, Any]:
    """Run PID protection on images in instance_data_dir."""
    if config.seed is not None:
        torch.manual_seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16 if device.type == 'cuda' else torch.float32

    # W&B setup
    wandb_run = None
    if config.use_wandb and wandb is not None:
        init_kwargs = {"project": config.wandb_project}
        if config.wandb_run_name:
            init_kwargs["name"] = config.wandb_run_name
        if config.wandb_entity:
            init_kwargs["entity"] = config.wandb_entity
        if config.wandb_mode:
            init_kwargs["mode"] = config.wandb_mode
        wandb_run = wandb.init(**init_kwargs)
        wandb.config.update(vars(config), allow_val_change=True)

    # Load VAE
    vae = _load_vae(config.pretrained_model_name_or_path, config.revision, device)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # Dataset and DataLoaders creation:
    dataset = PIDDataset(
        instance_data_root=config.instance_data_dir,
        size=config.resolution,
        center_crop=config.center_crop,
        dtype=weight_dtype,  # Pass the target dtype
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=config.dataloader_num_workers,
    )

    # Attack model
    class AttackModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.epsilon = config.eps / 255
            
            # Initialize delta tensors with proper dimensions matching the dataset transforms
            # Use the dataset's transform to get the correct size AND match weight_dtype
            self.delta = []
            for i in range(len(dataset.instance_images_path)):
                # Get a sample from the dataset (already transformed to correct size and dtype)
                sample = dataset[i]['pixel_values']
                # Create delta with same shape and dtype as the transformed tensor
                delta_tensor = torch.empty_like(sample).uniform_(-self.epsilon, self.epsilon)
                self.delta.append(delta_tensor)
            
            # Make delta a proper parameter list
            self.delta = torch.nn.ParameterList([torch.nn.Parameter(d) for d in self.delta])

        def forward(self, vae_model, x, index, poison=False):
            if poison:
                # Get the delta parameter for this index
                delta_param = self.delta[index]
                delta_param.requires_grad_(True)
                # No need for dtype conversion - they already match!
                x = x + delta_param
            input_x = 2 * x - 1
            return vae_model.encode(input_x.to(device))

    attackmodel = AttackModel()
    # Use SGD optimizer with the delta parameters
    optimizer = torch.optim.SGD(attackmodel.delta.parameters(), lr=0)
    progress_bar = tqdm(range(config.max_train_steps), desc="PID Steps")

    os.makedirs(config.output_dir, exist_ok=True)

    # Optimization loop
    for step in progress_bar:
        total_loss = 0.0
        for batch in dataloader:
            # Save images periodically
            if step % 25 == 0:
                to_image = transforms.ToPILImage()
                for i in range(0, len(dataset.instance_images_path)):
                    img = dataset[i]['pixel_values']
                    # Add delta and ensure proper range for PIL conversion
                    perturbed_img = img + attackmodel.delta[i].detach()
                    # Convert to float32 for PIL (PIL expects float32)
                    perturbed_img = perturbed_img.to(dtype=torch.float32)
                    perturbed_img = torch.clamp(perturbed_img, 0, 1)
                    img_pil = to_image(perturbed_img)
                    img_pil.save(os.path.join(config.output_dir, f"{i}.png"))
                if wandb_run and config.wandb_log_images:
                    sample_path = os.path.join(config.output_dir, "0.png")
                    if os.path.exists(sample_path):
                        wandb.log({"sample_image": wandb.Image(sample_path)}, step=step)

            # Compute loss
            batch_index = batch['index'][0].item()  # Get scalar index
            clean_embedding = attackmodel(vae, batch['pixel_values'], batch_index, False)
            poison_embedding = attackmodel(vae, batch['pixel_values'], batch_index, True)
            clean_latent = clean_embedding.latent_dist
            poison_latent = poison_embedding.latent_dist

            if config.attack_type == 'var':
                loss = F.mse_loss(clean_latent.std, poison_latent.std, reduction="mean")
            elif config.attack_type == 'mean':
                loss = F.mse_loss(clean_latent.mean, poison_latent.mean, reduction="mean")
            elif config.attack_type == 'KL':
                sigma_2, mu_2 = poison_latent.std, poison_latent.mean
                sigma_1, mu_1 = clean_latent.std, clean_latent.mean
                KL_diver = torch.log(sigma_2 / sigma_1) - 0.5 + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2 ** 2)
                loss = KL_diver.flatten().mean()
            elif config.attack_type == 'latent_vector':
                clean_vector = clean_latent.sample()
                poison_vector = poison_latent.sample()
                loss = F.mse_loss(clean_vector, poison_vector, reduction="mean")
            elif config.attack_type == 'add':
                loss_2 = F.mse_loss(clean_latent.std, poison_latent.std, reduction="mean")
                loss_1 = F.mse_loss(clean_latent.mean, poison_latent.mean, reduction="mean")
                loss = loss_1 + loss_2
            elif config.attack_type == 'add-log':
                loss_1 = F.mse_loss(clean_latent.var.log(), poison_latent.var.log(), reduction="mean")
                loss_2 = F.mse_loss(clean_latent.mean, poison_latent.mean, reduction='mean')
                loss = loss_1 + loss_2
            else:
                loss = F.mse_loss(clean_latent.mean, poison_latent.mean, reduction="mean")

            loss.backward()

            # Perform PGD update on the loss
            batch_index = batch['index'][0].item()
            delta_param = attackmodel.delta[batch_index]
            with torch.no_grad():
                # Update the parameter data directly - no dtype conversion needed
                grad_sign = delta_param.grad.sign() if delta_param.grad is not None else 0
                delta_param.data += grad_sign * config.step_size
                delta_param.data = torch.clamp(delta_param.data, -attackmodel.epsilon, attackmodel.epsilon)
                
                # Only clamp to pixel values if dimensions and dtypes match exactly
                pixel_values = batch['pixel_values'][0].detach()  # Keep original dtype and device
                if delta_param.data.dim() == pixel_values.dim() and delta_param.data.dtype == pixel_values.dtype:
                    delta_param.data = torch.clamp(delta_param.data, -pixel_values, 1 - pixel_values)
            
            # Clear gradients
            optimizer.zero_grad()

            total_loss += loss.detach().cpu()

        logs = {"loss": total_loss.item()}
        progress_bar.set_postfix(**logs)
        if wandb_run:
            wandb.log(logs, step=step)

    # Final save
    to_image = transforms.ToPILImage()
    output_paths = []
    for i in range(len(dataset.instance_images_path)):
        img = dataset[i]['pixel_values']
        # Add the delta perturbation
        perturbed_img = img + attackmodel.delta[i].detach()
        # Clamp to valid range [0, 1]
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        # Convert to PIL image
        img_pil = to_image(perturbed_img)
        save_path = os.path.join(config.output_dir, f"{i}.png")
        img_pil.save(save_path)
        output_paths.append(save_path)

    if wandb_run:
        wandb.finish()

    return {"status": "success", "output_paths": output_paths}


# CLI support
def parse_args(input_args=None):
    """Parse command line arguments for PID."""
    parser = argparse.ArgumentParser(description="PID Protection Script")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="pid_output")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--eps", type=float, default=12.75)
    parser.add_argument("--step_size", type=float, default=1/255)
    parser.add_argument("--attack_type", type=str, default="add",
                        choices=['var', 'mean', 'KL', 'add-log', 'latent_vector', 'add'])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pid")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_log_images", action="store_true")

    return parser.parse_args(input_args)


def main(args):
    """Main function for CLI usage."""
    config = PIDConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        instance_data_dir=args.instance_data_dir,
        output_dir=args.output_dir,
        revision=args.revision,
        seed=args.seed,
        resolution=args.resolution,
        center_crop=args.center_crop,
        max_train_steps=args.max_train_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        eps=args.eps,
        step_size=args.step_size,
        attack_type=args.attack_type,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_log_images=args.wandb_log_images,
    )
    return run_pid_protection(config)


class PIDProtection(ProtectionMethod):
    """PID Protection wrapper using local implementation."""
    
    MODEL_MAP = {
        "sd1.4": "CompVis/stable-diffusion-v1-4",
        "sd1.5": "runwayml/stable-diffusion-v1-5",
        "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
        "flux": "black-forest-labs/FLUX.1-dev",
    }

    def protect(
        self,
        input_image_path: str,
        output_image_path: str,
        model_name: str = "sd1.4",
        prompt: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply PID protection to an image.
        
        Args:
            input_image_path: Path to input image.
            output_image_path: Path to save protected image.
            model_name: Model identifier (sd1.4, sd1.5, sd3, flux).
            prompt: Unused for PID (kept for interface compatibility).
            **kwargs: Additional PID config options.
            
        Returns:
            Dict with status and metadata.
        """
        hf_model_id = self.MODEL_MAP.get(model_name.lower(), model_name)

        # PID works on a directory of images, so we use temp dirs
        with tempfile.TemporaryDirectory() as temp_input_dir, \
             tempfile.TemporaryDirectory() as temp_output_dir:
            
            # Copy input image to temp input dir
            img_name = os.path.basename(input_image_path)
            shutil.copy(input_image_path, os.path.join(temp_input_dir, img_name))

            # Build config
            config = PIDConfig(
                pretrained_model_name_or_path=hf_model_id,
                instance_data_dir=temp_input_dir,
                output_dir=temp_output_dir,
                resolution=kwargs.get("resolution", 512),
                center_crop=kwargs.get("center_crop", False),
                max_train_steps=kwargs.get("max_train_steps", 100),
                dataloader_num_workers=kwargs.get("dataloader_num_workers", 0),
                eps=kwargs.get("eps", 12.75),
                step_size=kwargs.get("step_size", 1/255),
                attack_type=kwargs.get("attack_type", "add"),
                seed=kwargs.get("seed"),
                revision=kwargs.get("revision"),
                use_wandb=kwargs.get("use_wandb", False),
                wandb_project=kwargs.get("wandb_project", "pid"),
                wandb_run_name=kwargs.get("wandb_run_name"),
                wandb_entity=kwargs.get("wandb_entity"),
                wandb_mode=kwargs.get("wandb_mode"),
                wandb_log_images=kwargs.get("wandb_log_images", False),
            )

            try:
                result = run_pid_protection(config)
                
                if result.get("status") != "success":
                    return result
                
                # Find the output file (PID saves as 0.png for single image)
                output_files = os.listdir(temp_output_dir)
                if not output_files:
                    return {"status": "failed", "error": "No output file generated"}
                
                # Move the first output to target path
                generated_file = os.path.join(temp_output_dir, output_files[0])
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                shutil.move(generated_file, output_image_path)
                
                return {
                    "status": "success", 
                    "model": hf_model_id,
                    "output": output_image_path,
                    "method": "pid",
                    "parameters": {
                        "max_train_steps": config.max_train_steps,
                        "eps": config.eps,
                        "resolution": config.resolution,
                        "attack_type": config.attack_type
                    }
                }
                
            except Exception as e:
                return {"status": "failed", "method": "pid", "error": str(e)}