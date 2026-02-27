import torch
import lpips
import numpy as np

# Try to import CLIP from local modules first, fallback to pip install
try:
    import clip
except ImportError:
    # Try importing from local Diffusion-PID-Protection module
    import sys
    import os
    clip_path = os.path.join(os.path.dirname(__file__), '../../modules/Diffusion-PID-Protection')
    if os.path.exists(clip_path):
        sys.path.insert(0, clip_path)
        try:
            import clip
        except ImportError:
            clip = None
    else:
        clip = None

from PIL import Image
from torchvision import transforms
from typing import Dict, Optional, List, Union
import os

# Try importing other metrics, handle if not installed
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None

from ..interfaces import Evaluator

class MetricEvaluator(Evaluator):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize LPIPS
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        except:
            print("Warning: LPIPS not available.")
            self.lpips_fn = None
        
        # Initialize CLIP
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        except:
            print("Warning: CLIP not available.")
            self.clip_model = None
            self.clip_preprocess = None
            
        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _load_tensor(self, path_or_img: Union[str, Image.Image]):
        if isinstance(path_or_img, str):
            img = Image.open(path_or_img).convert('RGB')
        else:
            img = path_or_img.convert('RGB')
        return self.transform_tensor(img).unsqueeze(0).to(self.device)

    def _load_pil(self, path_or_img: Union[str, Image.Image]):
        if isinstance(path_or_img, str):
            return Image.open(path_or_img).convert('RGB')
        return path_or_img.convert('RGB')

    def calculate_psnr(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        Based on AtkPDM and Diff-Protect implementations.
        """
        a = np.array(img1).astype(np.float32)
        b = np.array(img2).astype(np.float32)
        mse = np.mean((a - b) ** 2)
        if mse == 0:
            return 100.0
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def calculate_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        Based on AtkPDM implementation (using skimage).
        """
        if ssim is None:
            return -1.0
        
        # Convert to grayscale for standard SSIM, or use multichannel=True
        # AtkPDM converts to Gray
        a_gray = np.array(img1.convert('L'))
        b_gray = np.array(img2.convert('L'))
        
        return ssim(a_gray, b_gray, data_range=255)

    def calculate_clip_score(self, image: Image.Image, text: str) -> float:
        """
        Calculate CLIP Score (cosine similarity between image and text)
        Used in PID evaluation (IQS concept uses similar approach).
        """
        if self.clip_model is None:
            return 0.0
            
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
            
        return similarity

    def calculate_lpips(self, path1, path2) -> float:
        """
        Calculate LPIPS distance
        Used in AtkPDM and Diff-Protect.
        """
        if self.lpips_fn is None:
            return -1.0
            
        t1 = self._load_tensor(path1)
        t2 = self._load_tensor(path2)
        
        with torch.no_grad():
            dist = self.lpips_fn(t1, t2).item()
        return dist

    def evaluate_protection(self, original_image_path: str, protected_image_path: str) -> Dict[str, float]:
        """
        Evaluate Protection Quality.
        Metrics from modules:
        - LPIPS (Perceptual similarity) - Lower is better (images look similar)
        - SSIM (Structural similarity) - Higher is better
        - PSNR (Pixel fidelity) - Higher is better
        """
        metrics = {}
        
        img_orig = self._load_pil(original_image_path)
        img_prot = self._load_pil(protected_image_path)
        
        # LPIPS
        metrics['lpips'] = self.calculate_lpips(original_image_path, protected_image_path)
        
        # SSIM
        metrics['ssim'] = self.calculate_ssim(img_orig, img_prot)
        
        # PSNR
        metrics['psnr'] = self.calculate_psnr(img_orig, img_prot)
        
        return metrics

    def evaluate_editing(self, original_image_path: str, edited_image_path: str, target_prompt: str) -> Dict[str, float]:
        """
        Evaluate Editing Quality.
        Metrics:
        - CLIP Score (Text-Image alignment) - Higher is better
        - LPIPS (Structure preservation vs Original/Protected) - Depends on goal, usually want some structure preserved.
        """
        metrics = {}
        
        img_edit = self._load_pil(edited_image_path)
        
        # CLIP Score (Alignment with target prompt)
        metrics['clip_score'] = self.calculate_clip_score(img_edit, target_prompt)
        
        # Structure distance from original (how much did we change content?)
        metrics['structure_dist_original'] = self.calculate_lpips(original_image_path, edited_image_path)
        
        return metrics

    def calculate_fid(self, real_path: str, fake_path: str) -> float:
        """
        Calculate FID (Frechet Inception Distance).
        Requires directories of images, not single images.
        Used in PID.
        """
        if calculate_metrics is None:
            return -1.0
            
        # Check if paths are directories
        if not (os.path.isdir(real_path) and os.path.isdir(fake_path)):
            print("FID requires directory paths.")
            return -1.0
            
        metrics_dict = calculate_metrics(
            input1=real_path,
            input2=fake_path,
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=False,
        )
        return metrics_dict['frechet_inception_distance']
