"""
Image Generation Module
=======================
Diffusion-based image generation using Flux and Stable Diffusion models.
Supports multiple aspect ratios, formats, and batch generation.
"""

import os
import torch
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from diffusers import FluxPipeline, StableDiffusionPipeline, DiffusionPipeline
from huggingface_hub import login

# ==================== CONFIGURATION ====================
class GenerationConfig:
    """Configuration for image generation"""
    
    # Model paths (exact paths on system)
    STABLE_DIFFUSION_PATH = os.environ.get(
        'STABLE_DIFFUSION_PATH',
        "/mnt/myssd/models/stable-diffusion-xl"
    )
    FLUX_PATH = os.environ.get(
        'FLUX_PATH',
        "/mnt/myssd/models/flux/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/"
    )
    
    # Hugging Face authentication (if needed)
    HUGGINGFACE_TOKEN = "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"
    
    # Output directory
    OUTPUT_BASE_DIR = "pipeline_outputs/generated_images"
    
    # Generation parameters
    FLUX_INFERENCE_STEPS = 30
    FLUX_GUIDANCE_SCALE = 3.5
    FLUX_MAX_SEQUENCE_LENGTH = 512
    
    SD_INFERENCE_STEPS = 50
    SD_GUIDANCE_SCALE = 7.5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Aspect ratio mappings
    ASPECT_RATIOS = {
        "1:1": (1024, 1024),
        "16:9": (1024, 576),
        "4:3": (800, 600),
        "2:1": (1024, 512),
    }

# ==================== PIPELINE MANAGER ====================
class PipelineManager:
    """Manages loading and caching of diffusion pipelines"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.pipelines = {}
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Hugging Face if token provided"""
        if self.config.HUGGINGFACE_TOKEN:
            try:
                login(self.config.HUGGINGFACE_TOKEN)
                print("✓ Hugging Face authentication successful")
            except Exception as e:
                print(f"⚠ Hugging Face authentication failed: {e}")
    
    def load_stable_diffusion(self):
        """Load Stable Diffusion pipeline"""
        if "Stable Diffusion" in self.pipelines:
            return self.pipelines["Stable Diffusion"]
        
        print("Loading Stable Diffusion model...")
        
        try:
            if not os.path.exists(self.config.STABLE_DIFFUSION_PATH):
                raise FileNotFoundError(
                    f"Stable Diffusion model not found at {self.config.STABLE_DIFFUSION_PATH}"
                )
            
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.STABLE_DIFFUSION_PATH,
                torch_dtype=torch.float16 if self.config.DEVICE == "cuda" else torch.float32,
                local_files_only=True,
                use_safetensors=True,
            )
            
            pipe = pipe.to(self.config.DEVICE)
            
            # Enable memory optimizations
            if self.config.DEVICE == "cuda":
                pipe.enable_attention_slicing()
                if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except:
                        pass
            
            self.pipelines["Stable Diffusion"] = pipe
            print("✓ Stable Diffusion model loaded successfully")
            return pipe
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Stable Diffusion model: {e}")
    
    def load_flux(self):
        """Load Flux pipeline"""
        if "Flux" in self.pipelines:
            return self.pipelines["Flux"]
        
        print("Loading Flux model...")
        
        try:
            if not os.path.exists(self.config.FLUX_PATH):
                raise FileNotFoundError(
                    f"Flux model not found at {self.config.FLUX_PATH}"
                )
            
            if self.config.DEVICE == "cuda":
                pipe = FluxPipeline.from_pretrained(
                    self.config.FLUX_PATH,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    use_safetensors=True,
                )
                # Enable Flux-specific optimizations
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_slicing()
                if hasattr(pipe, 'enable_vae_tiling'):
                    pipe.enable_vae_tiling()
            else:
                pipe = FluxPipeline.from_pretrained(
                    self.config.FLUX_PATH,
                    torch_dtype=torch.float32,
                    local_files_only=True,
                    use_safetensors=True,
                )
                pipe = pipe.to(self.config.DEVICE)
            
            self.pipelines["Flux"] = pipe
            print("✓ Flux model loaded successfully")
            return pipe
            
        except Exception as e:
            # Try alternative loading method
            print(f"⚠ Primary Flux loading failed, trying alternative method...")
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    self.config.FLUX_PATH,
                    torch_dtype=torch.bfloat16 if self.config.DEVICE == "cuda" else torch.float32,
                    local_files_only=True,
                )
                if self.config.DEVICE == "cuda":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to(self.config.DEVICE)
                
                self.pipelines["Flux"] = pipe
                print("✓ Flux model loaded successfully (alternative method)")
                return pipe
            except Exception as e2:
                raise RuntimeError(f"Failed to load Flux model: {e2}")
    
    def get_pipeline(self, model_name: str):
        """Get pipeline by model name"""
        if model_name == "Stable Diffusion":
            return self.load_stable_diffusion()
        elif model_name == "Flux":
            return self.load_flux()
        else:
            raise ValueError(f"Unknown model: {model_name}")

# ==================== IMAGE GENERATOR ====================
class ImageGenerator:
    """Main image generation engine"""
    
    def __init__(self):
        self.config = GenerationConfig()
        self.pipeline_manager = PipelineManager(self.config)
        
        # Create output directory
        os.makedirs(self.config.OUTPUT_BASE_DIR, exist_ok=True)
    
    def _get_dimensions(self, aspect_ratio: str) -> Tuple[int, int]:
        """Convert aspect ratio string to width, height"""
        dimensions = self.config.ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
        return dimensions  # Returns (width, height)
    
    def _create_output_path(
        self,
        model_name: str,
        image_format: str,
        index: int,
        session_id: str
    ) -> str:
        """Create output file path for generated image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = "jpg" if image_format == "JPEG" else "png"
        
        filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}_{session_id[:8]}_img{index}.{extension}"
        return os.path.join(self.config.OUTPUT_BASE_DIR, filename)
    
    def _save_metadata(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        seed: int,
        width: int,
        height: int,
        **kwargs
    ):
        """Save generation metadata as JSON"""
        metadata = {
            "image_path": image_path,
            "prompt": prompt,
            "model": model_name,
            "seed": seed,
            "dimensions": {"width": width, "height": height},
            "timestamp": datetime.now().isoformat(),
            "device": self.config.DEVICE,
            **kwargs
        }
        
        metadata_path = image_path.rsplit('.', 1)[0] + "_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def generate_with_stable_diffusion(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: Optional[int] = None
    ):
        """Generate single image with Stable Diffusion"""
        pipe = self.pipeline_manager.get_pipeline("Stable Diffusion")
        
        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator(device=self.config.DEVICE).manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            result = pipe(
                prompt,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=self.config.SD_INFERENCE_STEPS,
                guidance_scale=self.config.SD_GUIDANCE_SCALE,
            )
            image = result.images[0]
        
        return image, seed
    
    def generate_with_flux(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: Optional[int] = None
    ):
        """Generate single image with Flux"""
        pipe = self.pipeline_manager.get_pipeline("Flux")
        
        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        generator = torch.Generator(device=self.config.DEVICE).manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            result = pipe(
                prompt,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=self.config.FLUX_INFERENCE_STEPS,
                guidance_scale=self.config.FLUX_GUIDANCE_SCALE,
                max_sequence_length=self.config.FLUX_MAX_SEQUENCE_LENGTH,
            )
            image = result.images[0]
        
        return image, seed
    
    def generate_images(
        self,
        prompt: str,
        num_images: int,
        aspect_ratio: str,
        model_name: str,
        image_format: str = "PNG",
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Generate multiple images based on parameters
        
        Args:
            prompt: Text prompt for generation
            num_images: Number of images to generate
            aspect_ratio: Aspect ratio (e.g., "1:1", "16:9")
            model_name: "Flux" or "Stable Diffusion"
            image_format: "PNG" or "JPEG"
            seed: Random seed (optional, will generate random if None)
        
        Returns:
            List of paths to generated images
        """
        print(f"Starting image generation: {num_images} images with {model_name}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Get dimensions
        width, height = self._get_dimensions(aspect_ratio)
        print(f"Dimensions: {width}x{height}")
        
        # Validate dimensions
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        image_paths = []
        
        for i in range(num_images):
            try:
                print(f"Generating image {i+1}/{num_images}...")
                
                # Generate image based on model
                if model_name == "Stable Diffusion":
                    image, used_seed = self.generate_with_stable_diffusion(
                        prompt, width, height, seed
                    )
                    generation_params = {
                        "inference_steps": self.config.SD_INFERENCE_STEPS,
                        "guidance_scale": self.config.SD_GUIDANCE_SCALE,
                    }
                elif model_name == "Flux":
                    image, used_seed = self.generate_with_flux(
                        prompt, width, height, seed
                    )
                    generation_params = {
                        "inference_steps": self.config.FLUX_INFERENCE_STEPS,
                        "guidance_scale": self.config.FLUX_GUIDANCE_SCALE,
                        "max_sequence_length": self.config.FLUX_MAX_SEQUENCE_LENGTH,
                    }
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Save image
                output_path = self._create_output_path(
                    model_name, image_format, i+1, session_id
                )
                
                if image_format == "JPEG":
                    image.save(output_path, quality=95, optimize=True)
                else:
                    image.save(output_path, optimize=True)
                
                print(f"✓ Saved: {output_path}")
                
                # Save metadata
                self._save_metadata(
                    output_path,
                    prompt,
                    model_name,
                    used_seed,
                    width,
                    height,
                    **generation_params
                )
                
                image_paths.append(output_path)
                
                # Clear cache
                if self.config.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"✗ Error generating image {i+1}: {e}")
                raise RuntimeError(f"Failed to generate image {i+1}: {e}")
        
        print(f"✓ Successfully generated {len(image_paths)} images")
        return image_paths

# ==================== UTILITY FUNCTIONS ====================
def batch_generate_from_prompts(
    prompts: List[str],
    model_name: str = "Flux",
    aspect_ratio: str = "1:1",
    images_per_prompt: int = 1,
    image_format: str = "PNG"
) -> dict:
    """
    Batch generate images from multiple prompts
    
    Returns:
        Dictionary mapping prompts to their generated image paths
    """
    generator = ImageGenerator()
    results = {}
    
    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Processing prompt {idx+1}/{len(prompts)}")
        print(f"{'='*60}")
        
        try:
            image_paths = generator.generate_images(
                prompt=prompt,
                num_images=images_per_prompt,
                aspect_ratio=aspect_ratio,
                model_name=model_name,
                image_format=image_format
            )
            results[prompt] = image_paths
        except Exception as e:
            print(f"Failed to generate images for prompt: {e}")
            results[prompt] = []
    
    return results