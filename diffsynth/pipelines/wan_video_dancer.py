import torch
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union, List

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dancer_dit import WanVideoDancerDiT
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder

class WanVideoDancerPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanVideoDancerDiT = None
        self.vae: WanVideoVAE = None
        
        self.units = [
            WanVideoDancerUnit_ShapeChecker(),
            WanVideoDancerUnit_NoiseInitializer(),
            WanVideoDancerUnit_PromptEmbedder(),
            WanVideoDancerUnit_ImageEmbedderCLIP(),
            WanVideoDancerUnit_PoseEmbedder(),
        ]
        self.model_fn = model_fn_wan_video_dancer

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoDancerPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_video_dancer_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        pose_video: List[Image.Image] = None,
        reference_image: Image.Image = None,
        reference_pose_image: Image.Image = None,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        cfg_scale: Optional[float] = 5.0,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"prompt": negative_prompt}
        inputs_shared = {
            "pose_video": pose_video,
            "reference_image": reference_image,
            "reference_pose_image": reference_pose_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "cfg_scale": cfg_scale,
        }
        
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(self.dit, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(self.dit, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
        
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video


class WanVideoDancerUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoDancerPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoDancerUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("latents",)
        )

    def process(self, pipe: WanVideoDancerPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"latents": noise}


class WanVideoDancerUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "prompt"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoDancerPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoDancerPipeline, prompt) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}


class WanVideoDancerUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "reference_pose_image", "height", "width"),
            output_params=("clip_feature", "clip_fea_c"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoDancerPipeline, reference_image, reference_pose_image, height, width):
        if reference_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        
        # Reference Image CLIP
        image = pipe.preprocess_image(reference_image.resize((width, height))).to(pipe.device)
        clip_feature = pipe.image_encoder.encode_image([image])
        clip_feature = clip_feature.to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # Reference Pose CLIP
        clip_fea_c = None
        if reference_pose_image is not None:
            pose_image = pipe.preprocess_image(reference_pose_image.resize((width, height))).to(pipe.device)
            clip_fea_c = pipe.image_encoder.encode_image([pose_image])
            clip_fea_c = clip_fea_c.to(dtype=pipe.torch_dtype, device=pipe.device)
            
        return {"clip_feature": clip_feature, "clip_fea_c": clip_fea_c}


class WanVideoDancerUnit_PoseEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("pose_video", "reference_image", "reference_pose_image", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("condition", "ref_x", "ref_c"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoDancerPipeline, pose_video, reference_image, reference_pose_image, height, width, tiled, tile_size, tile_stride):
        if pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        
        # Encode Pose Video -> condition
        pose_video_tensor = pipe.preprocess_video(pose_video)
        condition = pipe.vae.encode(pose_video_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        condition = condition.to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # Encode Reference Image -> ref_x
        ref_x = None
        if reference_image is not None:
            ref_img_tensor = pipe.preprocess_image(reference_image.resize((width, height))).unsqueeze(0).transpose(1, 2) # [1, C, 1, H, W]
            ref_x = pipe.vae.encode(ref_img_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            ref_x = ref_x.squeeze(2).to(dtype=pipe.torch_dtype, device=pipe.device) # [1, C, H_lat, W_lat]
            
        # Encode Reference Pose -> ref_c
        ref_c = None
        if reference_pose_image is not None:
            ref_pose_tensor = pipe.preprocess_image(reference_pose_image.resize((width, height))).unsqueeze(0).transpose(1, 2)
            ref_c = pipe.vae.encode(ref_pose_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            ref_c = ref_c.squeeze(2).to(dtype=pipe.torch_dtype, device=pipe.device)

        return {"condition": condition, "ref_x": ref_x, "ref_c": ref_c}


def model_fn_wan_video_dancer(
    dit: WanVideoDancerDiT,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    condition: Optional[torch.Tensor] = None,
    ref_x: Optional[torch.Tensor] = None,
    ref_c: Optional[torch.Tensor] = None,
    clip_fea_c: Optional[torch.Tensor] = None,
    **kwargs,
):
    # Timestep embedding
    # In WanVideoDancerDiT, forward handles timestep embedding
    
    # Context embedding
    # In WanVideoDancerDiT, forward handles context embedding
    
    # Call DiT
    x_out = dit(
        x=latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        condition=condition,
        ref_x=ref_x,
        ref_c=ref_c,
        clip_fea_c=clip_fea_c
    )
    
    return x_out
