import torch
import os
from PIL import Image
import numpy as np
from diffsynth import ModelConfig
from diffsynth.pipelines.wan_video_dancer import WanVideoDancerPipeline
from diffsynth.utils.data import save_video, VideoData
from diffsynth.utils.controlnet.annotator import Annotator


def get_pose_video(video_path, annotator):
    video_data = VideoData(video_file=video_path)
    frames = video_data.raw_data()

    pose_frames = []
    for frame in frames:
        pose_frame = annotator(frame)
        pose_frames.append(pose_frame)
    return pose_frames


def run_inference():
    # Configuration
    # Define models
    model_config_list = [
        ModelConfig(
            model_id="MCG-NJU/SteadyDancer-14B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            download_source="huggingface",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="Wan2.1_VAE.pth",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ),
    ]

    pipe = WanVideoDancerPipeline.from_pretrained(
        model_configs=model_config_list,
        device="cuda",
        torch_dtype=torch.bfloat16,
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
    )

    # Load inputs
    ref_image_path = "test_image.jpg"
    pose_video_path = "test_pose.mp4"

    if not os.path.exists(ref_image_path):
        print(f"Please provide a reference image at {ref_image_path}")
        # Create a dummy image for testing
        Image.new("RGB", (832, 480), color="red").save(ref_image_path)

    if not os.path.exists(pose_video_path):
        print(f"Please provide a pose video at {pose_video_path}")
        # Create a dummy video for testing
        import imageio

        writer = imageio.get_writer(pose_video_path, fps=30)
        for i in range(30):
            writer.append_data(np.zeros((480, 832, 3), dtype=np.uint8))
        writer.close()

    ref_image = Image.open(ref_image_path).convert("RGB")

    # Extract pose
    # Note: Ensure you have controlnet_aux installed and models downloaded
    try:
        annotator = Annotator("openpose", device="cuda")
        print("Extracting pose from video...")
        pose_video = get_pose_video(pose_video_path, annotator)
    except Exception as e:
        print(f"Failed to load annotator or extract pose: {e}")
        print("Using dummy pose video.")
        pose_video = [Image.new("RGB", (832, 480), color="black") for _ in range(30)]

    # The first frame of pose video is used as reference pose image
    ref_pose_image = pose_video[0]

    # Run inference
    video = pipe(
        prompt="A person dancing",
        negative_prompt="blur, low quality",
        pose_video=pose_video,
        reference_image=ref_image,
        reference_pose_image=ref_pose_image,
        num_frames=len(pose_video),
        height=480,
        width=832,
        cfg_scale=5.0,
        num_inference_steps=20,
    )

    # Save video
    save_video(video, "output_dancer.mp4", fps=30)
    print("Video saved to output_dancer.mp4")


if __name__ == "__main__":
    run_inference()
