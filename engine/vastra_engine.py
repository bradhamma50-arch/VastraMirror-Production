import modal
from modal import Image, Volume, Stub

stub = Stub("vastra-engine")
idm_vton_weights = Volume.from_name("idm_vton_weights", create_if_missing=True)

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download("yisol/IDM-VTON", local_dir="/weights")

vton_image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "gradio",
        "gradio_client"
    )
    .run_function(download_model, volumes={"/weights": idm_vton_weights})
)

@stub.cls(
    gpu="L4",
    image=vton_image,
    volumes={"/weights": idm_vton_weights},
    container_idle_timeout=300
)
class VastraModel:
    def __enter__(self):
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
        
        # Load in FP16 for stability; FP8 can be complex in generic environments
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "/weights",
            torch_dtype=torch.float16, 
            use_safetensors=True
        ).to("cuda")
        
        # Use SDXL-Lightning scheduler for fast inference
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, 
            timestep_spacing="trailing"
        )

    @modal.method()
    def process_tryon(self, human_img_url, garment_img_url):
        # 15 inference steps as requested for <12s processing
        image = self.pipe(
            prompt="high quality virtual try-on",
            num_inference_steps=15,
            guidance_scale=0.0
        ).images[0]
        return image

@stub.function()
@modal.asgi_app()
def gradio_api():
    import gradio as gr
    
    def try_on(human, garment):
        model = VastraModel()
        return model.process_tryon.remote(human, garment)

    demo = gr.Interface(
        fn=try_on,
        inputs=[gr.Textbox(label="Human Image URL"), gr.Textbox(label="Garment Image URL")],
        outputs=gr.Image(label="Result"),
        title="VastraMirror AI Engine"
    )
    return gr.routes.App.create_app(demo)
