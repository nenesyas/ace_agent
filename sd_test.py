from diffusers import StableDiffusionPipeline
import torch, os

os.makedirs("ace_workspace", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", device, "torch:", torch.__version__, "cuda:", torch.version.cuda)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
).to(device)

prompt = "a cute cat, digital art"
with torch.autocast(device):
    image = pipe(prompt).images[0]

out_path = os.path.join("ace_workspace", "sd_test.png")
image.save(out_path)
print("Kaydedildi:", out_path)
