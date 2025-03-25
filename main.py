from sentimental import analyze
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch
from PIL import Image
import requests

# Device setup
device = "cpu"

# Load Florence-2 model and processor
model_name = "gokaygokay/Florence-2-Flux-Large"
model = AutoModelForCausalLM.from_pretrained("gokaygokay/Florence-2-Flux-Large", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Function to perform OCR using Florence-2
def run_ocr(image):
    task_prompt = "<OCR>"

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,  # OCR output is usually shorter
        num_beams=3,
        repetition_penalty=1.10,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer["<OCR>"]

# Load image
image_path = "test_images/test2.jpeg"
image = Image.open(image_path)

# Perform OCR
ocr_text = run_ocr(image)
print(analyze(ocr_text))
