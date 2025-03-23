from sentimental import analyze
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch
from PIL import Image
import requests
from flask import Flask, request, jsonify, render_template
import io

# Flask app setup
app = Flask(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return parsed_answer.get("<OCR>","No text detected")

# Flask Funtion 
@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    sentiment = ""
    error = None  # Initialize error variable

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image uploaded"
        else:
            file = request.files["image"]
            image = Image.open(io.BytesIO(file.read()))
            
            extracted_text = run_ocr(image)
            sentiment = analyze(extracted_text)

    return render_template("index.html", extracted_text=extracted_text, sentiment=sentiment, error=error)

if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0", port = 5000)





# Load image
# image_path = "/home/jinwoo/Desktop/handwriting_recognition/test2.jpeg"
# image = Image.open(image_path)
# Perform OCR
# ocr_text = run_ocr(image)
# print(analyze(ocr_text))
