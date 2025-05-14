import time
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import psutil

# Define models and text sizes to test
models = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "gpt2",
    "facebook/bart-large"
]

text_sizes = [
    "This is a short text.",
    "This is a medium length text. " * 10,
    "This is a long text. " * 100
]


# Function to measure VRAM usage
def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB


# Function to measure RAM usage
def get_ram_usage():
    return psutil.virtual_memory().used / (1024 ** 2)  # Convert bytes to MB


# Function to perform inference and measure runtime and memory usage
def perform_inference(model_name, text, device):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move model to the specified device
    model.to(device)

    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt")

    # Measure VRAM/RAM usage before inference
    if device == "cuda":
        vram_before = get_vram_usage()
    else:
        ram_before = get_ram_usage()

    # Perform inference and measure runtime
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    end_time = time.time()

    runtime = end_time - start_time

    # Measure VRAM/RAM usage after inference
    if device == "cuda":
        vram_after = get_vram_usage()
        vram_usage = vram_after - vram_before
        return runtime, vram_usage
    else:
        ram_after = get_ram_usage()
        ram_usage = ram_after - ram_before
        return runtime, ram_usage


# Perform experiments and store results
results = []

for model_name in models:
    for text in text_sizes:
        for device in ["cuda", "cpu"]:
            runtime, memory_usage = perform_inference(model_name, text, device)
            results.append({
                "model": model_name,
                "text_length": len(text),
                "device": device,
                "runtime": runtime,
                "memory_usage": memory_usage
            })

# Print results
for result in results:
    print(result)
