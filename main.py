import time
import torch
from transformers import pipeline
import psutil
import nvidia_smi
from torch.profiler import profile, record_function, ProfilerActivity

import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define models and text sizes to test
model_name = "bart-large-mnli"  # Model for zero-shot classification

text_sizes = [
    "This is a short text.",
    "This is a medium length text. " * 10,
    "This is a long text. " * 100
]

batch_sizes = [1, 8, 32]

# Function to measure VRAM usage
def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB

# Function to measure RAM usage
def get_ram_usage():
    return psutil.virtual_memory().used / (1024 ** 2)  # Convert bytes to MB

# Function to perform inference and measure runtime and memory usage
def perform_inference(model_name, texts, batch_size, device):
    # Load pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    # Measure VRAM/RAM usage before inference
    if device == "cuda":
        vram_before = get_vram_usage()
    else:
        ram_before = get_ram_usage()

    # Perform inference and measure runtime
    start_time = time.time()
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                outputs = classifier(texts, candidate_labels=["label1", "label2", "label3"])
    end_time = time.time()

    runtime = end_time - start_time

    # Measure VRAM/RAM usage after inference
    if device == "cuda":
        vram_after = get_vram_usage()
        vram_usage = vram_after - vram_before
        return runtime, vram_usage, prof.key_averages().table(sort_by="cuda_time_total")
    else:
        ram_after = get_ram_usage()
        ram_usage = ram_after - ram_before
        return runtime, ram_usage, prof.key_averages().table(sort_by="cpu_time_total")

# Perform experiments and store results
results = []

for text in text_sizes:
    for batch_size in batch_sizes:
        for device in ["cuda", "cpu"]:
            texts = [text] * batch_size
            runtime, memory_usage, profiler_table = perform_inference(model_name, texts, batch_size, device)
            results.append({
                "model": model_name,
                "text_length": len(text),
                "batch_size": batch_size,
                "device": device,
                "runtime": runtime,
                "memory_usage": memory_usage,
                "profiler_table": profiler_table
            })

# Print results
for result in results:
    print(result)
