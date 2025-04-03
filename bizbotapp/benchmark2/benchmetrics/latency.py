import time
import torch

def time_inference(model, tokenizer, prompt, max_tokens=100):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2)
    latency = time.time() - start
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return latency, output_text
