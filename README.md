# Edge AI Deployment Workflow: MobileNetV2 with ONNX Runtime

I'm learning how to take an AI model from a notebook and make it practical for real devices.

This project walks through that workflow end to end: start with a pretrained PyTorch model, measure it, convert it to ONNX, shrink it with INT8 quantization, and test it under edge-like limits. Each step is measured with the same three numbers: **size, speed, and accuracy**.


**Result in one line:** the model became about **3.6x smaller** and about **9x faster** than the PyTorch baseline, with **no accuracy change** on my 200-image test.

<img width="1500" height="600" alt="benchmark_chart" src="https://github.com/user-attachments/assets/d4280b0d-d80e-4122-a513-d7c056e8aadc" />

---

## Cloud AI vs Edge AI

**Cloud AI** runs the model on powerful remote servers. The device (phone, camera, sensor) sends its data over the internet and gets the answer back.

**Edge AI** runs the model on the device itself, or on a small computer right next to it. The data never has to leave.

| | Cloud AI | Edge AI |
|---|---|---|
| Where the model runs | Remote servers | On the device |
| Response time | Depends on network | Fast, no round trip |
| Internet needed | Yes | No, works offline |
| Privacy | Data is sent away | Data stays local |
| Running cost | Pay for servers per use | Hardware you already own |
| Model size and power | Large models, lots of compute | Small models, limited memory and battery |

### Why explore edge AI?

Edge AI is a good fit when a product needs instant responses, has weak or no internet, handles private data (cameras, health, voice), or would cost too much to run in the cloud for every request. Cloud AI is still the better choice for very large models and heavy training.

The catch is that edge devices are small, so a model that works fine in a notebook often has to be converted and shrunk before it fits. That is the skill this project practices.

### What I explored

- **TinyML:** running machine learning on small, low-power hardware.
- **On-device inference:** making predictions locally instead of calling a server.
- **Model optimization:** making a model smaller and faster (here: ONNX conversion and INT8 quantization) while checking what it costs in accuracy.

---

## Tools

- **PyTorch / torchvision:** MobileNetV2 with pretrained ImageNet weights (the baseline)
- **ONNX:** a common model format that can run outside PyTorch
- **ONNX Runtime:** the engine that runs `.onnx` models efficiently on CPU
- **Imagenette:** a 10-class subset of ImageNet, used to measure accuracy

---

## Workflow (what the notebook does)

1. **Load a pretrained model.** MobileNetV2 from torchvision, put in evaluation mode, and sanity-checked on a test image.
2. **Measure the baseline.** File size and parameter count, latency (10 warm-up runs, then the mean of 100 timed runs), and accuracy on 200 validation images (20 per class from 10 Imagenette classes, mapped to their ImageNet class IDs).
3. **Export to ONNX** (opset 18) and validate the file with `onnx.checker`.
4. **Run it in ONNX Runtime** (CPU) and confirm its outputs match PyTorch (`rtol=1e-3`, `atol=1e-4`).
5. **Re-measure the ONNX FP32 model** with the same size, speed, and accuracy tests.
6. **Quantize to INT8** with post-training static quantization: QDQ format, per-channel weights, calibrated on 50 real images from the training split (not the test images).
7. **Re-measure the INT8 model.**
8. **Simulate an edge device** by limiting ONNX Runtime to one CPU thread.
9. **Compare and conclude.**

---

## Model Performance Comparison

| Model | Device | Size | Latency | FPS | Accuracy |
|---|---|---|---|---|---|
| PyTorch FP32 | Laptop (CPU) | 14.3 MB | 72.3 ms | 13.8 | 65.0% (130/200) |
| ONNX FP32 | Laptop (CPU) | 14.21 MB | 15.3 ms | 65.3 | 65.0% (130/200) |
| ONNX INT8 | Laptop (CPU) | 4.02 MB | 7.9 ms | 125.9 | 65.0% (130/200) |
| ONNX INT8 | Simulated edge (1 CPU thread) | 4.02 MB | 16.0 ms | 62.5 | same file as above |


### What I gained

- **Smaller:** 14.3 MB to 4.02 MB (about 3.6x) from INT8 quantization. Parameter count stays about 3.5 M, because quantization makes each number smaller rather than removing any.
- **Faster:** converting to ONNX Runtime alone cut latency from 72.3 ms to 15.3 ms (about 4.7x), because ONNX Runtime optimizes the graph. INT8 roughly halved that again to 7.9 ms (about 9x faster than PyTorch overall).
- **Still fast when constrained:** with one CPU thread, the INT8 model runs at 62.5 FPS.

---

## How to run it

```bash
git clone https://github.com/<your-username>/edge-ai-onnx-mobilenetv2.git
cd edge-ai-onnx-mobilenetv2
pip install -r requirements.txt
jupyter notebook Edge_Deployment_ONNX.ipynb
```

- Add any photo named `test.jpeg` next to the notebook (used for the sanity check and speed tests).
- The notebook downloads Imagenette (about 100 MB) the first time it runs.
- Run the cells in order. The notebook creates `mobilenetv2.onnx` and `mobilenetv2_int8.onnx` in the same folder.

Note: the newer PyTorch ONNX exporter can save weights in a separate `.onnx.data` file. The size cells add both files together so the reported size is the real one.

---


