# ppq_yolo26n_int8
基于 PPQ 量化后的 YOLO26n ONNX(包含处理源代码）
项目的onnx是yolo26n官方预训练权重。可直接使用，压缩包中的图片为量化效果
使用混合量化，检测头保持fp32（不要修改，否则完全无法检出物体）

# Quantized YOLO Object Detection Model (PPQ / PPL INT8)

This repository contains the quantized **YOLO** object detection model processed using the **PPL Quantization Tool (PPQ)**. The model is optimized with Quantize-Dequantize (QDQ) nodes for efficient INT8 inference on TensorRT / PPL CUDA runtimes.

---

##  torchvision / Model Overview

- **Base Architecture:** YOLO (Custom / YOLOv8 / YOLOv10 backbone with Attention modules)
- **Quantization Framework:** [PPL Quantization Tool (PPQ)](https://github.com/openppl-ai/ppq)
- **Quantization Precision:** INT8 (Symmetric / Per-Channel Axis Quantization)
- **Format:** ONNX with `QuantizeLinear` and `DequantizeLinear` (QDQ) node pairs

---

## 🛠️ Quantization Details

The model utilizes **Per-Channel (Axis) Quantization** for Conv / MatMul weights and **Per-Tensor Quantization** for activation layers, ensuring minimal accuracy loss while significantly speeding up inference.

Key quantized operators in the computation graph include:
- `Conv` / `ConvTranspose` layers (Per-axis quantization on weights)
- `MatMul` & `Softmax` (Attention mechanism layers)
- Element-wise operations (`Mul`, `Add`, `Concat`)

### Network Sub-blocks Quantized:
- **Backbone & Neck:** `model.0` ~ `model.9` (Conv, C2f/C3, SPPF)
- **Attention Modules:** `model.10.m.0.attn` (QKV projection, PE, MatMul, Proj)
- **FFN Modules:** `model.10.m.0.ffn` & `model.22.m.0.1.ffn`
- **Detection / Multi-scale Feature Fusion:** `model.13` ~ `model.22`

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following packages installed:

```bash
pip install torch onnx onnxruntime ppq












根据你提供的图片及 PPL/PPQ 量化模型的节点拓扑数据，这是一个**基于 PPQ (PPL Quantization Tool) 量化后的 YOLOv8 / YOLOv10 (带 Attention 模块) ONNX 模型架构**。

为你整理好了一份结构清晰、专业度高的 GitHub `README.md` 模板。你可以根据具体项目情况稍作补充和修改。

---

```markdown
# Quantized YOLO Object Detection Model (PPQ / PPL INT8)

This repository contains the quantized **YOLO** object detection model processed using the **PPL Quantization Tool (PPQ)**. The model is optimized with Quantize-Dequantize (QDQ) nodes for efficient INT8 inference on TensorRT / PPL CUDA runtimes.

---

##  torchvision / Model Overview

- **Base Architecture:** YOLO (Custom / YOLOv8 / YOLOv10 backbone with Attention modules)
- **Quantization Framework:** [PPL Quantization Tool (PPQ)](https://github.com/openppl-ai/ppq)
- **Quantization Precision:** INT8 (Symmetric / Per-Channel Axis Quantization)
- **Format:** ONNX with `QuantizeLinear` and `DequantizeLinear` (QDQ) node pairs

---

## 🛠️ Quantization Details

The model utilizes **Per-Channel (Axis) Quantization** for Conv / MatMul weights and **Per-Tensor Quantization** for activation layers, ensuring minimal accuracy loss while significantly speeding up inference.

Key quantized operators in the computation graph include:
- `Conv` / `ConvTranspose` layers (Per-axis quantization on weights)
- `MatMul` & `Softmax` (Attention mechanism layers)
- Element-wise operations (`Mul`, `Add`, `Concat`)

### Network Sub-blocks Quantized:
- **Backbone & Neck:** `model.0` ~ `model.9` (Conv, C2f/C3, SPPF)
- **Attention Modules:** `model.10.m.0.attn` (QKV projection, PE, MatMul, Proj)
- **FFN Modules:** `model.10.m.0.ffn` & `model.22.m.0.1.ffn`
- **Detection / Multi-scale Feature Fusion:** `model.13` ~ `model.22`

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following packages installed:

```bash
pip install torch onnx onnxruntime ppq

```

### 1. Model Inspection (Netron)

You can visualize the quantized ONNX model structure using [Netron](https://netron.app/). The graph explicitly shows `QuantizeLinear` and `DequantizeLinear` ops inserted around standard `Conv` and `MatMul` nodes.

### 2. Run Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load quantized ONNX model
model_path = "path/to/quantized_yolo_model.onnx"
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Prepare dummy input (adjust batch size and channels accordingly)
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: dummy_input})
print("Inference successful. Output shape:", outputs[0].shape)

```

### 3. Deploying with TensorRT / PPL CUDA

To deploy this QDQ model on TensorRT:

```bash
trtexec --onnx=quantized_yolo_model.onnx --saveEngine=yolo_int8.engine --int8

```

---

## 📊 Performance & Precision Benchmark

> *Feel free to update the table below with your actual benchmark data.*

| Precision | FP32 | INT8 (PPQ) | Speedup / Improvement |
| --- | --- | --- | --- |
| **mAP@0.5** | -- | -- | - |
| **mAP@0.5:0.95** | -- | -- | - |
| **Latency (ms)** | -- | -- | **X.X x** |
| **Model Size** | ~XX MB | ~XX MB | **~75% reduction** |

---

## 📄 License

Distributed under the [MIT License](https://www.google.com/search?q=LICENSE).

```

---

### 说明与建议：
1. **模型类型调整**：从节点名称 `model.10.m.0.attn.qkv.conv.weight` 可以看到模型里包含了 Transformer/Attention 结构（如 YOLOv10 或结合了 Self-Attention/C2f-Attention 的变体）。你可以根据具体模型的版本（如 YOLOv8s / YOLOv10m 等）修改 README 中的说明。
2. **测试数据补充**：建议在表格部分（`Performance & Precision Benchmark`）填入你实际测得的精度（mAP）和耗时对比，这样 GitHub 仓库会更有说服力。

```
