# -*- coding: utf-8 -*-
# 修复尝试 #2: KL -> minmax 校准 + 开启 bias_correct
# 其余与 calibration_head_fp32.py 相同(检测头 /model.23/ 保持 FP32)
import onnx  # NOQA: 必须先导入, 否则 ppq 内部 import onnx DLL 崩溃

from pathlib import Path
from typing import Iterable

import torch

from ppq import TargetPlatform, graphwise_error_analyse
from ppq.api.interface import export_ppq_graph, load_onnx_graph, quantize_native_model
from ppq.api.setting import QuantizationSettingFactory

BATCHSIZE = 1
INPUT_SHAPE = [BATCHSIZE, 3, 640, 640]
DEVICE = "cpu"
PLATFORM = TargetPlatform.TRT_INT8

ONNX_PATH = r"E:\ppq_test\yolo26n_fused.onnx"
OUTPUT_ONNX_PATH = r"E:\ppq_test\yolo26n_int8_headfp32_minmax.onnx"
CALIBRATION_DIR = Path(r"E:\ppq_test\calibration")
FP32_HEAD_PREFIX = "/model.23/"


def load_calibration_dataset() -> Iterable:
    calibration = []
    for path in sorted(CALIBRATION_DIR.glob("*.pt")):
        x = torch.load(path, map_location="cpu").float()
        assert list(x.shape) == [1, 3, 640, 640], f"{path.name} shape错误: {x.shape}"
        calibration.append(x)
    print("Calibration samples:", len(calibration))
    return calibration


CALIBRATION = load_calibration_dataset()


def collate_fn(batch):
    if isinstance(batch, torch.Tensor):
        return batch.to(DEVICE)
    if batch and all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in batch):
        return torch.cat(batch, dim=0).to(DEVICE)
    return torch.stack(batch).to(DEVICE)


graph = load_onnx_graph(onnx_import_file=ONNX_PATH)

QSetting = QuantizationSettingFactory.default_setting()

# 修复 1: activation 校准 KL -> minmax(避免长尾分布被截断)
QSetting.quantize_activation_setting.calib_algorithm = "minmax"
QSetting.quantize_parameter_setting.calib_algorithm = "minmax"

# 修复 2: 开启 bias correction(校正量化后卷积偏置, 治 logit 负偏)
QSetting.bias_correct = True

# 检测头整段保持 FP32
fp32_count = 0
for op_name in graph.operations:
    if op_name.startswith(FP32_HEAD_PREFIX):
        QSetting.dispatching_table.append(op_name, TargetPlatform.FP32)
        fp32_count += 1
print(f"检测头 {fp32_count} 个算子保持 FP32; 校准: minmax; bias_correct: {QSetting.bias_correct}")

quantized = quantize_native_model(
    model=graph,
    calib_dataloader=CALIBRATION,
    calib_steps=len(CALIBRATION),
    input_shape=INPUT_SHAPE,
    setting=QSetting,
    collate_fn=collate_fn,
    platform=PLATFORM,
    device=DEVICE,
    verbose=1,
)

reports = graphwise_error_analyse(
    graph=quantized,
    running_device=DEVICE,
    collate_fn=collate_fn,
    dataloader=CALIBRATION,
)

export_ppq_graph(
    graph=quantized,
    platform=TargetPlatform.ONNXRUNTIME,
    graph_save_to=OUTPUT_ONNX_PATH,
)

# 补回元数据(类别名等)
try:
    src_meta = onnx.load(ONNX_PATH)
    dst_meta = onnx.load(OUTPUT_ONNX_PATH)
    del dst_meta.metadata_props[:]
    for kv in src_meta.metadata_props:
        e = dst_meta.metadata_props.add()
        e.key, e.value = kv.key, kv.value
    onnx.save(dst_meta, OUTPUT_ONNX_PATH)
    print("已补回元数据(类别名等)")
except Exception as e:
    print(f"警告: 补回元数据失败({e})")

print("INT8 QDQ ONNX 已保存:", OUTPUT_ONNX_PATH)
