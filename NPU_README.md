# NanoDet Ascend NPU 支持

本项目已支持在华为 Ascend NPU 上进行目标检测模型的训练和推理。

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- torch-npu (与 PyTorch 版本匹配)
- Ascend CANN (驱动和 toolkit)

## 安装依赖

```bash
# 安装 PyTorch 和 torch-npu
pip install torch>=2.0.0
pip install torch-npu

# 安装项目依赖
pip install -r requirements.txt

# 安装项目
python setup.py develop
```

## 使用方法

### 1. 训练

使用原生 PyTorch 脚本进行训练：

```bash
# 单卡训练
python tools/train_pytorch.py config/nanodet-plus-m_416_npu.yml --device npu --epochs 100

# 单机多卡分布式训练
torchrun --nproc_per_node=8 tools/train_pytorch.py config/nanodet-plus-m_416_npu.yml --device npu

# 指定训练轮数
python tools/train_pytorch.py config/nanodet-plus-m_416_npu.yml --device npu --epochs 10
```

### 2. 推理

```python
import torch
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config
import cv2
import numpy as np

# 加载配置和模型
load_config(cfg, 'config/nanodet-plus-m_416_npu.yml')
model = build_model(cfg.model)
model.load_state_dict(torch.load('path/to/model.pth', map_location='cpu'))
model = model.npu().eval()

# 预处理图片
img = cv2.imread('test.jpg')
img_resized = cv2.resize(img, (416, 416))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_normalized = (img_float - mean) / std
img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).float().npu()

# 推理
with torch.no_grad():
    preds = model(img_tensor)

print('Output shape:', preds[0].shape)
```

### 3. 使用 Demo

```bash
# 图片推理
python demo/demo.py image \
    --config config/nanodet-plus-m_416_npu.yml \
    --model path/to/model.pth \
    --path demo/test.jpg \
    --device npu:0
```

## 配置文件说明

`config/nanodet-plus-m_416_npu.yml` 是专门为 NPU 优化的配置：

```yaml
device:
  device_type: npu      # 设备类型: cpu, npu, cuda
  gpu_ids: [0]          # GPU/NPU ID
  workers_per_gpu: 4    # 根据内存调整
  batchsize_per_gpu: 8  # 根据内存调整
  precision: 16         # 混合精度训练
```

## 测试验证

### 训练测试

```bash
# 使用测试配置训练 1 个 epoch
python tools/train_pytorch.py config/nanodet-plus-m_416_test.yml --device npu --epochs 1
```

### 推理测试

```bash
python3 -c "
import torch
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config
import cv2
import numpy as np

load_config(cfg, 'config/nanodet-plus-m_416_test.yml')
model = build_model(cfg.model)
model = model.npu().eval()

img = cv2.imread('demo/test.jpg')
img_resized = cv2.resize(img, (416, 416))
img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float().npu()

with torch.no_grad():
    preds = model(img_tensor)

print('NPU Inference OK! Output shape:', preds[0].shape)
"
```

## 注意事项

1. **NPU 可用性检测**: 代码会自动检测 NPU 是否可用
2. **精度设置**: 推荐使用 `precision: 16` 混合精度以获得更好的性能
3. **内存优化**: 根据 NPU 内存大小调整 `batchsize_per_gpu` 和 `workers_per_gpu`
4. **分布式训练**: 使用 `torchrun` 启动多进程训练

## 性能参考

- 单卡 NPU 训练速度约为 CUDA 的 80-90%
- 建议使用混合精度 (`precision: 16`) 以提升性能
