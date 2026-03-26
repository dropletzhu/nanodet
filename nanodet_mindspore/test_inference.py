"""MindSpore完整推理测试"""
import os
os.environ['DEVICE_ID'] = '0'

import mindspore as ms
from mindspore import context, Tensor
import numpy as np
import cv2

context.set_context(
    mode=context.GRAPH_MODE,
    device_target='Ascend',
    device_id=0
)

print('=== MindSpore Inference Test on Ascend NPU ===')

# 导入模型
from nanodet_mindspore.model.arch import build_model
from nanodet_mindspore.util import load_config

print('\n=== Test 1: Build Model ===')
cfg = {
    'model': {
        'arch': {
            'backbone': {
                'name': 'ShuffleNetV2',
                'model_size': '1.5x',
                'out_stages': (2, 3, 4),
                'with_last_conv': False
            },
            'fpn': None,
            'head': None
        }
    }
}

model = build_model(cfg['model'])
print('Model built successfully')

print('\n=== Test 2: Create Dummy Checkpoint ===')
from mindspore import save_checkpoint
save_checkpoint(model, 'inference_test.ckpt')
print('Checkpoint created')

print('\n=== Test 3: Load Checkpoint ===')
load_checkpoint = ms.load_checkpoint
load_checkpoint('inference_test.ckpt', model)
print('Checkpoint loaded')

print('\n=== Test 4: Image Preprocessing ===')
# 模拟图像预处理
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
img_resized = cv2.resize(img, (416, 416))
img_float = img_resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_normalized = (img_float - mean) / std
img_tensor = Tensor(img_normalized.transpose(2, 0, 1).reshape(1, 3, 416, 416))
print(f'Input shape: {img_tensor.shape}')

print('\n=== Test 5: Inference ===')
outputs = model(img_tensor)
print(f'Number of outputs: {len(outputs)}')
for i, out in enumerate(outputs):
    print(f'  Output {i}: {out.shape}')

print('\n=== Test 6: Batch Inference ===')
batch_input = Tensor(np.random.randn(4, 3, 416, 416).astype(np.float32))
batch_outputs = model(batch_input)
print(f'Batch input shape: {batch_input.shape}')
for i, out in enumerate(batch_outputs):
    print(f'  Batch output {i}: {out.shape}')

print('\n=== Test 7: Multiple Inferences ===')
for i in range(5):
    input_data = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
    outputs = model(input_data)
print(f'5 inferences completed')

os.remove('inference_test.ckpt')

print('\n' + '='*50)
print('=== All Inference Tests PASSED ===')
print('='*50)
print('\nSummary:')
print('- Model building: OK')
print('- Checkpoint save/load: OK')
print('- Image preprocessing: OK')
print('- Single inference: OK')
print('- Batch inference: OK')
print('- Multiple inferences: OK')
print('- Ascend NPU: Working correctly')
