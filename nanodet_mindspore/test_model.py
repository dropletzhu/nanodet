"""测试NanoDet MindSpore模型"""
import os
os.environ['DEVICE_ID'] = '0'

import mindspore as ms
from mindspore import context, Tensor
import numpy as np

context.set_context(
    mode=context.GRAPH_MODE,
    device_target='Ascend',
    device_id=0
)

print('=== Testing NanoDet MindSpore Model ===')

# 导入迁移的模块
from nanodet_mindspore.model.backbone import ShuffleNetV2


print('\n=== Test 1: ShuffleNetV2 Backbone ===')
backbone = ShuffleNetV2(
    model_size="1.5x",
    out_stages=(2, 3, 4),
    with_last_conv=False
)

input_data = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
outputs = backbone(input_data)
print(f'Input: {input_data.shape}')
for i, out in enumerate(outputs):
    print(f'Output {i}: {out.shape}')


print('\n=== Test 2: Full Model Forward ===')
from nanodet_mindspore.model.arch import OneStageDetector

model = OneStageDetector(
    backbone_cfg=dict(
        name="ShuffleNetV2",
        model_size="1.5x",
        out_stages=(2, 3, 4),
        with_last_conv=False
    ),
    fpn_cfg=None,
    head_cfg=None
)

x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
output = model(x)
print(f'Model output: {type(output)}')
if isinstance(output, tuple):
    for i, o in enumerate(output):
        print(f'  Output {i}: {o.shape}')


print('\n=== Test 3: Checkpoint Save/Load ===')
from mindspore import save_checkpoint, load_checkpoint

save_checkpoint(model, 'nanodet_test.ckpt')
print('Checkpoint saved')

load_checkpoint('nanodet_test.ckpt', model)
print('Checkpoint loaded')
os.remove('nanodet_test.ckpt')


print('\n=== All Tests PASSED ===')
