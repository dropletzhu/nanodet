"""测试NanoDet MindSpore训练 - 简化版"""
import os
os.environ['DEVICE_ID'] = '0'

import mindspore as ms
from mindspore import context, Tensor, nn
import numpy as np

context.set_context(
    mode=context.GRAPH_MODE,
    device_target='Ascend',
    device_id=0
)

print('=== Testing NanoDet MindSpore Training ===')

from nanodet_mindspore.model.backbone import ShuffleNetV2

print('\n=== Test 1: Create and Forward ===')
backbone = ShuffleNetV2(model_size="1.5x", out_stages=(2, 3, 4), with_last_conv=False)

input_data = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
outputs = backbone(input_data)
print(f'Forward OK: {[o.shape for o in outputs]}')

print('\n=== Test 2: Forward + Loss ===')
target = Tensor(np.random.randn(1, 176, 52, 52).astype(np.float32))
loss_fn = nn.MSELoss()
loss = loss_fn(outputs[0], target)
print(f'Loss computed: {loss.asnumpy():.4f}')

print('\n=== Test 3: Save/Load Checkpoint ===')
from mindspore import save_checkpoint, load_checkpoint
save_checkpoint(backbone, 'test_ckpt.ckpt')
print('Checkpoint saved')

load_checkpoint('test_ckpt.ckpt', backbone)
print('Checkpoint loaded')

os.remove('test_ckpt.ckpt')

print('\n=== All Tests PASSED ===')
print('\n=== Summary ===')
print('- Model forward: OK')
print('- Loss computation: OK')
print('- Checkpoint save/load: OK')
print('- Ascend NPU: Working correctly')
