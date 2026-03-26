"""MindSpore模型测试脚本 - Ascend NPU"""
import os
os.environ['DEVICE_ID'] = '0'

import mindspore as ms
from mindspore import context, Tensor, nn
import numpy as np

# 设置Ascend NPU
context.set_context(
    mode=context.GRAPH_MODE,
    device_target='Ascend',
    device_id=0
)

print('=== Testing on Ascend NPU ===')
print(f'Device target: {ms.get_context("device_target")}')
print(f'Device id: {ms.get_context("device_id")}')


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


print('\n=== Testing Model Forward ===')
net = SimpleNet()
input_data = Tensor(np.random.randn(1, 3, 320, 320).astype(np.float32))
output = net(input_data)
print(f'Forward pass OK, output shape: {output.shape}')


print('\n=== Testing Loss Backward ===')

class LossNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.MSELoss()
        
    def construct(self, x, target):
        out = self.net(x)
        loss = self.loss_fn(out, target)
        return loss


target = Tensor(np.random.randn(1, 32, 320, 320).astype(np.float32))
loss_net = LossNet(net)
grad_fn = ms.value_and_grad(loss_net, None, net.trainable_params())
loss, grads = grad_fn(input_data, target)
print(f'Backward pass OK, loss: {loss.asnumpy():.4f}')


print('\n=== Testing Save/Load Checkpoint ===')
from mindspore import save_checkpoint, load_checkpoint

save_checkpoint(net, 'test_model.ckpt')
load_checkpoint('test_model.ckpt', net)
print('Save/Load checkpoint OK')
os.remove('test_model.ckpt')


print('\n=== All Tests PASSED ===')
