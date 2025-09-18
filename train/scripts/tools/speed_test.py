import sys
import time
import torch
import os.path as op
from thop import profile

sys.path.append(op.abspath('../..'))
from scripts.model import ModelInterface

# Define a namespace for args
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
args = Namespace(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    batch_size=4
)

# Load model
print('Loading model...')
checkpoint_path = op.join(r'/tsukimi/v2ce-project/logs/2023_09_05_23_00_09_ablation2-no-match/checkpoints/best-epoch=91-val_BinaryMatchF1_sum_c=0.5372-val_BinaryMatch_raw=0.9705.ckpt')
model = ModelInterface.load_from_checkpoint(checkpoint_path, **vars(args), strict=False)
model = model.cuda().eval()

# Print model size
print('Model size: {:.2f} MB'.format(sum(p.numel() for p in model.parameters()) / 1e6))

# Print model flops
input = {'image_units': torch.randn(1, 16, 2, 512, 512).float().cuda()} 
with torch.no_grad():
    macs, params = profile(model, inputs=(input, ))
print('Model flops: {:.2f} GFLOPS'.format(macs / 1e9))

# Model average inference time
n = 100
t = 0
with torch.no_grad():
    for i in range(n):
        start = time.time()
        res = model(input)
        t += time.time() - start
print('Model average inference time: {:.2f} ms'.format(t / n * 1000))