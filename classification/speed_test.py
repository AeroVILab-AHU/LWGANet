"""
Testing the speed of different models
"""
import datetime
import os
import torch
import time
import utils
from models.LWGANet import LWGANet_L0_1242_e32_k11_GELU, LWGANet_L1_1242_e64_k11_GELU, LWGANet_L2_1442_e96_k11_ReLU
from models.MobileNetV2 import mobilenet_v2_10, mobilenet_v2_20, mobilenet_v2_25
from models.efficientformerv2 import efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2
from models.fasternet import fasternet_t0, fasternet_t1, fasternet_t2
from models.edgevit import edgevit_xxs, edgevit_xs, edgevit_s
from models.edgenext.edgenext import edgenext_xx_small, edgenext_x_small, edgenext_small
from models.ghostnetv2 import ghostnetv2_06, ghostnetv2_10, ghostnetv2_20
from models.mobileViT import mobilevit_xxs, mobilevit_xs, mobilevit_s
from models.PVT_V2 import pvt_v2_b0, pvt_v2_b1
from models.starnet import starnet_s1, starnet_s3, starnet_s4


torch.autograd.set_grad_enabled(False)


T0 = 10
T1 = 60


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

    throughput_cpu_results = "throughput_cpu_results.txt"
    throughput_cpu_results = os.path.join('./outputs', throughput_cpu_results)
    with open(throughput_cpu_results, "a") as f:
        info = f"[backbone_name: {name}]  " \
               f"device: {device}  " \
               f"batch_size: {batch_size}  " \
               f"images/s: {(batch_size / timing.mean().item()):.4f}  "
        f.write(info + "\n")

def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

    throughput_gpu_results = "throughput_gpu_results.txt"
    throughput_gpu_results = os.path.join('./outputs', throughput_gpu_results)
    with open(throughput_gpu_results, "a") as f:
        info = f"[backbone_name: {name}]  " \
               f"device: {device}  " \
               f"batch_size: {batch_size}  " \
               f"images/s: {(batch_size / timing.mean().item()):.4f}  "
        f.write(info + "\n")

# for device in ['cuda:0', 'cpu']:
for device in ['cpu', 'cuda:0']:
    if 'cuda' in device and not torch.cuda.is_available():
        print("no cuda")
        continue

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    else:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    for n, batch_size0, resolution in [
        ('starnet_s1', 256, 224),
        ('starnet_s3', 256, 224),
        ('starnet_s4', 256, 224),
        ('mobilenet_v2_10', 256, 224),
        ('mobilenet_v2_20', 256, 224),
        ('mobilenet_v2_25', 256, 224),
        ('efficientformerv2_s0', 256, 224),
        ('efficientformerv2_s1', 256, 224),
        ('efficientformerv2_s2', 256, 224),
        ('fasternet_t0', 256, 224),
        ('fasternet_t1', 256, 224),
        ('fasternet_t2', 256, 224),
        ('edgevit_xxs', 256, 224),
        ('edgevit_xs', 256, 224),
        ('edgevit_s', 256, 224),
        ('edgenext_xx_small', 256, 224),
        ('edgenext_x_small', 256, 224),
        ('edgenext_small', 256, 224),
        ('ghostnetv2_06', 256, 224),
        ('ghostnetv2_10', 256, 224),
        ('ghostnetv2_20', 256, 224),
        ('mobilevit_xxs', 256, 256),
        ('mobilevit_xs', 256, 256),
        ('mobilevit_s', 256, 256),
        ('pvt_v2_b0', 256, 224),
        ('pvt_v2_b1', 256, 224),
        ('LWGANet_L0_1242_e32_k11_GELU', 256, 224),
        ('LWGANet_L1_1242_e64_k11_GELU', 256, 224),
        ('LWGANet_L2_1442_e96_k11_ReLU', 256, 224),
    ]:

        if device == 'cpu':
            batch_size = 16
        else:
            batch_size = batch_size0
            torch.cuda.empty_cache()
        inputs = torch.randn(batch_size, 3, resolution,
                             resolution, device=device)
        model = eval(n)(num_classes=1000)
        utils.replace_batchnorm(model)
        model.to(device)
        model.eval()
        model = torch.jit.trace(model, inputs)
        compute_throughput(n, model, device,
                           batch_size, resolution=resolution)
