import torch
import time


def bench(device="cpu"):
    print(f"running on {device}:")
    a = torch.randn(size=(2000, 20000)).to(device=device)
    b = torch.randn_like(a).to(device=device)

    start = time.time()
    c = a + b
    end = time.time()

    c.to(device="cpu")
    # print(f'available devices: {torch.dml.device_count()}')
    # print(f'current device: { torch.dml.current_device()}')
    print(f"--took {end - start:.2f} seconds")


bench("cpu")
bench("cuda:0")
bench("cuda:1")
