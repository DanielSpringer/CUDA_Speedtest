#%%
import numpy as np
import torch
# import cupy as cp
import time

# %%


def cpu1(N):
    M = torch.rand(N,N)
    M2 = torch.rand(N,N)
    M_c = M#.cuda()
    M2_c = M2#.cuda()
    Z = torch.einsum('ik,kj->ij', M_c, M2_c)

def cpu2(N):
    M = torch.rand(N,N)
    M2 = torch.rand(N,N)
    M_c = M#.cuda()
    M2_c = M2#.cuda()
    Z = torch.matmul(M_c, M2_c)

def cuda1(N):
    M = torch.rand(N,N)
    M2 = torch.rand(N,N)
    M_c = M.cuda()
    M2_c = M2.cuda()
    Z = torch.einsum('ik,kj->ij', M_c, M2_c)

def cuda2(N):
    M = torch.rand(N,N)
    M2 = torch.rand(N,N)
    M_c = M.cuda()
    M2_c = M2.cuda()
    Z = torch.matmul(M_c, M2_c)



N = 1000

start_cpu = time.time()
cpu1(N)
end_cpu = time.time()
print("CPU einsum:", end_cpu-start_cpu)

start_cpu = time.time()
cpu2(N)
end_cpu = time.time()
print("CPU matmul:", end_cpu-start_cpu)

start_gpu = time.time()
cuda1(N)
end_gpu = time.time()
print("GPU einsum:", end_gpu-start_gpu)

start_gpu = time.time()
cuda2(N)
end_gpu = time.time()
print("GPU matmul:", end_gpu-start_gpu)

print("Done")
