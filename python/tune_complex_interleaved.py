#!/usr/bin/env python3
import warnings

import argparse
import kernel_tuner as kt
import numpy as np
import os
import re

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Device name, used in output filename")
    parser.add_argument("-m", type=int, required=True, help="Size of M axis")
    parser.add_argument("-n", type=int, required=True, help="Size of N axis")
    parser.add_argument("-k", type=int, required=True, help="Size of K axis")
    parser.add_argument("-b", type=int, default=1, help="Size of Batch axis (default: %(default)s)")
    parser.add_argument("--backend", required=True, choices=["cupy", "hip"], help="Kernel Tuner backend")
    parser.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Overwrite any existing .json files",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    name = args.name
    m_global = args.m
    n_global = args.n
    k_global = args.k
    batch_size = args.b
    backend = args.backend

    # on AMD GPUs, the warp size can be 32 or 64 and the shared memory size is different from nvidia
    if backend == "hip":
        from pyhip import hip
        device_properties = hip.hipGetDeviceProperties(0)
        warp_size = device_properties.warpSize
        smem_size = device_properties.sharedMemPerBlock
    else:
        # assume nvidia defaults
        warp_size = 32
        smem_size = 49152

    # block size x is always warp_size, so the other block sizes can be at
    # most 1024 / warp_size
    tune_params = {
        "block_size_x": [warp_size],  # must be warp size
        "block_size_y": [1],
        "block_size_z": [1],
        # "block_size_y": [2**i for i in range(0, 6) if warp_size * 2**i <= 1024],
        # "block_size_z": [2**i for i in range(0, 6) if warp_size * 2**i <= 1024],
        "M_PER_BLOCK": [2**i for i in range(4, 9)],  # minimum m_per_wmma
        "N_PER_BLOCK": [2**i for i in range(4, 9)],  # minimum n_per_wmma
        "K_PER_BUFFER": [2**i for i in range(4, 9)], # minimum k_per_wmma
    }

    defines = {key: key for key in tune_params}

    defines["kernel_tuner"] = 1
    defines["BATCH_SIZE"] = batch_size
    defines["M_GLOBAL"] = m_global
    defines["N_GLOBAL"] = n_global
    defines["K_GLOBAL"] = k_global
    #defines["N_PER_WARP"] = lambda p: int(p["N_PER_BLOCK"] // p["block_size_y"])
    #defines["M_PER_WARP"] = lambda p: int(p["M_PER_BLOCK"] // p["block_size_z"])
    defines["WARP_SIZE"] = warp_size
    defines["M_WMMA"] = 16
    defines["N_WMMA"] = 16
    defines["K_WMMA"] = 16

    kernel_file = "kernel_complex_interleaved.cu"
    kernel_name = "wmma_complex_gemm_basic_interleaved"
    dtype_ab = np.float16
    dtype_c = np.float32

    A = np.zeros((batch_size, m_global, k_global, 2), dtype=dtype_ab)
    B = np.zeros((batch_size, n_global, k_global, 2), dtype=dtype_ab)
    C = np.zeros((batch_size, n_global, m_global, 2), dtype=dtype_c)

    problem_size = (n_global, m_global, batch_size)
    arguments = (C, A, B)

    grid_div = {
        "grid_div_x": lambda p: p["N_PER_BLOCK"],
        "grid_div_y": lambda p: p["M_PER_BLOCK"],
        "grid_div_z": lambda p: 1
    }

    metrics = {
        "TFLOPS": lambda p: 8e-9 * m_global * n_global * k_global * batch_size / p["time"],
        # "N_PER_WARP": lambda p: p["N_PER_BLOCK"] // p["block_size_y"],
        # "M_PER_WARP": lambda p: p["M_PER_BLOCK"] // p["block_size_z"]
    }

    with open(f"../kernels/{kernel_file}", "r") as fp:
        kernel_source = fp.read()

    compiler_options = ["-std=c++17"]

    def restrict(*args):
        param_names = list(tune_params.keys())
        assert len(args) == len(param_names)
        p = {}
        for i in range(len(param_names)):
            p[param_names[i]] = args[i]

        # __shared__ Tin A_s[M_PER_BLOCK][K_PER_BUFFER][COMPLEX];
        # __shared__ Tin B_s[N_PER_BLOCK][2][K_PER_BUFFER][COMPLEX];
        a_size = p["M_PER_BLOCK"] * p["K_PER_BUFFER"] * 2 * np.dtype(dtype_ab).itemsize
        b_size = p["N_PER_BLOCK"] * 2 * p["K_PER_BUFFER"] * 2 * np.dtype(dtype_ab).itemsize
        valid = (
            a_size + b_size <= smem_size
            and p["block_size_x"] * p["block_size_y"] * p["block_size_z"] <= 1024
        )
        return valid

    filename_cache = (
        f"tuning/{name}_{kernel_name}_{batch_size}x{m_global}x{n_global}x{k_global}.json"
    )
    if args.overwrite and os.path.exists(filename_cache):
            os.remove(filename_cache)

    kt.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params,
                   restrictions=restrict,
                   compiler_options=compiler_options,
                   # strategy="dual_annealing", strategy_options=dict(max_fevals=200),
                   # strategy="random_sample",
                   cache=filename_cache,
                   metrics=metrics,
                   defines=defines, lang=backend, verbose=True,
                   **grid_div)
