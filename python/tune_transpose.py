#!/usr/bin/env python3
import warnings

import argparse
import kernel_tuner as kt
import numpy as np
import os
import re

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def align(a, b):
    return int(np.ceil(a / b) * b)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Device name, used in output filename")
    parser.add_argument("-m", type=int, required=True, help="Size of M axis")
    parser.add_argument("-n", type=int, required=True, help="Size of N axis")
    parser.add_argument("-b", type=int, default=1, help="Size of Batch axis (default: %(default)s)")
    parser.add_argument("--m_chunk", type=int, required=True, help="Chunk size in M")
    parser.add_argument("--n_chunk", type=int, required=True, help="Chunk size in N")
    parser.add_argument("--complex_axis_location", choices=["planar", "interleaved"], default="planar", help="Complex axis location (default: %(default)s)")
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
    batch_size = args.b
    m_chunk = args.m_chunk
    n_chunk = args.n_chunk
    backend = args.backend

    dtype = np.float16

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
        "block_size_x": [2**i for i in range(10)],
        "block_size_y": [2**i for i in range(10)],
    }

    defines = {key: key for key in tune_params}

    defines["kernel_tuner"] = 1
    defines["BATCH_SIZE"] = batch_size
    defines["M_GLOBAL"] = m_global
    defines["N_GLOBAL"] = n_global
    defines["M_CHUNK"] = m_chunk
    defines["N_CHUNK"] = n_chunk
    if args.complex_axis_location == "planar":
        defines["INPUT_COMPLEX_PLANAR"] = "1"
    else:
        defines["INPUT_COMPLEX_INTERLEAVED"] = "1"

    kernel_file = "kernel_transpose.cu"
    kernel_name = "transpose_original"

    m_padded = align(m_global, m_chunk)
    n_padded = align(n_global, n_chunk)

    A = np.zeros((batch_size, m_global, n_global, 2), dtype=dtype)
    A_trans = np.zeros((batch_size, m_padded // m_chunk, n_padded // n_chunk, 2, m_chunk, n_chunk), dtype=dtype)

    problem_size = (n_global, m_global, batch_size)
    arguments = (A_trans, A)

    metrics = {
        "GB/s": lambda p: (A.nbytes + A_trans.nbytes) / p["time"] / 1e6
    }

    compiler_options = ["-std=c++17"]

    filename_cache = (
        f"tuning/{name}_{kernel_name}_{batch_size}_{m_global}_{n_global}_{m_chunk}_{n_chunk}.json"
    )
    if args.overwrite and os.path.exists(filename_cache):
            os.remove(filename_cache)

    kt.tune_kernel(kernel_name, f"../kernels/{kernel_file}", problem_size, arguments, tune_params,
                   compiler_options=compiler_options,
                   cache=filename_cache,
                   metrics=metrics,
                   defines=defines, lang=backend)
