#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <cmath>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>
#include <ccglib/common/helper.h>

#include "kernel_transpose_complex.cu.o.h"

float runTranspose(cu::Stream &stream, cu::Function &kernel, cu::DeviceMemory &d_A, cu::DeviceMemory &d_B,
                   unsigned N, unsigned M) {
    dim3 block(32, 8);
    dim3 grid(ccglib::helper::ceildiv(M, 32), ccglib::helper::ceildiv(N, 32));

    std::vector<const void*> params = {d_A.parameter(), d_B.parameter(), &N, &M};

    cu::Event start, end;
    stream.record(start);
    stream.launchKernel(kernel, grid.x, grid.y, 1, block.x, block.y, 1, 0, params);
    stream.record(end);
    stream.synchronize();

    return end.elapsedTime(start);
}

int main() {
    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
    cu::Stream stream;

    constexpr unsigned N = 8192;
    constexpr unsigned M = 8192;

    const size_t bytes = N * M * sizeof(float2);

    cu::HostMemory h_A(bytes);
    cu::HostMemory h_B(bytes);
    cu::HostMemory h_B_ref(bytes);

    // Fill matrix with test pattern
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    auto rng = std::bind(dist, gen);

    float2* h_A_ptr = static_cast<float2*>(h_A);
    for (unsigned i = 0; i < N * M; i++) {
        h_A_ptr[i].x = rng();
        h_A_ptr[i].y = rng();
    }

    cu::DeviceMemory d_A(bytes);
    cu::DeviceMemory d_B(bytes);
    d_B.zero(bytes);

    stream.memcpyHtoDAsync(d_A, h_A, bytes);
    stream.synchronize();

    // Runtime compile kernel
    std::string include_path = nvrtc::findIncludePath();
    nvrtc::Program program(kernel_transpose_complex_source, "kernel_transpose_complex.cu");
    std::vector<std::string> options = {"-std=c++17", "-arch=" + device.getArch(), "-I" + include_path};
    program.compile(options);

    cu::Module module(static_cast<const void*>(program.getPTX().data()));
    cu::Function kernel(module, "transpose");

    float runtime_ms = runTranspose(stream, kernel, d_A, d_B, N, M);

    // Copy back
    stream.memcpyDtoHAsync(h_B, d_B, bytes);
    stream.synchronize();

    // Check correctness on host
    float max_err = 0.f;
    float2* h_B_ptr = static_cast<float2*>(h_B);
    for (unsigned i = 0; i < N; ++i)
        for (unsigned j = 0; j < M; ++j) {
            float2 a = h_A_ptr[i * M + j];
            float2 b = h_B_ptr[j * N + i];
            float err = std::max(std::abs(a.x - b.x), std::abs(a.y - b.y));
            if (err > max_err) max_err = err;
        }

    std::cout << "Transpose correctness max error: " << max_err << std::endl;
    std::cout << "Runtime: " << runtime_ms << " ms" << std::endl;

    double bandwidth = 2.0 * bytes * 1e-6 / runtime_ms; // read + write in MB/ms
    std::cout << "Effective memory bandwidth: " << bandwidth << " GB/s" << std::endl;

    return 0;
}
