#include <complex>
#include <functional>
#include <iostream>
#include <random>

#include <ccglib/ccglib.hpp>
#include <ccglib/common/helper.h>
#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#include "kernel_complex_interleaved.cu.o.h"

template<typename T>
inline T align(T a, T b) {
  return ccglib::helper::ceildiv(a, b) * b;
}

float run(cu::Stream &stream, ccglib::pipeline::Pipeline &pipeline, cu::DeviceMemory &d_a, cu::DeviceMemory &d_b, cu::DeviceMemory &d_c) {
  // run the kernels
  cu::Event start, end;
  stream.record(start);
  pipeline.Run(d_a, d_b, d_c);
  stream.record(end);
  stream.synchronize();

  return end.elapsedTime(start);
}

int main(int argc, char *argv[]) {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  // data size and type
//   const unsigned global_m = 8192;
//   const unsigned global_n = 8192;
//   const unsigned global_k = 8192;
  const unsigned global_m = 16;
  const unsigned global_n = 16;
  const unsigned global_k = 16;
  const unsigned batch_size = 1;
  const unsigned COMPLEX = 2;
  const unsigned REAL = 0;
  const unsigned IMAG = 1;

  const double ops = 8ULL * 1e-9 * global_m * global_n * global_k;

  using Tin = half;
  using Tout = float;

  const size_t bytes_a =
      sizeof(Tin) * batch_size * COMPLEX * global_m * global_k;
  const size_t bytes_b =
      sizeof(Tin) * batch_size * COMPLEX * global_n * global_k;
  const size_t bytes_c =
      sizeof(Tout) * batch_size * COMPLEX * global_m * global_n;

  // Build the experimental kernel
  const std::string kernel_name = "wmma_complex_gemm_basic_interleaved";

  const std::string include_path = nvrtc::findIncludePath();

  const std::string arch = device.getArch();
  const unsigned warp_size = device.getAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);

  // tuning parameters
  const unsigned BLOCK_TILE_SIZE_M = 16;
  const unsigned BLOCK_TILE_SIZE_N = 16;
  const unsigned WARP_TILE_SIZE_M = 16;
  const unsigned WARP_TILE_SIZE_N = 16;
  const unsigned TILE_SIZE_K = 16;

  // fixed parameters
  const unsigned M_WMMA = 16;
  const unsigned N_WMMA = 16;
  const unsigned K_WMMA = 16;

  dim3 grid{ccglib::helper::ceildiv(global_n, BLOCK_TILE_SIZE_N), ccglib::helper::ceildiv(global_m, BLOCK_TILE_SIZE_M), 1};
  dim3 threads{warp_size, ccglib::helper::ceildiv(WARP_TILE_SIZE_N, N_WMMA), ccglib::helper::ceildiv(WARP_TILE_SIZE_M, M_WMMA)};

  std::cout << "block size: " << threads.x << " " << threads.y << " " << threads.z << std::endl;
  std::cout << "grid size: " << grid.x << " " << grid.y << " " << grid.z << std::endl;

  std::vector<std::string> options = {
    "-std=c++17",
#if defined(__HIP__)
    "--offload-arch=" + arch,
#else
    "-arch=" + arch,
#endif
    "-I" + include_path,
    "-DCOMPLEX=" + std::to_string(COMPLEX),
    "-DWARP_SIZE=" + std::to_string(warp_size),
    "-DM_GLOBAL=" + std::to_string(global_m) + "UL",
    "-DN_GLOBAL=" + std::to_string(global_n) + "UL",
    "-DK_GLOBAL=" + std::to_string(global_k) + "UL",
    "-DBLOCK_TILE_SIZE_M=" + std::to_string(BLOCK_TILE_SIZE_M),
    "-DBLOCK_TILE_SIZE_N=" + std::to_string(BLOCK_TILE_SIZE_N),
    "-DWARP_TILE_SIZE_M=" + std::to_string(WARP_TILE_SIZE_M),
    "-DWARP_TILE_SIZE_N=" + std::to_string(WARP_TILE_SIZE_N),
    "-DTILE_SIZE_K=" + std::to_string(TILE_SIZE_K),
    "-DM_WMMA=" + std::to_string(M_WMMA),
    "-DN_WMMA=" + std::to_string(N_WMMA),
    "-DK_WMMA=" + std::to_string(K_WMMA)
  };

  nvrtc::Program program(kernel_complex_interleaved_source, "kernel_complex_interleaved.cu");
  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }
  cu::Module module(static_cast<const void *>(program.getPTX().data()));
  cu::Function function(module, kernel_name.c_str());

  // initalize host memory
  cu::HostMemory h_a(bytes_a);
  cu::HostMemory h_b(bytes_b);
  cu::HostMemory h_c(bytes_c);
  cu::HostMemory h_c_ref(bytes_c);

  // Allocate device memory for input data
  cu::DeviceMemory d_a(bytes_a);
  cu::DeviceMemory d_b(bytes_b);

  // Create input data
  auto generator = std::bind(std::uniform_real_distribution<float>(-10, 10),
                             std::default_random_engine());
  for (unsigned i = 0; i < global_m * global_k * COMPLEX; i ++) {
    static_cast<Tin *>(h_a)[i] = static_cast<Tin>(generator());
  }
  for (unsigned i = 0; i < global_n * global_k * COMPLEX; i ++) {
    static_cast<Tin *>(h_b)[i] = static_cast<Tin>(generator());
  }

  // Transfer the input data
  stream.memcpyHtoDAsync(d_a, h_a, bytes_a);
  stream.memcpyHtoDAsync(d_b, h_b, bytes_b);

  // allocate device memory for output data and initialize to zero
  cu::DeviceMemory d_c(bytes_c);
  d_c.zero(bytes_c);
  float runtime;
  double tflops;

  // run the experimental kernel
  std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                          d_b.parameter()};

  cu::Event start, end;
  stream.record(start);
  stream.launchKernel(function, grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, 0, parameters);
  stream.record(end);
  stream.synchronize();

  runtime = end.elapsedTime(start);

  stream.memcpyDtoHAsync(h_c, d_c, bytes_c);
  stream.synchronize();

  tflops = ops / runtime;
  std::cout << std::endl;
  std::cout << "Custom complex interleaved" << std::endl;
  std::cout << "runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;

  // compare to existing ccglib pipelines
  ccglib::pipeline::Pipeline pipeline_opt_planar(batch_size, global_m, global_n, global_k, device, stream,
                                                 // input/output complex axis location
                                                 ccglib::complex_planar, ccglib::complex_planar,
                                                 // a, b, c mem order
                                                 ccglib::mma::row_major, ccglib::mma::col_major, ccglib::mma::row_major,
                                                 // input/output precision
                                                 ccglib::float16, ccglib::float32,
                                                 // kernel variant
                                                 ccglib::mma::opt
                                                 );


  d_c.zero(bytes_c);
  runtime = run(stream, pipeline_opt_planar, d_a, d_b, d_c);

  tflops = ops / runtime;
  std::cout << std::endl;
  std::cout << "Pipeline opt, complex planar" << std::endl;
  std::cout << "runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;

  ccglib::pipeline::Pipeline pipeline_opt_interleaved(batch_size, global_m, global_n, global_k, device, stream,
                                                      // input/output complex axis location
                                                      ccglib::complex_interleaved, ccglib::complex_interleaved,
                                                      // a, b, c mem order
                                                      ccglib::mma::row_major, ccglib::mma::col_major, ccglib::mma::row_major,
                                                      // input/output precision
                                                      ccglib::float16, ccglib::float32,
                                                      // kernel variant
                                                      ccglib::mma::opt
                                                      );

  d_c.zero(bytes_c);
  runtime = run(stream, pipeline_opt_interleaved, d_a, d_b, d_c);
  // copy C to host
  stream.memcpyDtoHAsync(h_c_ref, d_c, bytes_c);
  stream.synchronize();

  tflops = ops / runtime;
  std::cout << std::endl;
  std::cout << "Pipeline opt, complex interleaved" << std::endl;
  std::cout << "runtime: " << runtime << " ms" << std::endl;
  std::cout << "TFLOPS: " << tflops << std::endl;

  // Compare pipeline interleaved to new kernel
  unsigned errs = 0;
  for (unsigned m = 0; m < global_m; m ++ ) {
    for (unsigned n = 0; n < global_n; n++) {
      if (errs > 10) {
          break;
      }
      const unsigned idx = n * COMPLEX + m * global_n * COMPLEX;
      const std::complex<Tout> value{static_cast<Tout *>(h_c)[idx + REAL], static_cast<Tout *>(h_c)[idx + IMAG]};
      const std::complex<Tout> value_ref{static_cast<Tout *>(h_c_ref)[idx + REAL], static_cast<Tout *>(h_c_ref)[idx + IMAG]};
      const Tout diff = std::max(std::abs(value.real() - value_ref.real()), std::abs(value.imag() - value_ref.imag()));
      if (diff > .1) {
        errs++;
      }
    }
  }

  std::cout << std::endl;
  if (errs > 0) {
    std::cout << "Result not ok, errs: " << errs << std::endl;
  } else {
    std::cout << "Result ok" << std::endl;
  }

  if (true) {
    std::cout << std::endl;
    for (unsigned m = 0; m < global_m; m ++ ) {
      for (unsigned n = 0; n < global_n; n++) {
          const unsigned idx = n * COMPLEX + m * global_n * COMPLEX;
          std::cout << std::setw(12) << static_cast<Tout *>(h_c)[idx + REAL] << " + " << static_cast<Tout *>(h_c)[idx + IMAG] << "i";
          std::cout << "   ";
          std::cout << std::setw(12) << static_cast<Tout *>(h_c_ref)[idx + REAL] << " + " << static_cast<Tout *>(h_c_ref)[idx + IMAG] << "i";
          std::cout << std::endl;
      }
    break;
    }
  }
}
