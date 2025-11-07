#ifdef __HIP_PLATFORM_AMD__
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;
#else
#include <mma.h>
using namespace nvcuda;
#endif

#define COMPLEX 2
#define REAL 0
#define IMAG 1

using Tin = half;
using Tout = float;

using A_t = Tin[BATCH_SIZE][M_GLOBAL][K_GLOBAL][COMPLEX];
using B_t = Tin[BATCH_SIZE][N_GLOBAL][K_GLOBAL][COMPLEX];
using C_t = Tout[BATCH_SIZE][M_GLOBAL][N_GLOBAL][COMPLEX];

using A_eff_t = Tin[M_PER_BLOCK][K_PER_BUFFER * COMPLEX];
using B_eff_t = Tin[N_PER_BLOCK * 2][K_PER_BUFFER * COMPLEX];
using C_eff_t = Tout[M_GLOBAL][N_GLOBAL * COMPLEX];

extern "C" __global__ void wmma_complex_gemm_basic_interleaved(C_t C, const A_t A, const B_t B) {
  const unsigned blockN = blockIdx.x;
  const unsigned blockM = blockIdx.y;
  const unsigned batch = blockIdx.z;
  const unsigned warpN = threadIdx.y;
  const unsigned warpM = threadIdx.z;

  const unsigned tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  const unsigned block_size = blockDim.x * blockDim.y * blockDim.z;

  const unsigned block_m_start = blockM * M_PER_BLOCK;
  const unsigned block_n_start = blockN * N_PER_BLOCK;
  const unsigned warp_m_start = warpM * M_PER_WARP;
  const unsigned warp_n_start = warpN * N_PER_WARP;

  __shared__ Tin A_s[M_PER_BLOCK][K_PER_BUFFER][COMPLEX];
  __shared__ Tin B_s[N_PER_BLOCK][2][K_PER_BUFFER][COMPLEX];

  // for tensor core operations, we pretend we do a matmul with N_ = N * 2 and K_ = K * COMPLEX;
  const unsigned M_ = M_PER_WARP;
  const unsigned N_ = N_PER_WARP * 2;
  const unsigned K_ = K_PER_BUFFER * COMPLEX;

  const unsigned TILES_M = M_ / M_WMMA;
  const unsigned TILES_N = N_ / N_WMMA;

  wmma::fragment<wmma::accumulator, M_WMMA, N_WMMA, K_WMMA, Tout> fragC[TILES_M][TILES_N];
  for (unsigned m = 0; m < TILES_M; m++) {
    for (unsigned n = 0; n < TILES_N; n++) {
      wmma::fill_fragment(fragC[m][n], static_cast<Tout>(0));
    }
  }

  for (unsigned k_start = 0; k_start < K_GLOBAL; k_start += K_PER_BUFFER) {
    for (unsigned i = tid; i < M_PER_BLOCK * K_PER_BUFFER * COMPLEX; i += block_size) {
      const unsigned c = i % COMPLEX;
      const unsigned k = (i / COMPLEX) % K_PER_BUFFER;
      const unsigned m = (i / (COMPLEX * K_PER_BUFFER));

      A_s[m][k][c] = A[batch][block_m_start + m][k_start + k][c];
    }

    for (unsigned i = tid; i < N_PER_BLOCK * K_PER_BUFFER; i += block_size) {
      const unsigned k = i % K_PER_BUFFER;
      const unsigned n = i / K_PER_BUFFER;

      B_s[n][0][k][REAL] = B[batch][block_n_start + n][k_start + k][REAL];
      B_s[n][0][k][IMAG] = -B[batch][block_n_start + n][k_start + k][IMAG];

      B_s[n][1][k][REAL] = B[batch][block_n_start + n][k_start + k][IMAG];
      B_s[n][1][k][IMAG] = B[batch][block_n_start + n][k_start + k][REAL];
    }
    __syncthreads();

    const A_eff_t *A_ = reinterpret_cast<const A_eff_t *>(A_s);
    const B_eff_t *B_ = reinterpret_cast<const B_eff_t *>(B_s);

    for (unsigned m = 0; m < TILES_M; m++) {
      for (unsigned n = 0; n < TILES_N; n++) {
        for (unsigned k = 0; k < K_ / K_WMMA; k++) {
          wmma::fragment<rocwmma::matrix_a, M_WMMA, N_WMMA, K_WMMA, Tin, wmma::row_major> fragA;
          wmma::fragment<rocwmma::matrix_b, M_WMMA, N_WMMA, K_WMMA, Tin, wmma::col_major> fragB;
          wmma::load_matrix_sync(fragA, &(*A_)[warp_m_start + m * M_WMMA][k * K_WMMA], K_);
          wmma::load_matrix_sync(fragB, &(*B_)[(warp_n_start * 2) + n * N_WMMA][k * K_WMMA], K_);
          wmma::mma_sync(fragC[m][n], fragA, fragB, fragC[m][n]);
        }
      }
    }
    __syncthreads();
  }

  C_eff_t *C_ = reinterpret_cast<C_eff_t *>(&C[batch]);

  for (unsigned m = 0; m < TILES_M; m++) {
    for (unsigned n = 0; n < TILES_N; n++) {
      wmma::store_matrix_sync(
          &(*C_)[block_m_start + warp_m_start + m * M_WMMA][(block_n_start + warp_n_start) * COMPLEX + n * N_WMMA],
          fragC[m][n], N_GLOBAL * COMPLEX, wmma::mem_row_major);
    }
  }
}
