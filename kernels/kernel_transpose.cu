#define COMPLEX 2
#define REAL 0
#define IMAG 1

using T = half;

#define M_IS_PADDED ((M_GLOBAL % M_CHUNK) != 0)
#define N_IS_PADDED ((N_GLOBAL % N_CHUNK) != 0)

#define M_GLOBAL_PADDED ((M_GLOBAL / M_CHUNK + M_IS_PADDED) * M_CHUNK)
#define N_GLOBAL_PADDED ((N_GLOBAL / N_CHUNK + N_IS_PADDED) * N_CHUNK)

#if defined(INPUT_COMPLEX_PLANAR)
using Input = T[BATCH_SIZE][COMPLEX][M_GLOBAL][N_GLOBAL];
#elif defined(INPUT_COMPLEX_INTERLEAVED)
using Input = T[BATCH_SIZE][M_GLOBAL][N_GLOBAL][COMPLEX];
#endif
using Output =
    T[BATCH_SIZE][M_GLOBAL_PADDED / M_CHUNK][N_GLOBAL_PADDED / N_CHUNK][COMPLEX]
     [M_CHUNK][N_CHUNK];

extern "C" {
__global__ void transpose_original(Output out, const Input in) {
  const size_t idx_B = blockIdx.z;
  const size_t idx_N =
      threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  const size_t idx_M =
      threadIdx.y + blockIdx.y * static_cast<size_t>(blockDim.y);

  static_assert(M_GLOBAL_PADDED % M_CHUNK == 0);
  static_assert(N_GLOBAL_PADDED % N_CHUNK == 0);

  if (idx_B < BATCH_SIZE && idx_M < M_GLOBAL_PADDED &&
      idx_N < N_GLOBAL_PADDED) {
    size_t b = idx_B;
    size_t m = idx_M / M_CHUNK;
    size_t m_c = idx_M % M_CHUNK;
    size_t n = idx_N / (N_CHUNK);
    size_t n_c = idx_N % (N_CHUNK);

    if (idx_M < M_GLOBAL && idx_N < N_GLOBAL) {
#if defined(INPUT_COMPLEX_PLANAR)
      out[b][m][n][REAL][m_c][n_c] = in[b][REAL][idx_M][idx_N];
      out[b][m][n][IMAG][m_c][n_c] = in[b][IMAG][idx_M][idx_N];
#elif defined(INPUT_COMPLEX_INTERLEAVED)
      out[b][m][n][REAL][m_c][n_c] = in[b][idx_M][idx_N][REAL];
      out[b][m][n][IMAG][m_c][n_c] = in[b][idx_M][idx_N][IMAG];
#endif
    } else {
      out[b][m][n][REAL][m_c][n_c] = static_cast<T>(0.0f);
      out[b][m][n][IMAG][m_c][n_c] = static_cast<T>(0.0f);
    }
  }
}
}
