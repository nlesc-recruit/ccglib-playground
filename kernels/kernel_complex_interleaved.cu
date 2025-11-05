#include <rocwmma/rocwmma.hpp>

#define REAL 0
#define IMAG 1

using Tin = half;
using Tout = float;

using A_t = Tin[M_GLOBAL][K_GLOBAL][COMPLEX];
using B_t = Tin[N_GLOBAL][K_GLOBAL][COMPLEX];
using C_t = Tout[M_GLOBAL][N_GLOBAL][COMPLEX];

using A_eff_t = Tin[M_GLOBAL][K_GLOBAL * COMPLEX];
using B_eff_t = Tin[N_GLOBAL * 2][K_GLOBAL * COMPLEX];
using C_eff_t = Tout[M_GLOBAL][N_GLOBAL * COMPLEX];

extern "C" __global__ void wmma_complex_gemm_basic_interleaved(C_t C, const A_t A, const B_t B) {
    __shared__ Tin B_s[N_GLOBAL][2][K_GLOBAL][COMPLEX];
    for (unsigned k = 0; k < K_GLOBAL; k++) {
        for (unsigned n = 0; n < N_GLOBAL; n++) {
            B_s[n][0][k][REAL] = B[n][k][REAL];
            B_s[n][0][k][IMAG] = -B[n][k][IMAG];

            B_s[n][1][k][REAL] = B[n][k][IMAG];
            B_s[n][1][k][IMAG] = B[n][k][REAL];
        }
    }
    __syncthreads();
    // pretend we do a matmul with N_ = N * 2 and K_ = K * COMPLEX;
    const unsigned M_ = M_GLOBAL;  // unchanged
    const unsigned N_ = N_GLOBAL * 2;
    const unsigned K_ = K_GLOBAL * COMPLEX;

    const A_eff_t *A_ = reinterpret_cast<const A_eff_t *>(A);
    const B_eff_t *B_ = reinterpret_cast<const B_eff_t *>(B_s);
    const C_eff_t *C_ = reinterpret_cast<const C_eff_t *>(C);

    for (unsigned n = 0; n < N_; n += N_WMMA) {
        for (unsigned m = 0; m < M_; m += M_WMMA) {
            rocwmma::fragment<rocwmma::accumulator, M_WMMA, N_WMMA, K_WMMA, Tout> fragC;
            rocwmma::fill_fragment(fragC, static_cast<Tout>(0));
            for (unsigned k = 0; k < K_; k += K_WMMA) {
                rocwmma::fragment<rocwmma::matrix_a, M_WMMA, N_WMMA, K_WMMA, Tin, rocwmma::row_major> fragA;
                rocwmma::fragment<rocwmma::matrix_b, M_WMMA, N_WMMA, K_WMMA, Tin, rocwmma::col_major> fragB;
                rocwmma::load_matrix_sync(fragA,  &(*A_)[m][k], K_);
                rocwmma::load_matrix_sync(fragB,  &(*B_)[n][k], K_);
                rocwmma::mma_sync(fragC, fragA, fragB, fragC);
            }
            rocwmma::store_matrix_sync(&(*C)[m][n], fragC, N_, rocwmma::mem_row_major);
        }
    }
    __syncthreads();


    // rocwmma::load_matrix_sync(fragA, &A[0][0][0], K_GLOBAL * COMPLEX);
    // rocwmma::load_matrix_sync(fragB, &B[0][0][0], K_GLOBAL * COMPLEX);

    /*
    const size_t blockN = blockIdx.x;
    const size_t blockM = blockIdx.y;
    const size_t warpN = threadIdx.y;
    const size_t warpM = threadIdx.z;

    const size_t block_size = blockDim.x * blockDim.y * blockDim.z;
    const size_t tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    __shared__ Tin A_s[BLOCK_TILE_SIZE_M][TILE_SIZE_K][COMPLEX];
    __shared__ Tin B_s[BLOCK_TILE_SIZE_N][TILE_SIZE_K][COMPLEX];

    const unsigned block_start_m = blockM * BLOCK_TILE_SIZE_M;
    const unsigned block_start_n = blockN * BLOCK_TILE_SIZE_N;

    const unsigned warp_tiles_m = WARP_TILE_SIZE_M / M_WMMA;
    const unsigned warp_tiles_n = WARP_TILE_SIZE_N / N_WMMA;

    rocwmma::fragment<rocwmma::accumulator, M_WMMA, N_WMMA, K_WMMA, Tout> fragC[warp_tiles_m][warp_tiles_n];
    for (unsigned n = 0; n < warp_tiles_n; n++) {
        for (unsigned m = 0; m < warp_tiles_m; m++) {
            rocwmma::fill_fragment(fragC[m][n], static_cast<Tin>(0.0f));
        }
    }

    // todo: padding in K
    for (unsigned k_tile = 0; k_tile < K_GLOBAL / TILE_SIZE_K; k_tile++) {
        const unsigned k_start = k_tile * TILE_SIZE_K;

        // load A and B into shared memory
        for (unsigned i = 0; i < BLOCK_TILE_SIZE_M * TILE_SIZE_K * COMPLEX; i += block_size) {
            const unsigned c = i % COMPLEX;
            const unsigned k = (i / c) % TILE_SIZE_K;
            const unsigned m = (i / (k + c));
            A_s[m][k][c] = A[block_start_m + m][k_start + k][c];
        }

        for (unsigned i = 0; i < BLOCK_TILE_SIZE_N * TILE_SIZE_K * COMPLEX; i += block_size) {
            const unsigned c = i % COMPLEX;
            const unsigned k = (i / c) % TILE_SIZE_K;
            const unsigned n = (i / (k + c));
            B_s[n][k][c] = A[block_start_n + n][k_start + k][c];
        }

        __syncthreads();

        // run mma per K tile, note that we load 2 values per K due to complex axis
        for (unsigned k = 0; k < TILE_SIZE_K; k+= K_WMMA / COMPLEX) {
            // load MMA fragments
            rocwmma::fragment<rocwmma::matrix_a, M_WMMA, N_WMMA, K_WMMA, Tin, rocwmma::row_major> fragA[warp_tiles_m];
            rocwmma::fragment<rocwmma::matrix_b, M_WMMA, N_WMMA, K_WMMA, Tin, rocwmma::col_major> fragB[warp_tiles_n];

            for (unsigned warp_tile_m = 0; warp_tile_m < warp_tiles_m; warp_tile_m ++) {
                rocwmma::load_matrix_sync(fragA[warp_tile_m], &A_s[warp_tile_m * WARP_TILE_SIZE_M][k][0], COMPLEX * TILE_SIZE_K);
            }

            for (unsigned warp_tile_n = 0; warp_tile_n < warp_tiles_n; warp_tile_n ++) {
                rocwmma::load_matrix_sync(fragB[warp_tile_n], &A_s[warp_tile_n * WARP_TILE_SIZE_N][k][0], COMPLEX * TILE_SIZE_K);
            }

            // run MMA
            for (unsigned n = 0; n < warp_tiles_n; n++) {
                for (unsigned m = 0; m < warp_tiles_m; m++) {
                    rocwmma::mma_sync(fragC[m][n], fragA[m], fragB[n], fragC[m][n]);
                }
            }
        }

        // K loop done, store result
        for (unsigned n = 0; n < warp_tiles_n; n++) {
                for (unsigned m = 0; m < warp_tiles_m; m++) {
                    rocwmma::store_matrix_sync(&C[block_start_m + m * WARP_TILE_SIZE_M][block_start_n + n * WARP_TILE_SIZE_N][0], fragC[m][n], N_GLOBAL * COMPLEX, rocwmma::mem_row_major);
            }
        }

        // ensure all threads are done before new data is written to shared memory in next loop iteration
        __syncthreads();
    }

    */
}
