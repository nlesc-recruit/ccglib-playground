#pragma once

extern "C" __global__ void transpose(const float2* __restrict__ A, float2* __restrict__ B, unsigned N, unsigned M) {
  constexpr int TILE_DIM = 32;
  constexpr int BLOCK_ROWS = 8;

  __shared__ float2 tile[TILE_DIM][TILE_DIM + 1];

  unsigned x = blockIdx.x * TILE_DIM + threadIdx.x;
  unsigned y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    if (x < M && (y + i) < N) tile[threadIdx.y + i][threadIdx.x] = A[(y + i) * M + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // swap blockIdx
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    if (x < N && (y + i) < M) {
      B[(y + i) * N + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}
