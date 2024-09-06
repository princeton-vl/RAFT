#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W
#define CHANNEL_STRIDE 32


__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void corr_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr,
    int r)
{
  const int b = blockIdx.x;
  const int h0 = blockIdx.y * blockDim.x;
  const int w0 = blockIdx.z * blockDim.y;
  const int tid = threadIdx.x * blockDim.y + threadIdx.y;

  const int H1 = fmap1.size(1);
  const int W1 = fmap1.size(2);
  const int H2 = fmap2.size(1);
  const int W2 = fmap2.size(2);
  const int N = coords.size(1);
  const int C = fmap1.size(3);

  __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW+1];
  __shared__ scalar_t f2[CHANNEL_STRIDE][BLOCK_HW+1];
  __shared__ scalar_t x2s[BLOCK_HW];
  __shared__ scalar_t y2s[BLOCK_HW];

  for (int c=0; c<C; c+=CHANNEL_STRIDE) {
    for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
      int k1 = k + tid / CHANNEL_STRIDE;
      int h1 = h0 + k1 / BLOCK_W;
      int w1 = w0 + k1 % BLOCK_W;
      int c1 = tid % CHANNEL_STRIDE;

      auto fptr = fmap1[b][h1][w1];
      if (within_bounds(h1, w1, H1, W1))
        f1[c1][k1] = fptr[c+c1];
      else
        f1[c1][k1] = 0.0;
    }

    __syncthreads();

    for (int n=0; n<N; n++) {
      int h1 = h0 + threadIdx.x;
      int w1 = w0 + threadIdx.y;
      if (within_bounds(h1, w1, H1, W1)) {
        x2s[tid] = coords[b][n][h1][w1][0];
        y2s[tid] = coords[b][n][h1][w1][1];
      }

      scalar_t dx = x2s[tid] - floor(x2s[tid]);
      scalar_t dy = y2s[tid] - floor(y2s[tid]);

      int rd = 2*r + 1;
      for (int iy=0; iy<rd+1; iy++) {
        for (int ix=0; ix<rd+1; ix++) {
          for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
            int k1 = k + tid / CHANNEL_STRIDE;
            int h2 = static_cast<int>(floor(y2s[k1]))-r+iy;
            int w2 = static_cast<int>(floor(x2s[k1]))-r+ix;
            int c2 = tid % CHANNEL_STRIDE;

            auto fptr = fmap2[b][h2][w2];
            if (within_bounds(h2, w2, H2, W2))
              f2[c2][k1] = fptr[c+c2];
            else
              f2[c2][k1] = 0.0;
          }

          __syncthreads();
      
          scalar_t s = 0.0;
          for (int k=0; k<CHANNEL_STRIDE; k++)
            s += f1[k][tid] * f2[k][tid];

          int ix_nw = H1*W1*((iy-1) + rd*(ix-1));
          int ix_ne = H1*W1*((iy-1) + rd*ix);
          int ix_sw = H1*W1*(iy + rd*(ix-1));
          int ix_se = H1*W1*(iy + rd*ix);

          scalar_t nw = s * (dy) * (dx);
          scalar_t ne = s * (dy) * (1-dx);
          scalar_t sw = s * (1-dy) * (dx);
          scalar_t se = s * (1-dy) * (1-dx);

          scalar_t* corr_ptr = &corr[b][n][0][h1][w1];

          if (iy > 0 && ix > 0 && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_nw) += nw;

          if (iy > 0 && ix < rd && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_ne) += ne;

          if (iy < rd && ix > 0 && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_sw) += sw;

          if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_se) += se;
        }
      } 
    }
  }
}


// template <typename scalar_t>
// __global__ void corr_backward_kernel(
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
//     const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords,
//     const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1_grad,
//     torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2_grad,
//     torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords_grad,
//     int r)
// {

//   const int b = blockIdx.x;
//   const int h0 = blockIdx.y * blockDim.x;
//   const int w0 = blockIdx.z * blockDim.y;
//   const int tid = threadIdx.x * blockDim.y + threadIdx.y;

//   const int H1 = fmap1.size(1);
//   const int W1 = fmap1.size(2);
//   const int H2 = fmap2.size(1);
//   const int W2 = fmap2.size(2);
//   const int N = coords.size(1);
//   const int C = fmap1.size(3);

//   __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW+1];
//   __shared__ scalar_t f2[CHANNEL_STRIDE][BLOCK_HW+1];

//   __shared__ scalar_t f1_grad[CHANNEL_STRIDE][BLOCK_HW+1];
//   __shared__ scalar_t f2_grad[CHANNEL_STRIDE][BLOCK_HW+1];

//   __shared__ scalar_t x2s[BLOCK_HW];
//   __shared__ scalar_t y2s[BLOCK_HW];

//   for (int c=0; c<C; c+=CHANNEL_STRIDE) {

//     for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
//       int k1 = k + tid / CHANNEL_STRIDE;
//       int h1 = h0 + k1 / BLOCK_W;
//       int w1 = w0 + k1 % BLOCK_W;
//       int c1 = tid % CHANNEL_STRIDE;

//       auto fptr = fmap1[b][h1][w1];
//       if (within_bounds(h1, w1, H1, W1))
//         f1[c1][k1] = fptr[c+c1];
//       else
//         f1[c1][k1] = 0.0;

//       f1_grad[c1][k1] = 0.0;
//     }

//     __syncthreads();

//     int h1 = h0 + threadIdx.x;
//     int w1 = w0 + threadIdx.y;

//     for (int n=0; n<N; n++) {  
//       x2s[tid] = coords[b][n][h1][w1][0];
//       y2s[tid] = coords[b][n][h1][w1][1];

//       scalar_t dx = x2s[tid] - floor(x2s[tid]);
//       scalar_t dy = y2s[tid] - floor(y2s[tid]);

//       int rd = 2*r + 1;
//       for (int iy=0; iy<rd+1; iy++) {
//         for (int ix=0; ix<rd+1; ix++) {
//           for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
//             int k1 = k + tid / CHANNEL_STRIDE;
//             int h2 = static_cast<int>(floor(y2s[k1]))-r+iy;
//             int w2 = static_cast<int>(floor(x2s[k1]))-r+ix;
//             int c2 = tid % CHANNEL_STRIDE;

//             auto fptr = fmap2[b][h2][w2];
//             if (within_bounds(h2, w2, H2, W2))
//               f2[c2][k1] = fptr[c+c2];
//             else
//               f2[c2][k1] = 0.0;

//             f2_grad[c2][k1] = 0.0;
//           }

//           __syncthreads();
      
//           const scalar_t* grad_ptr = &corr_grad[b][n][0][h1][w1];
//           scalar_t g = 0.0;

//           int ix_nw = H1*W1*((iy-1) + rd*(ix-1));
//           int ix_ne = H1*W1*((iy-1) + rd*ix);
//           int ix_sw = H1*W1*(iy + rd*(ix-1));
//           int ix_se = H1*W1*(iy + rd*ix);

//           if (iy > 0 && ix > 0 && within_bounds(h1, w1, H1, W1))
//             g +=  *(grad_ptr + ix_nw) * dy * dx;

//           if (iy > 0 && ix < rd && within_bounds(h1, w1, H1, W1))
//             g += *(grad_ptr + ix_ne) * dy * (1-dx);

//           if (iy < rd && ix > 0 && within_bounds(h1, w1, H1, W1))
//             g += *(grad_ptr + ix_sw) * (1-dy) * dx;

//           if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1))
//             g += *(grad_ptr + ix_se) * (1-dy) * (1-dx);
            
//           for (int k=0; k<CHANNEL_STRIDE; k++) {
//             f1_grad[k][tid] += g * f2[k][tid];
//             f2_grad[k][tid] += g * f1[k][tid];
//           }

//           for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
//             int k1 = k + tid / CHANNEL_STRIDE;
//             int h2 = static_cast<int>(floor(y2s[k1]))-r+iy;
//             int w2 = static_cast<int>(floor(x2s[k1]))-r+ix;
//             int c2 = tid % CHANNEL_STRIDE;

//             scalar_t* fptr = &fmap2_grad[b][h2][w2][0];
//             if (within_bounds(h2, w2, H2, W2))
//               atomicAdd(fptr+c+c2, f2_grad[c2][k1]);
//           }
//         }
//       } 
//     }
//     __syncthreads();


//     for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
//       int k1 = k + tid / CHANNEL_STRIDE;
//       int h1 = h0 + k1 / BLOCK_W;
//       int w1 = w0 + k1 % BLOCK_W;
//       int c1 = tid % CHANNEL_STRIDE;

//       scalar_t* fptr = &fmap1_grad[b][h1][w1][0];
//       if (within_bounds(h1, w1, H1, W1))
//         fptr[c+c1] += f1_grad[c1][k1];
//     }
//   }
// }


template <typename scalar_t>
__global__ void corr_backward_kernel(
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
  const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
  const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords,
  const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1_grad,
  torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2_grad,
  torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords_grad,
  int r
) {
  const int B = fmap1.size(0);
  const int H1 = fmap1.size(1);
  const int W1 = fmap1.size(2);
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < B*H1*W1) {
    const int H2 = fmap2.size(1);
    const int W2 = fmap2.size(2);
    const int N = coords.size(1);
    const int C = fmap1.size(3);
    const int rd = int(sqrt(corr_grad.size(2)));
    const int b = idx / (H1 * W1);
    const int h = (idx % (H1*W1)) / W1;
    const int w = (idx % (H1*W1)) % W1;
    const int r = (rd - 1) / 2;
    for (int n = 0; n < N; n++) {
      scalar_t coords_x = coords[b][n][h][w][0], coords_y = coords[b][n][h][w][1];
      for (int iy = -r; iy <= r; iy++) {
        for (int ix = -r; ix <= r; ix++) {
          scalar_t x = coords_x + ix, y = coords_y + iy;
          int x_floor = static_cast<int>(floor(x)), y_floor = static_cast<int>(floor(y));
					scalar_t dx = x - x_floor, dy = y - y_floor;
					scalar_t weights_ii[2] = {1 - dx, dx}, weights_jj[2] = {1 - dy, dy};
          scalar_t g = corr_grad[b][n][(ix + r) * rd + iy + r][h][w];
          for (int c = 0; c < C; c++) {
            scalar_t f = fmap1[b][h][w][c];
            scalar_t gf = 0;
            for (int ii = 0; ii < 2; ii++) {
              for (int jj = 0; jj < 2; jj++) {
                int x0 = x_floor + ii, y0 = y_floor + jj;
                if (within_bounds(y0, x0, H2, W2)) {
                  gf += g * fmap2[b][y0][x0][c] * weights_ii[ii] * weights_jj[jj];
                  atomicAdd(&fmap2_grad[b][y0][x0][c], g * f * weights_ii[ii] * weights_jj[jj]);
                }
              }
            }
            fmap1_grad[b][h][w][c] += gf;
          }
        }
      }
    }
  }
}

std::vector<torch::Tensor> corr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius)
{
  const auto B = coords.size(0);
  const auto N = coords.size(1);
  const auto H = coords.size(2);
  const auto W = coords.size(3);

  const auto rd = 2 * radius + 1;
  auto opts = fmap1.options();
  auto corr = torch::zeros({B, N, rd*rd, H, W}, opts);
  
  const dim3 blocks(B, (H+BLOCK_H-1)/BLOCK_H, (W+BLOCK_W-1)/BLOCK_W);
  const dim3 threads(BLOCK_H, BLOCK_W);

  corr_forward_kernel<float><<<blocks, threads>>>(
    fmap1.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    fmap2.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    corr.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    radius);

  return {corr};
}

std::vector<torch::Tensor> corr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius)
{
  const auto B = coords.size(0);
  const auto N = coords.size(1);

  const auto H1 = fmap1.size(1);
  const auto W1 = fmap1.size(2);
  const auto H2 = fmap2.size(1);
  const auto W2 = fmap2.size(2);
  const auto C = fmap1.size(3);

  auto opts = fmap1.options();
  auto fmap1_grad = torch::zeros({B, H1, W1, C}, opts);
  auto fmap2_grad = torch::zeros({B, H2, W2, C}, opts);
  auto coords_grad = torch::zeros({B, N, H1, W1, 2}, opts);

  int threadsPerBlock = 256;
  int blocksPerGrid = (B*H1*W1 + threadsPerBlock - 1) / threadsPerBlock;

  corr_backward_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(
    fmap1.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    fmap2.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    corr_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    fmap1_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    fmap2_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    coords_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    radius);

  return {fmap1_grad, fmap2_grad, coords_grad};
}