#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_
#include <iostream>

// 检查 CUDA Error
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}
/**
 * @brief Get the Thread Id object
 *
 * @return int
 */
__device__ inline int getThreadId() {
  int block_id = blockIdx.z * (gridDim.x * gridDim.y) +
                 blockIdx.y * (gridDim.x) + blockIdx.x;
  int thread_id = threadIdx.z * (blockDim.x * blockDim.y) +
                  threadIdx.y * (blockDim.x) + threadIdx.x +
                  block_id * (blockDim.x * blockDim.y * blockDim.z);
  return thread_id;
}
#endif