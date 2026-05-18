#include "trt_utils.hpp"

#include <stdexcept>

void checkCuda(cudaError_t status, const std::string& action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(action + ": " + cudaGetErrorString(status));
  }
}
