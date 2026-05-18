#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

template <typename T>
struct TrtDestroy {
  void operator()(T* ptr) const {
    delete ptr;
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroy<T>>;

void checkCuda(cudaError_t status, const std::string& action);
