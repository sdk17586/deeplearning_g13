#include "binary_tensor_calibrator.hpp"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "file_utils.hpp"
#include "trt_utils.hpp"

BinaryTensorCalibrator::BinaryTensorCalibrator(std::vector<std::string> tensorFiles,
                                               int batchSize, std::size_t inputBytes,
                                               std::string cachePath)
    : tensorFiles_(std::move(tensorFiles)),
      batchSize_(batchSize),
      inputBytes_(inputBytes),
      cachePath_(std::move(cachePath)) {
  checkCuda(cudaMalloc(&deviceInput_, inputBytes_), "cudaMalloc calibration input");
}

BinaryTensorCalibrator::~BinaryTensorCalibrator() noexcept {
  if (deviceInput_) {
    cudaFree(deviceInput_);
  }
}

int32_t BinaryTensorCalibrator::getBatchSize() const noexcept {
  return batchSize_;
}

bool BinaryTensorCalibrator::getBatch(void* bindings[], char const* names[],
                                      int32_t nbBindings) noexcept {
  (void)names;
  (void)nbBindings;
  if (index_ >= tensorFiles_.size()) {
    return false;
  }

  try {
    const std::string& path = tensorFiles_[index_++];
    std::vector<char> host = readBinary(path);
    if (host.size() != inputBytes_) {
      std::cerr << "[ERROR] Unexpected tensor byte size: " << path
                << " got=" << host.size() << " expected=" << inputBytes_ << '\n';
      return false;
    }
    checkCuda(cudaMemcpy(deviceInput_, host.data(), inputBytes_, cudaMemcpyHostToDevice),
              "cudaMemcpy calibration input");
    bindings[0] = deviceInput_;
    std::cout << "[CALIB] " << index_ << "/" << tensorFiles_.size() << " " << path << '\n';
    return true;
  } catch (const std::exception& exc) {
    std::cerr << "[ERROR] " << exc.what() << '\n';
    return false;
  }
}

void const* BinaryTensorCalibrator::readCalibrationCache(std::size_t& length) noexcept {
  cache_.clear();
  std::ifstream in(cachePath_, std::ios::binary);
  if (!in) {
    length = 0;
    return nullptr;
  }
  cache_ = std::vector<char>(std::istreambuf_iterator<char>(in), {});
  length = cache_.size();
  std::cout << "[INFO] Using existing cache: " << cachePath_ << '\n';
  return cache_.data();
}

void BinaryTensorCalibrator::writeCalibrationCache(void const* cache,
                                                  std::size_t length) noexcept {
  try {
    writeBinary(cachePath_, cache, length);
    std::cout << "[OK] Wrote calibration cache: " << cachePath_ << '\n';
  } catch (const std::exception& exc) {
    std::cerr << "[ERROR] " << exc.what() << '\n';
  }
}
