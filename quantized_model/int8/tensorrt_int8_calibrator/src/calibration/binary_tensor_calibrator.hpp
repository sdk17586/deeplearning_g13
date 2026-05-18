#pragma once

#include <NvInfer.h>

#include <cstddef>
#include <string>
#include <vector>

class BinaryTensorCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  BinaryTensorCalibrator(std::vector<std::string> tensorFiles, int batchSize,
                         std::size_t inputBytes, std::string cachePath);
  ~BinaryTensorCalibrator() noexcept override;

  int32_t getBatchSize() const noexcept override;
  bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override;
  void const* readCalibrationCache(std::size_t& length) noexcept override;
  void writeCalibrationCache(void const* cache, std::size_t length) noexcept override;

 private:
  std::vector<std::string> tensorFiles_;
  int batchSize_ = 1;
  std::size_t inputBytes_ = 0;
  std::string cachePath_;
  std::size_t index_ = 0;
  void* deviceInput_ = nullptr;
  std::vector<char> cache_;
};
