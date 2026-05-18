#include "int8_engine_builder.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "binary_tensor_calibrator.hpp"
#include "file_utils.hpp"
#include "trt_logger.hpp"
#include "trt_utils.hpp"

Int8EngineBuilder::Int8EngineBuilder(CalibratorOptions options)
    : options_(std::move(options)) {}

void Int8EngineBuilder::build() {
  TrtLogger logger;

  TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) throw std::runtime_error("Failed to create TensorRT builder");

  TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  if (!network) throw std::runtime_error("Failed to create TensorRT network");

  TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  if (!parser) throw std::runtime_error("Failed to create ONNX parser");

  std::vector<char> onnxData = readBinary(options_.onnx);
  if (!parser->parse(onnxData.data(), onnxData.size())) {
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << parser->getError(i)->desc() << '\n';
    }
    throw std::runtime_error("Failed to parse ONNX: " + options_.onnx);
  }

  TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) throw std::runtime_error("Failed to create TensorRT builder config");

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             static_cast<std::size_t>(options_.workspaceMiB) << 20);
  config->setFlag(nvinfer1::BuilderFlag::kINT8);

  std::vector<std::string> tensorFiles = readLines(options_.tensorList);
  std::size_t inputBytes = static_cast<std::size_t>(options_.batchSize) * options_.channels *
                           options_.frames * options_.height * options_.width * sizeof(float);
  BinaryTensorCalibrator calibrator(tensorFiles, options_.batchSize, inputBytes, options_.cache);
  config->setInt8Calibrator(&calibrator);

  std::cout << "[INFO] ONNX: " << options_.onnx << '\n';
  std::cout << "[INFO] Engine: " << options_.engine << '\n';
  std::cout << "[INFO] Cache: " << options_.cache << '\n';
  std::cout << "[INFO] Calibration tensors: " << tensorFiles.size() << '\n';

  TrtUniquePtr<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *config));
  if (!serialized) {
    throw std::runtime_error("TensorRT failed to build INT8 engine");
  }

  writeBinary(options_.engine, serialized->data(), serialized->size());
  std::cout << "[OK] Wrote INT8 engine: " << options_.engine << '\n';
}
