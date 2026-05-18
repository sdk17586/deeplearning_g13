#pragma once

#include <NvInfer.h>

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};
