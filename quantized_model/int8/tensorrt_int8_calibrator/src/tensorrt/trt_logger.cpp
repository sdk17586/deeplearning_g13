#include "trt_logger.hpp"

#include <iostream>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TRT] " << msg << '\n';
  }
}
