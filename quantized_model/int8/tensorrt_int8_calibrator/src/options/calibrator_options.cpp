#include "calibrator_options.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

CalibratorOptions CalibratorOptions::parse(int argc, char** argv) {
  CalibratorOptions args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto requireValue = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--onnx") args.onnx = requireValue(key);
    else if (key == "--engine") args.engine = requireValue(key);
    else if (key == "--cache") args.cache = requireValue(key);
    else if (key == "--tensor-list") args.tensorList = requireValue(key);
    else if (key == "--batch-size") args.batchSize = std::stoi(requireValue(key));
    else if (key == "--workspace-mib") args.workspaceMiB = std::stoi(requireValue(key));
    else if (key == "--help") {
      std::cout
          << "Usage: trt_int8_calibrator [--onnx path] [--engine path] [--cache path]\n"
          << "                           [--tensor-list path] [--batch-size 1]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  return args;
}
