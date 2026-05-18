#include <exception>
#include <iostream>

#include "calibrator_options.hpp"
#include "int8_engine_builder.hpp"

int main(int argc, char** argv) {
  try {
    CalibratorOptions options = CalibratorOptions::parse(argc, argv);
    Int8EngineBuilder builder(options);
    builder.build();
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "[ERROR] " << exc.what() << '\n';
    return 1;
  }
}
