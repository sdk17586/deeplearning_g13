#pragma once

#include "calibrator_options.hpp"

class Int8EngineBuilder {
 public:
  explicit Int8EngineBuilder(CalibratorOptions options);

  void build();

 private:
  CalibratorOptions options_;
};
