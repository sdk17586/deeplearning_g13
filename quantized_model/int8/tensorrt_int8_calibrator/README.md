# TensorRT INT8 Calibrator

Small C++ package for building a TensorRT INT8 engine from an ONNX model and
binary calibration tensors.

## Build

```sh
cmake --preset=debug
cmake --build build-debug
```

For optimized builds:

```sh
cmake --preset=release
cmake --build build-release
```

The debug preset writes `build-debug/compile_commands.json`, and `.clangd` points
clangd at that build directory.

If TensorRT is installed in a non-system location, configure with
`TENSORRT_ROOT` set to the TensorRT install prefix.
