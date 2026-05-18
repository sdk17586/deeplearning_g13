# TensorRT INT8 Calibrator

Small C++ package for building a TensorRT INT8 engine from an ONNX model and
binary calibration tensors.

## Layout

```text
.
├── CMakeLists.txt
├── CMakePresets.json
├── src/
│   ├── app/          # CLI entry point
│   ├── calibration/  # TensorRT INT8 calibrator implementation
│   ├── engine/       # ONNX parsing and TensorRT engine build orchestration
│   ├── io/           # Text/binary file utilities
│   ├── options/      # Command-line option parsing and defaults
│   └── tensorrt/     # TensorRT logger and CUDA/TensorRT helpers
├── docs/             # Package documentation
├── build-debug/      # Debug build output
└── build-release/    # Release build output
```

Each feature directory keeps its `.cpp` and `.hpp` files together so related
code is easy to inspect as a unit.

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
