# Instructions

- Prerequisites: Install cmake and make a build directory
- Debug builds will require the validation layer (VK_LAYER_KHRONOS_validation)
- If you make changes to shaders, run ./compile.sh to compile with glslc

```code
// generate the Makefile (run one)
cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake -DCMAKE_BUILD_TYPE=Release -B build

// build
cmake --build build

// run
./build/VulkanApp
```
