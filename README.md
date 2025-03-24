Install cmake, make a build directory, and generate a Makefile for debug/release builds
Debug builds will require the validation layer (VK_LAYER_KHRONOS_validation)
If you make changes to shaders, run ./compile.sh to compile

Generate the Makefile
cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake -DCMAKE_BUILD_TYPE=Release -B build

Build
cmake --build build

Run
./build/VulkanApp
