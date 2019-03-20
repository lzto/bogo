mkdir build
pushd build
cmake ../ \
    -DLLVM_DIR=/opt/toolchain/llvm \
    -DLLVM_ROOT=/opt/toolchain/llvm \
    -DCMAKE_BUILD_TYPE=Debug
popd
pushd rtlib
make
popd
