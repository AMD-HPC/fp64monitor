#!/bin/bash -e 
rm -rf $(pwd)/install

versions=("6.4.3" "7.13.0-gfx90a")
for version in "${versions[@]}"; do
    module load rocm/${version}
    rm -rf $(pwd)/build
    set -x
    cmake -B $(pwd)/build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON \
    -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DGPU_TARGETS=gfx90a
    cmake --build $(pwd)/build --parallel 8
    ctest --test-dir $(pwd)/build --rerun-failed --output-on-failure
    cmake --build $(pwd)/build --target install
    set +x
    module unload rocm
done
