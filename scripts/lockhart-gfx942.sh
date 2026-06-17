#!/bin/bash -e 
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE in the project root for license information.

rm -rf $(pwd)/install

versions=("6.4.3" "7.13.0-gfx94x")
for version in "${versions[@]}"; do
    module load rocm/${version}
    rm -rf $(pwd)/build
    cmake -B $(pwd)/build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON \
    -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DGPU_TARGETS=gfx942
    cmake --build $(pwd)/build --parallel 8
    ctest --test-dir $(pwd)/build --rerun-failed --output-on-failure
    cmake --build $(pwd)/build --target install
    module unload rocm
done
