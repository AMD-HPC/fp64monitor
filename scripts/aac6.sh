#!/bin/bash -e 
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE in the project root for license information.

#rm -rf $(pwd)/install

versions=("6.4.3" "7.2.3")
for version in "${versions[@]}"; do
    module load rocm/${version}
    rm -rf $(pwd)/build
    cmake -B $(pwd)/build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON \
    -DCMAKE_CXX_COMPILER=`which amdclang++` \
    -DCMAKE_INSTALL_PREFIX=/shared/apps/ubuntu/opt/fp64monitor -DGPU_TARGETS=gfx942
    cmake --build $(pwd)/build --parallel 8
    ctest --test-dir $(pwd)/build --rerun-failed --output-on-failure
    cmake --build $(pwd)/build --target install
    module unload rocm
done

chgrp -Rf adm /shared/apps/ubuntu/opt/fp64monitor
chmod -Rf g=u /shared/apps/ubuntu/opt/fp64monitor

