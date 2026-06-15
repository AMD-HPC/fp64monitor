module load rocm/7.13.0-gfx94x
rm -rf build install
PATH=${ROCM_PATH}/llvm/bin:$PATH
cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=./install -DGPU_TARGETS=gfx942
cmake --build build --target install --parallel 8
