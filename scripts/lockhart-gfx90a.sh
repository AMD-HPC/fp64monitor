module load rocm/7.13.0-gfx90a
rm -rf build install
cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=./install -DGPU_TARGETS=gfx90a
cmake --build build --target install --parallel 8
