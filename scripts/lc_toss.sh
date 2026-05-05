rm -rf install
for v in 7.2.1 6.4.3 ; do
    rm -rf build-${v}
    module load rocm/${v}
    PATH=${ROCM_PATH}/llvm/bin:$PATH
    cmake -B build-${v} -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_AMD_SMI=OFF \
    -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=./install -DGPU_TARGETS=gfx942
    cmake --build build-${v}/ --target install --parallel 8
    module unload rocm/${v}
done

# Not necessary
#for v in 6.4.1-magic 6.4.2-magic 6.4.3-magic 7.0.1-magic 7.2.0-magic 7.2.1-magic ; do
#    rm -rf build-${v} install-${v}
#    module load rocmcc/${v}
#    PATH=${ROCM_PATH}/llvm/bin:$PATH
#    cmake -B build-${v} -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#    -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=./install-${v} -DGPU_TARGETS=gfx942
#    cmake --build build-${v}/ --target install --parallel 8
#    module unload rocmcc/${v}
#done


