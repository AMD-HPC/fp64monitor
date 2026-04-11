
for v in 6.4.1 6.4.3 7.2.0 ; do
    rm -rf build-${v} install-${v}
    module load rocm/${v}
    PATH=${ROCM_PATH}/llvm/bin:$PATH
    cmake -B build-${v} -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=./install-${v} -DGPU_TARGETS=gfx942
    cmake --build build-${v}/ --target install --parallel 8
    module unload rocm/${v}
done

for v in 7.13.0a20260324-gfx94X ; do
    rm -rf build-${v} install-${v}
    source /home/kevihuck/src/therock-tarball/sourceme-${v}.sh
    deactivate
    cmake -B build-${v} -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_INSTALL_PREFIX=./install-${v} -DGPU_TARGETS=gfx942
    cmake --build build-${v}/ --target install --parallel 8
done


