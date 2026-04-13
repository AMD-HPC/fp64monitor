# CMakeLists.txt for test programs

# Set GPU architecture - detect or use gfx942 for MI300
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "GPU architectures to compile for")
endif()

# STREAM Triad FP64 test (HIP workload for validation)
set_source_files_properties(stream_triad_fp64.hip PROPERTIES LANGUAGE HIP)
add_executable(stream_triad_fp64 stream_triad_fp64.hip)
set_target_properties(stream_triad_fp64 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
target_compile_options(stream_triad_fp64 PRIVATE -O3)

# STREAM Triad FP32 test (HIP workload for FP32 validation)
set_source_files_properties(stream_triad_fp32.hip PROPERTIES LANGUAGE HIP)
add_executable(stream_triad_fp32 stream_triad_fp32.hip)
set_target_properties(stream_triad_fp32 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
target_compile_options(stream_triad_fp32 PRIVATE -O3)

# Unit test for InstructionAnalyzer
add_executable(test_instruction_analyzer test_instruction_analyzer.cpp)
target_link_libraries(test_instruction_analyzer PRIVATE measure_flops_lib)
target_compile_options(test_instruction_analyzer PRIVATE -Wall -Wextra)

# Direct MFMA test for GEMM analysis validation (inline assembly)
set_source_files_properties(test_mfma_direct.cpp PROPERTIES LANGUAGE HIP)
add_executable(test_mfma_direct test_mfma_direct.cpp)
set_target_properties(test_mfma_direct PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
target_compile_options(test_mfma_direct PRIVATE -O3)

# rocWMMA MFMA test for GEMM analysis validation (high-level API)
# rocWMMA is a header-only library included with ROCm
find_path(ROCWMMA_INCLUDE_DIR
    NAMES rocwmma/rocwmma.hpp
    HINTS ${ROCM_PATH}/include $ENV{ROCM_PATH}/include /opt/rocm/include
)

if(ROCWMMA_INCLUDE_DIR)
    message(STATUS "Found rocWMMA: ${ROCWMMA_INCLUDE_DIR}")
    set_source_files_properties(test_mfma_rocwmma.cpp PROPERTIES LANGUAGE HIP)
    add_executable(test_mfma_rocwmma test_mfma_rocwmma.cpp)
    set_target_properties(test_mfma_rocwmma PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_include_directories(test_mfma_rocwmma PRIVATE ${ROCWMMA_INCLUDE_DIR})
    target_compile_options(test_mfma_rocwmma PRIVATE -O3)
    install(TARGETS test_mfma_rocwmma DESTINATION bin)
else()
    message(STATUS "rocWMMA not found - skipping test_mfma_rocwmma")
endif()

# Install test programs
install(TARGETS stream_triad_fp64 stream_triad_fp32 test_instruction_analyzer test_mfma_direct DESTINATION bin)
