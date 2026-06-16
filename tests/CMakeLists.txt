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
target_link_libraries(test_instruction_analyzer PRIVATE fp64monitor_lib)
target_compile_options(test_instruction_analyzer PRIVATE -Wall -Wextra)

# Direct MFMA test for GEMM analysis validation (inline assembly)
set_source_files_properties(test_mfma_direct.cpp PROPERTIES LANGUAGE HIP)
add_executable(test_mfma_direct test_mfma_direct.cpp)
set_target_properties(test_mfma_direct PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
target_compile_options(test_mfma_direct PRIVATE -O3)

# SpMV test for FP64, FP32, and INT counter validation
set_source_files_properties(test_spmv.cpp PROPERTIES LANGUAGE HIP)
add_executable(test_spmv test_spmv.cpp)
set_target_properties(test_spmv PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
target_compile_options(test_spmv PRIVATE -O3)

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
else()
    message(STATUS "rocWMMA not found - skipping test_mfma_rocwmma")
endif()

# Basic FP64 test

add_test(
    NAME stream64
    COMMAND ${PROJECT_BINARY_DIR}/bin/fp64_monitor.sh --rocm ${ROCM_VERSION_STRING} --counters fp64 --interval 100 -- ${PROJECT_BINARY_DIR}/tests/stream_triad_fp64
)
add_test(NAME verify_stream64
    COMMAND ${CMAKE_COMMAND} -E cat "${CMAKE_CURRENT_BINARY_DIR}/fp64_monitor_output/fp64_report.txt"
)
set_tests_properties(verify_stream64 PROPERTIES DEPENDS stream64)
add_test(NAME clean_stream64 COMMAND ${CMAKE_COMMAND} -E rm -r ${CMAKE_CURRENT_BINARY_DIR}/fp64_monitor_output)
set_tests_properties(clean_stream64 PROPERTIES DEPENDS verify_stream64)

# Test all counters

set(TEST_LIST "fp64" "fp32")
set(COUNTER_LIST "fp64" "fp32" "fp16" "int" "salu" "highlevel" "highlevel2")
foreach(MYTEST IN LISTS TEST_LIST)
    foreach(MYCOUNTER IN LISTS COUNTER_LIST)
        add_test(
            NAME test_${MYTEST}_${MYCOUNTER}
            COMMAND ${PROJECT_BINARY_DIR}/bin/fp64_monitor.sh --rocm ${ROCM_VERSION_STRING} --counters ${MYCOUNTER} --interval 100 -- ${PROJECT_BINARY_DIR}/tests/stream_triad_${MYTEST}
        )
        add_test(NAME verify_${MYTEST}_${MYCOUNTER}
            COMMAND ${CMAKE_COMMAND} -E cat "${CMAKE_CURRENT_BINARY_DIR}/fp64_monitor_output/fp64_report.txt"
        )
        set_tests_properties(verify_${MYTEST}_${MYCOUNTER} PROPERTIES DEPENDS ${MYTEST}_${MYCOUNTER})
        add_test(NAME clean_${MYTEST}_${MYCOUNTER} COMMAND ${CMAKE_COMMAND} -E rm -r ${CMAKE_CURRENT_BINARY_DIR}/fp64_monitor_output)
        set_tests_properties(clean_${MYTEST}_${MYCOUNTER} PROPERTIES DEPENDS verify_${MYTEST}_${MYCOUNTER})
    endforeach()
endforeach()


