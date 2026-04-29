cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

# Find HIP compiler if not set
if(NOT CMAKE_HIP_COMPILER)
    find_program(amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin
        NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)
    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(measure_flops VERSION 1.0.0 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# Copy HIP flags from CXX if not set
foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

# Find dependencies
find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)
find_package(hip REQUIRED)

# Option to enable AMD SMI support (enabled by default - will auto-detect library)
option(ENABLE_AMD_SMI "Enable AMD SMI memory monitoring support (requires libamd_smi)" ON)

# AMD SMI support - only search if enabled
set(HAVE_AMD_SMI FALSE)
if(ENABLE_AMD_SMI)
    message(STATUS "AMD SMI support: ENABLED (searching for library)")
    
    find_library(AMD_SMI_LIBRARY
        NAMES amd_smi
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES lib lib64
    )
    find_path(AMD_SMI_INCLUDE_DIR
        NAMES amd_smi/amdsmi.h
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES include
    )

    # Check if AMD SMI can actually be compiled
    if(AMD_SMI_LIBRARY AND AMD_SMI_INCLUDE_DIR)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES ${AMD_SMI_INCLUDE_DIR})
        set(CMAKE_REQUIRED_LIBRARIES ${AMD_SMI_LIBRARY})
        check_cxx_source_compiles("
            #include <amd_smi/amdsmi.h>
            int main() {
                amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
                return 0;
            }
        " AMD_SMI_COMPILES)
        
        if(AMD_SMI_COMPILES)
            set(HAVE_AMD_SMI TRUE)
            message(STATUS "AMD SMI found and compiles: ${AMD_SMI_LIBRARY}")
        else()
            message(WARNING "ENABLE_AMD_SMI=ON but AMD SMI headers incomplete")
        endif()
    else()
        message(WARNING "ENABLE_AMD_SMI=ON but libamd_smi not found")
    endif()
else()
    message(STATUS "AMD SMI support: DISABLED (use -DENABLE_AMD_SMI=ON to enable)")
endif()

# Determine ROCm version from path (e.g., /opt/rocm-6.0.0 or rocm-7.13.0a20260324-gfx94X)
if(DEFINED ENV{ROCM_PATH})
    set(_rocm_path "$ENV{ROCM_PATH}")
elseif(ROCM_PATH)
    set(_rocm_path "${ROCM_PATH}")
elseif(rocprofiler-sdk_DIR)
    # Extract from rocprofiler-sdk path
    get_filename_component(_rocm_path "${rocprofiler-sdk_DIR}" DIRECTORY)
    get_filename_component(_rocm_path "${_rocm_path}" DIRECTORY)
    get_filename_component(_rocm_path "${_rocm_path}" DIRECTORY)
else()
    set(_rocm_path "")
endif()

if(_rocm_path)
    # Get the last component of the path
    get_filename_component(_rocm_dirname "${_rocm_path}" NAME)
    # Try to extract version (handle patterns like "rocm-6.0.0" or "rocm-7.13.0a20260324-gfx94X")
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.?[0-9]*[a-zA-Z0-9-]*" _extracted_version "${_rocm_dirname}")
    if(_extracted_version)
        set(ROCM_VERSION_STRING "${_extracted_version}")
    else()
        set(ROCM_VERSION_STRING "${_rocm_dirname}")
    endif()
else()
    set(ROCM_VERSION_STRING "unknown")
endif()

# Print configuration info
message(STATUS "ROCm path: ${ROCM_PATH}")
message(STATUS "ROCm version: ${ROCM_VERSION_STRING}")
message(STATUS "rocprofiler-sdk found: ${rocprofiler-sdk_DIR}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Main library containing core functionality
add_library(measure_flops_lib STATIC
    src/counter_sampler.cpp
    src/instruction_analyzer.cpp
    src/output_formatter.cpp
    src/memory_monitor.cpp
)

# Enable PIC for static lib (needed when linking into shared lib)
set_target_properties(measure_flops_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(measure_flops_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(measure_flops_lib
    PUBLIC
        rocprofiler-sdk::rocprofiler-sdk
        hip::host
    PRIVATE
        Threads::Threads
)

# Add AMD SMI if available
if(HAVE_AMD_SMI)
    target_include_directories(measure_flops_lib PRIVATE 
        ${AMD_SMI_INCLUDE_DIR}
    )
    target_link_libraries(measure_flops_lib PRIVATE 
        ${AMD_SMI_LIBRARY}
    )
    target_compile_definitions(measure_flops_lib PRIVATE HAVE_AMD_SMI)
endif()

target_compile_options(measure_flops_lib PRIVATE
    -Wall -Wextra
)

# Main executable (for info commands like --list-gpus, --list-counters)
add_executable(measure_flops src/main.cpp)

target_link_libraries(measure_flops
    PRIVATE
        measure_flops_lib
        Threads::Threads
)

target_compile_options(measure_flops PRIVATE
    -Wall -Wextra
)

# Tool library for ROCP_TOOL_LIBRARIES injection
set(LIBNAME fp64_monitor_tool_${ROCM_VERSION_STRING})
add_library(${LIBNAME} SHARED 
    tools/fp64_monitor_tool.cpp
    tools/preload.cpp
)

target_link_libraries(${LIBNAME}
    PRIVATE
        measure_flops_lib
        rocprofiler-sdk::rocprofiler-sdk
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

target_compile_options(${LIBNAME} PRIVATE
    -Wall -Wextra
)

target_compile_definitions(${LIBNAME} PRIVATE
    ROCM_VERSION_STRING="${ROCM_VERSION_STRING}"
    FP64_MONITOR_VERSION="${PROJECT_VERSION}"
    $<$<BOOL:${HAVE_AMD_SMI}>:HAVE_AMD_SMI>
)

set_target_properties(${LIBNAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    INSTALL_RPATH_USE_LINK_PATH ON
)

# Optional: Build tests
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Install targets
install(TARGETS measure_flops ${LIBNAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/tools/fp64_monitor.sh
    ${PROJECT_BINARY_DIR}/tools/fp64_monitor.sh @ONLY)

# Install wrapper script and analysis tools
install(PROGRAMS ${PROJECT_BINARY_DIR}/tools/fp64_monitor.sh
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(PROGRAMS tests/analyze_results.py tests/plot_timeseries.py scripts/job.flux
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install documentation and requirements
install(FILES README.md requirements.txt
    DESTINATION share/doc/fp64_monitor
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ
)
