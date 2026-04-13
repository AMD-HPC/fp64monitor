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

# Option to enable ROCm SMI support (disabled by default)
option(ENABLE_ROCM_SMI "Enable ROCm SMI memory monitoring support (requires librocm_smi64)" OFF)

# ROCm SMI support - only search if enabled
set(HAVE_ROCM_SMI FALSE)
if(ENABLE_ROCM_SMI)
    message(STATUS "ROCm SMI support: ENABLED (searching for library)")
    
    find_library(ROCM_SMI_LIBRARY
        NAMES rocm_smi64
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES lib lib64
    )
    find_path(ROCM_SMI_INCLUDE_DIR
        NAMES rocm_smi/rocm_smi.h
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES include
    )

    # Find libdrm (may be in rocm_sysdeps for TheRock builds)
    find_path(LIBDRM_INCLUDE_DIR
        NAMES libdrm/drm.h
        HINTS 
            ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
            ${ROCM_PATH}/lib/rocm_sysdeps ENV ROCM_PATH
        PATH_SUFFIXES include lib/rocm_sysdeps/include
    )

    find_library(LIBDRM_LIBRARY
        NAMES drm rocm_sysdeps_drm
        HINTS 
            ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
            ${ROCM_PATH}/lib/rocm_sysdeps ENV ROCM_PATH
        PATH_SUFFIXES lib lib64 lib/rocm_sysdeps/lib
    )

    if(LIBDRM_INCLUDE_DIR)
        message(STATUS "libdrm headers found: ${LIBDRM_INCLUDE_DIR}")
    endif()
    if(LIBDRM_LIBRARY)
        message(STATUS "libdrm library found: ${LIBDRM_LIBRARY}")
    endif()

    # Check if ROCm SMI can actually be compiled (may fail if libdrm headers are missing)
    if(ROCM_SMI_LIBRARY AND ROCM_SMI_INCLUDE_DIR)
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES ${ROCM_SMI_INCLUDE_DIR})
        if(LIBDRM_INCLUDE_DIR)
            list(APPEND CMAKE_REQUIRED_INCLUDES ${LIBDRM_INCLUDE_DIR})
        endif()
        set(CMAKE_REQUIRED_LIBRARIES ${ROCM_SMI_LIBRARY})
        if(LIBDRM_LIBRARY)
            list(APPEND CMAKE_REQUIRED_LIBRARIES ${LIBDRM_LIBRARY})
        endif()
        check_cxx_source_compiles("
            #include <rocm_smi/rocm_smi.h>
            int main() {
                rsmi_status_t status = rsmi_init(0);
                return 0;
            }
        " ROCM_SMI_COMPILES)
        
        if(ROCM_SMI_COMPILES)
            set(HAVE_ROCM_SMI TRUE)
            message(STATUS "ROCm SMI found and compiles: ${ROCM_SMI_LIBRARY}")
        else()
            message(WARNING "ENABLE_ROCM_SMI=ON but ROCm SMI headers incomplete (missing libdrm?)")
        endif()
    else()
        message(WARNING "ENABLE_ROCM_SMI=ON but librocm_smi64 not found")
    endif()
else()
    message(STATUS "ROCm SMI support: DISABLED (use -DENABLE_ROCM_SMI=ON to enable)")
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

# Add ROCm SMI if available
if(HAVE_ROCM_SMI)
    target_include_directories(measure_flops_lib PRIVATE 
        ${ROCM_SMI_INCLUDE_DIR}
        ${LIBDRM_INCLUDE_DIR}
    )
    target_link_libraries(measure_flops_lib PRIVATE 
        ${ROCM_SMI_LIBRARY}
        ${LIBDRM_LIBRARY}
    )
    target_compile_definitions(measure_flops_lib PRIVATE HAVE_ROCM_SMI)
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
add_library(fp64_monitor_tool SHARED 
    tools/fp64_monitor_tool.cpp
    tools/preload.cpp
)

target_link_libraries(fp64_monitor_tool
    PRIVATE
        measure_flops_lib
        rocprofiler-sdk::rocprofiler-sdk
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

target_compile_options(fp64_monitor_tool PRIVATE
    -Wall -Wextra
)

target_compile_definitions(fp64_monitor_tool PRIVATE
    ROCM_VERSION_STRING="${ROCM_VERSION_STRING}"
    FP64_MONITOR_VERSION="${PROJECT_VERSION}"
    $<$<BOOL:${HAVE_ROCM_SMI}>:HAVE_ROCM_SMI>
)

set_target_properties(fp64_monitor_tool PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Optional: Build tests
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Install targets
install(TARGETS measure_flops measure_flops_lib fp64_monitor_tool
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install wrapper script and analysis tools
install(PROGRAMS tools/fp64_monitor.sh
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(PROGRAMS tests/analyze_results.py tests/plot_timeseries.py
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install documentation and requirements
install(FILES README.md requirements.txt
    DESTINATION share/doc/fp64_monitor
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ
)
