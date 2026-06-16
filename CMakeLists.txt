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

project(fp64monitor VERSION 1.0.0 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

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
option(ENABLE_AMD_SMI "Enable AMD SMI memory monitoring support (requires libamd_smi)" OFF)

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

# ROCm major version label for tool library / wrapper (e.g. libfp64_monitor_tool_7.so).
# Derived from ROCM_PATH basename (e.g. rocm-7.13.0a... -> 7); falls back to HIP major, else "unknown".
if(DEFINED ENV{ROCM_PATH})
    set(_rocm_path "$ENV{ROCM_PATH}")
elseif(ROCM_PATH)
    set(_rocm_path "${ROCM_PATH}")
elseif(rocprofiler-sdk_DIR)
    get_filename_component(_rocm_path "${rocprofiler-sdk_DIR}" DIRECTORY)
    get_filename_component(_rocm_path "${_rocm_path}" DIRECTORY)
    get_filename_component(_rocm_path "${_rocm_path}" DIRECTORY)
else()
    set(_rocm_path "")
endif()

set(ROCM_VERSION_STRING "")
if(_rocm_path)
    get_filename_component(_rocm_dirname "${_rocm_path}" NAME)
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.?[0-9]*[a-zA-Z0-9-]*" _extracted_version "${_rocm_dirname}")
    if(_extracted_version)
        string(REGEX MATCH "^[0-9]+" ROCM_VERSION_STRING "${_extracted_version}")
    endif()
endif()

if(NOT ROCM_VERSION_STRING MATCHES "^[0-9]+$")
    if(DEFINED hip_VERSION AND hip_VERSION)
        string(REGEX MATCH "^[0-9]+" _hip_major "${hip_VERSION}")
        if(_hip_major MATCHES "^[0-9]+$")
            set(ROCM_VERSION_STRING "${_hip_major}")
        endif()
    elseif(DEFINED HIP_VERSION AND HIP_VERSION)
        string(REGEX MATCH "^[0-9]+" _hip_major "${HIP_VERSION}")
        if(_hip_major MATCHES "^[0-9]+$")
            set(ROCM_VERSION_STRING "${_hip_major}")
        endif()
    endif()
endif()

if(NOT ROCM_VERSION_STRING MATCHES "^[0-9]+$")
    set(ROCM_VERSION_STRING "unknown")
endif()

message(STATUS "ROCm path: ${ROCM_PATH}")
message(STATUS "ROCm major (tool suffix): ${ROCM_VERSION_STRING}")
message(STATUS "rocprofiler-sdk found: ${rocprofiler-sdk_DIR}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Main library containing core functionality
add_library(fp64monitor_lib STATIC
    src/counter_sampler.cpp
    src/instruction_analyzer.cpp
    src/output_formatter.cpp
    src/memory_monitor.cpp
)

# Enable PIC for static lib (needed when linking into shared lib)
set_target_properties(fp64monitor_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(fp64monitor_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(fp64monitor_lib
    PUBLIC
        rocprofiler-sdk::rocprofiler-sdk
        hip::host
    PRIVATE
        Threads::Threads
)

# Add AMD SMI if available
if(HAVE_AMD_SMI)
    target_include_directories(fp64monitor_lib PRIVATE 
        ${AMD_SMI_INCLUDE_DIR}
    )
    target_link_libraries(fp64monitor_lib PRIVATE 
        ${AMD_SMI_LIBRARY}
    )
    target_compile_definitions(fp64monitor_lib PRIVATE HAVE_AMD_SMI)
endif()

target_compile_options(fp64monitor_lib PRIVATE
    -Wall -Wextra
)

# Main executable (for info commands like --list-gpus, --list-counters)
add_executable(fp64monitor src/main.cpp)

target_link_libraries(fp64monitor
    PRIVATE
        fp64monitor_lib
        Threads::Threads
)

target_compile_options(fp64monitor PRIVATE
    -Wall -Wextra
)

# Tool library for ROCP_TOOL_LIBRARIES injection
set(LIBNAME fp64_monitor_tool_${ROCM_VERSION_STRING})
add_library(${LIBNAME} SHARED 
    src/fp64_monitor_tool.cpp
    src/preload.cpp
)

set_target_properties(fp64_monitor_tool_${ROCM_VERSION_STRING} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_link_libraries(${LIBNAME}
    PRIVATE
        fp64monitor_lib
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
    BUILD_WITH_INSTALL_RPATH ON
)

# Optional: Build tests
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install targets
install(TARGETS fp64monitor ${LIBNAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/src/fp64_monitor.sh
    ${PROJECT_BINARY_DIR}/bin/fp64_monitor.sh @ONLY)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fp64_monitor_plot_timeseries.py"
    "${CMAKE_CURRENT_BINARY_DIR}/bin/fp64_monitor_plot_timeseries.py"
    COPYONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fp64_monitor_analyze_results.py"
    "${CMAKE_CURRENT_BINARY_DIR}/bin/fp64_monitor_analyze_results.py"
    COPYONLY
)

# Install wrapper script and analysis tools
install(PROGRAMS ${PROJECT_BINARY_DIR}/bin/fp64_monitor.sh
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(PROGRAMS tests/fp64_monitor_analyze_results.py tests/fp64_monitor_plot_timeseries.py scripts/job.flux
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install documentation and requirements
install(FILES README.md requirements.txt
    DESTINATION share/doc/fp64monitor
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ
)
