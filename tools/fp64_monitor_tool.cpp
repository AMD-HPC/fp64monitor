// FP64 Monitor Tool - rocprofiler-sdk tool library
//
// This tool monitors FP64 instruction usage on AMD GPUs.
// Load via ROCP_TOOL_LIBRARIES to profile a HIP application.
//
// Usage:
//   ROCP_TOOL_LIBRARIES=/path/to/libfp64_monitor_tool.so ./your_hip_app
//
// Environment variables:
//   FP64_MONITOR_INTERVAL  - Sampling interval in ms (default: 1000)
//   FP64_MONITOR_OUTPUT    - Output file pattern (default: fp64_monitor_<rank>.csv)
//                            Supports: %r=MPI rank, %p=PID, %h=hostname
//                            Use "stdout" for console output
//   FP64_MONITOR_DIR       - Output directory (default: fp64_monitor_output)
//   FP64_MONITOR_GPU       - GPU ID to monitor (default: 0)
//   FP64_MONITOR_VERBOSE   - Set to 1 for verbose output
//   FP64_MONITOR_FORMAT    - Output format: text, csv, json (default: csv)
//   FP64_MONITOR_TMPDIR    - Write to fast local storage (e.g., /dev/shm), copy
//                            to final location at exit (reduces I/O on parallel FS)
//   FP64_MONITOR_USE_ROCMSMI - Set to 1 to enable rocm-smi memory monitoring

#include "measure_flops/counter_sampler.hpp"
#include "measure_flops/instruction_analyzer.hpp"
#include "measure_flops/memory_monitor.hpp"
#include "measure_flops/output_formatter.hpp"
#include "measure_flops/rocprofiler_compat.hpp"
#include "measure_flops/types.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using namespace measure_flops;

// Safe string-to-long conversion with validation
// Returns default_val if the string is null, empty, or not a valid integer
// Also returns default_val if the value is out of the specified range
long safe_strtol(const char* str, long default_val, long min_val = LONG_MIN, long max_val = LONG_MAX) {
    if (str == nullptr || str[0] == '\0') {
        return default_val;
    }
    
    char* endptr = nullptr;
    errno = 0;
    long val = std::strtol(str, &endptr, 10);
    
    // Check for conversion errors
    if (errno == ERANGE || endptr == str || *endptr != '\0') {
        return default_val;
    }
    
    // Check range
    if (val < min_val || val > max_val) {
        return default_val;
    }
    
    return val;
}

// Try to detect MPI rank from common environment variables
// Returns -1 if not found
int get_mpi_rank() {
    // Check various MPI implementations' rank environment variables
    const char* rank_vars[] = {
        "OMPI_COMM_WORLD_RANK",      // OpenMPI
        "PMI_RANK",                   // MPICH, Cray MPICH
        "SLURM_PROCID",              // SLURM
        "MV2_COMM_WORLD_RANK",       // MVAPICH2
        "ALPS_APP_PE",               // Cray ALPS
        "PALS_RANKID",               // HPE Cray PALS
        "MPI_LOCALRANKID",           // Intel MPI (local)
        "PMI_LOCAL_RANK",            // PMI local rank
        nullptr
    };
    
    for (const char** var = rank_vars; *var != nullptr; ++var) {
        if (const char* val = std::getenv(*var)) {
            long rank = safe_strtol(val, -1, 0, INT_MAX);
            return static_cast<int>(rank);
        }
    }
    return -1;
}

// Sanitize a string to be safe for use as a path component
// Removes or replaces characters that could cause path traversal
std::string sanitize_path_component(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        
        // Skip path separators
        if (c == '/' || c == '\\') {
            result += '_';
            continue;
        }
        
        // Skip .. sequences (path traversal)
        if (c == '.' && i + 1 < input.size() && input[i + 1] == '.') {
            result += '_';
            ++i;  // Skip next dot too
            continue;
        }
        
        // Skip null bytes
        if (c == '\0') {
            continue;
        }
        
        result += c;
    }
    
    // Don't allow empty result
    if (result.empty()) {
        return "unknown";
    }
    
    return result;
}

// Get hostname (sanitized for safe use in filenames)
std::string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[sizeof(hostname) - 1] = '\0';
        return sanitize_path_component(std::string(hostname));
    }
    return "unknown";
}

// Recursively create directories (like mkdir -p)
// Returns true on success, false on failure
// Thread-safe: uses mkdir with EEXIST handling
bool mkdir_recursive(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    
    // Check if it already exists
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    
    // Find the parent directory
    size_t last_slash = path.rfind('/');
    if (last_slash != std::string::npos && last_slash > 0) {
        std::string parent = path.substr(0, last_slash);
        if (!mkdir_recursive(parent)) {
            return false;
        }
    }
    
    // Create this directory
    // mkdir is atomic on POSIX, EEXIST means another process created it
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    
    return true;
}

// Get output directory name from environment or use default
// Creates the directory (recursively) if it doesn't exist
// All ranks attempt creation (mkdir with EEXIST is safe for concurrent calls)
std::string get_output_directory() {
    const char* dir_env = std::getenv("FP64_MONITOR_DIR");
    std::string dir_name = dir_env ? dir_env : "fp64_monitor_output";
    
    // Try to create the directory recursively
    struct stat st;
    if (stat(dir_name.c_str(), &st) != 0) {
        if (!mkdir_recursive(dir_name)) {
            std::cerr << "[FP64 Monitor] Warning: Could not create directory " 
                    << dir_name << ", using current directory\n";
            return "";
        }
    }
    
    return dir_name;
}

// Expand filename pattern with substitutions:
//   %r = MPI rank (or "norank" if not available)
//   %p = PID
//   %h = hostname
// If pattern doesn't contain a path separator, prepend output directory
std::string expand_filename_pattern(const std::string& pattern) {
    std::string result = pattern;
    
    int mpi_rank = get_mpi_rank();
    pid_t pid = getpid();
    std::string hostname = get_hostname();
    
    // Replace %r with MPI rank
    size_t pos;
    while ((pos = result.find("%r")) != std::string::npos) {
        if (mpi_rank >= 0) {
            result.replace(pos, 2, std::to_string(mpi_rank));
        } else {
            result.replace(pos, 2, "norank");
        }
    }
    
    // Replace %p with PID
    while ((pos = result.find("%p")) != std::string::npos) {
        result.replace(pos, 2, std::to_string(pid));
    }
    
    // Replace %h with hostname
    while ((pos = result.find("%h")) != std::string::npos) {
        result.replace(pos, 2, hostname);
    }
    
    // If no directory in path, prepend output directory
    if (result.find('/') == std::string::npos) {
        std::string dir = get_output_directory();
        if (!dir.empty()) {
            result = dir + "/" + result;
        }
    }
    
    return result;
}

// Copy a file from src to dst with verification
// Returns true on success, false on failure
bool copy_file(const std::string& src, const std::string& dst) {
    // Open source file and get size
    std::ifstream in(src, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    
    std::streamsize src_size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    if (src_size < 0) {
        return false;
    }
    
    // Open destination file
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    
    // Copy in chunks to handle large files and detect write errors
    const size_t buffer_size = 64 * 1024;  // 64KB chunks
    std::vector<char> buffer(buffer_size);
    std::streamsize total_written = 0;
    
    while (in && out && total_written < src_size) {
        in.read(buffer.data(), buffer_size);
        std::streamsize bytes_read = in.gcount();
        
        if (bytes_read > 0) {
            out.write(buffer.data(), bytes_read);
            if (!out) {
                // Write failed (disk full, etc.)
                return false;
            }
            total_written += bytes_read;
        }
    }
    
    // Flush to ensure data is written to disk
    out.flush();
    
    if (!out.good()) {
        return false;
    }
    
    // Close files
    out.close();
    in.close();
    
    // Verify destination file size matches source
    struct stat src_stat, dst_stat;
    if (stat(src.c_str(), &src_stat) != 0 || stat(dst.c_str(), &dst_stat) != 0) {
        return false;
    }
    
    if (src_stat.st_size != dst_stat.st_size) {
        // Size mismatch - copy failed
        return false;
    }
    
    return true;
}

// Generate default output filename
// Uses MPI rank if available, otherwise hostname_pid
// Files are placed in the output directory
std::string generate_default_filename(const std::string& format) {
    std::string extension;
    if (format == "json") {
        extension = ".json";
    } else if (format == "csv") {
        extension = ".csv";
    } else {
        extension = ".txt";
    }
    
    std::string dir = get_output_directory();
    std::string basename;
    
    int mpi_rank = get_mpi_rank();
    if (mpi_rank >= 0) {
        basename = "fp64_monitor_" + std::to_string(mpi_rank) + extension;
    } else {
        basename = "fp64_monitor_" + get_hostname() + "_" + std::to_string(getpid()) + extension;
    }
    
    if (dir.empty()) {
        return basename;
    }
    return dir + "/" + basename;
}

void tool_fini(void*);

struct ToolState {
    rocprofiler_client_id_t* client_id = nullptr;
    rocprofiler_client_finalize_t finalize_func = nullptr;
    
    rocprofiler_context_id_t context{};
    rocprofiler_buffer_id_t buffer{};
    rocprofiler_compat::config_id_t profile{.handle = 0};
    rocprofiler_agent_id_t agent_id{};
    
    std::unordered_map<std::string, rocprofiler_counter_id_t> counter_name_to_id;
    std::unordered_map<uint64_t, std::string> counter_id_to_name;
    
    std::vector<std::string> configured_counters;
    size_t expected_record_count = 0;
    
    std::unique_ptr<std::thread> sampling_thread;
    std::atomic<bool> running{false};
    std::atomic<bool> sampling{false};
    std::atomic<bool> shutdown_complete{false};
    
    std::ostream* output = &std::cout;
    std::unique_ptr<std::ofstream> output_file;
    std::unique_ptr<OutputFormatter> formatter;
    
    MonitorConfig config;
    
    std::chrono::steady_clock::time_point start_time;
    uint64_t sample_count = 0;
    
    std::vector<InstructionMix> collected_mixes;
    
    std::mutex output_mutex;
    std::mutex data_mutex;
    
    // Synchronization for clean shutdown
    std::mutex sampling_mutex;
    std::condition_variable sampling_cv;
    
    // Memory monitoring
    MemoryMonitor memory_monitor;
    
    // Temp file support (for writing to fast local storage)
    std::string temp_filepath;   // Path to temp file (if using tmpdir)
    std::string final_filepath;  // Final destination path
    
    // Destructor to ensure clean thread shutdown
    // This prevents std::terminate() if tool_fini wasn't called
    ~ToolState() {
        tool_fini(nullptr);
        /*
        // Signal thread to stop if not already done
        running.store(false);
        
        // Only attempt cleanup if tool_fini hasn't already done it
        if (!shutdown_complete.load()) {
            if (sampling_thread && sampling_thread->joinable()) {
                // Detach the thread to prevent std::terminate()
                // This is safer than join() during abnormal shutdown
                // since the thread may be blocked on rocprofiler calls
                sampling_thread->detach();
            }
            
            // Skip rocprofiler cleanup in destructor - it can cause segfaults
            // if the rocprofiler runtime has already been torn down
            // The OS will clean up when the process exits
        }
        */
    }
};

ToolState g_state;

#define ROCP_CHECK(result, msg)                                                 \
    do {                                                                        \
        rocprofiler_status_t status_ = (result);                                \
        if (status_ != ROCPROFILER_STATUS_SUCCESS) {                            \
            std::cerr << "[FP64 Monitor] " << msg << ": "                       \
                      << rocprofiler_get_status_string(status_) << std::endl;   \
            return -1;                                                          \
        }                                                                       \
    } while (0)

void populate_counter_map() {
    std::vector<rocprofiler_counter_id_t> counter_ids;

    auto counter_cb = [](rocprofiler_agent_id_t,
                         rocprofiler_counter_id_t* counters,
                         size_t num_counters,
                         void* user_data) -> rocprofiler_status_t {
        auto* ids = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
        for (size_t i = 0; i < num_counters; ++i) {
            ids->push_back(counters[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    rocprofiler_iterate_agent_supported_counters(
        g_state.agent_id, counter_cb, static_cast<void*>(&counter_ids));

    for (const auto& counter_id : counter_ids) {
        rocprofiler_counter_info_v0_t info;
        if (rocprofiler_query_counter_info(
                counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info) == 
            ROCPROFILER_STATUS_SUCCESS && info.name) {
            g_state.counter_name_to_id[info.name] = counter_id;
            g_state.counter_id_to_name[counter_id.handle] = info.name;
        }
    }
}

size_t get_counter_instance_count(rocprofiler_counter_id_t counter_id) {
#ifdef ROCPROFILER_SDK_V1
    // ROCm 7.x - use v1 counter info structure
    rocprofiler_counter_info_v1_t info;
    info.size = sizeof(info);
    if (rocprofiler_query_counter_info(
            counter_id, ROCPROFILER_COUNTER_INFO_VERSION_1, &info) == 
        ROCPROFILER_STATUS_SUCCESS) {
        return info.dimensions_instances_count;
    }
#else
    // ROCm 6.x - use deprecated query function
    size_t instance_count = 1;
    if (rocprofiler_query_counter_instance_count(
            g_state.agent_id, counter_id, &instance_count) ==
        ROCPROFILER_STATUS_SUCCESS) {
        return instance_count;
    }
#endif
    return 1;
}

bool configure_counters() {
    // Select counter set based on environment variable
    std::vector<std::string> counters;
    std::string counter_set_name = "fp64";
    const char* counter_set = std::getenv("FP64_MONITOR_COUNTERS");
    if (counter_set) {
        std::string cs(counter_set);
        if (cs == "fp32") {
            counters = CounterSets::fp32_minimal();
            counter_set_name = "fp32";
        } else if (cs == "fp16") {
            counters = CounterSets::fp16_minimal();
            counter_set_name = "fp16";
        } else if (cs == "int") {
            counters = CounterSets::int_minimal();
            counter_set_name = "int";
        } else {
            counters = CounterSets::fp64_minimal();
        }
    } else {
        counters = CounterSets::fp64_minimal();
    }
    
    if (g_state.config.verbose && get_mpi_rank() <= 0) {
        std::cerr << "[FP64 Monitor] Counter set: " << counter_set_name << "\n";
    }
    
    std::vector<rocprofiler_counter_id_t> counter_ids;
    size_t total_instances = 0;
    
    for (const auto& name : counters) {
        auto it = g_state.counter_name_to_id.find(name);
        if (it != g_state.counter_name_to_id.end()) {
            counter_ids.push_back(it->second);
            total_instances += get_counter_instance_count(it->second);
            g_state.configured_counters.push_back(name);
        }
    }
    
    if (counter_ids.empty()) {
        std::cerr << "[FP64 Monitor] No counters available\n";
        return false;
    }
    
    if (g_state.config.verbose && get_mpi_rank() <= 0) {
        std::cerr << "[FP64 Monitor] Configuring " << counter_ids.size() 
                  << " counters (" << total_instances << " instances)\n";
    }
    
    rocprofiler_status_t status = rocprofiler_compat::create_config(
        g_state.agent_id,
        counter_ids.data(),
        counter_ids.size(),
        &g_state.profile);
    
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] Failed to create counter config: "
                  << rocprofiler_get_status_string(status) << std::endl;
        return false;
    }
    
    g_state.expected_record_count = total_instances;
    return true;
}

std::string decode_counter_name(const rocprofiler_compat::counter_record_t& rec) {
    rocprofiler_counter_id_t counter_id{};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);
    
    auto it = g_state.counter_id_to_name.find(counter_id.handle);
    if (it != g_state.counter_id_to_name.end()) {
        return it->second;
    }
    return "UNKNOWN";
}

Sample take_sample() {
    Sample sample;
    sample.sample_id = ++g_state.sample_count;
    sample.timestamp = std::chrono::steady_clock::now();
    sample.gpu_id = g_state.config.gpu_ids.empty() ? 0 : g_state.config.gpu_ids[0];
    
    std::vector<rocprofiler_compat::counter_record_t> records(g_state.expected_record_count);
    size_t record_count = records.size();
    
    rocprofiler_user_data_t user_data;
    user_data.value = sample.sample_id;
    
    rocprofiler_status_t status = rocprofiler_sample_device_counting_service(
        g_state.context,
        user_data,
        ROCPROFILER_COUNTER_FLAG_NONE,
        records.data(),
        &record_count);
    
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        if (g_state.config.verbose && get_mpi_rank() <= 0) {
            std::cerr << "[FP64 Monitor] Sample " << sample.sample_id 
                      << " failed: " << rocprofiler_get_status_string(status) << "\n";
        }
        return sample;
    }
    
    records.resize(record_count);
    
    for (const auto& rec : records) {
        CounterValue cv;
        cv.name = decode_counter_name(rec);
        cv.value = rec.counter_value;
        cv.instance_id = rec.id;
        sample.counters.push_back(cv);
    }
    
    return sample;
}

void sampling_thread_func() {
    // Wait for HSA to be ready - retry with backoff
    rocprofiler_status_t status;
    int retries = 0;
    const int max_retries = 50;  // 5 seconds max wait
    
    int mpi_rank = get_mpi_rank();
    bool is_rank0 = (mpi_rank <= 0);
    
    while (retries < max_retries && g_state.running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        status = rocprofiler_start_context(g_state.context);
        if (status == ROCPROFILER_STATUS_SUCCESS) {
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] Context started after " << retries << " retries\n";
            }
            break;
        }
        if (status != ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED) {
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] Failed to start context: "
                          << rocprofiler_get_status_string(status) << std::endl;
            }
            return;
        }
        retries++;
    }
    
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] Timeout waiting for HSA to load (running=" 
                  << g_state.running.load() << ")\n";
        return;
    }
    
    // Write header
    {
        std::lock_guard<std::mutex> lock(g_state.output_mutex);
        g_state.formatter->write_header(*g_state.output, g_state.config);
    }
    
    g_state.start_time = std::chrono::steady_clock::now();
    
    while (true) {
        g_state.sampling.store(true);
        
        // Check if shutdown requested - if so, take one final sample then exit
        if (!g_state.running.load()) {
            // Take final sample before exiting
            Sample final_sample = take_sample();
            if (g_state.config.verbose && get_mpi_rank() <= 0) {
                std::cerr << "[FP64 Monitor] Final sample: " << final_sample.counters.size() 
                          << " counter records\n";
            }
            if (!final_sample.counters.empty()) {
                InstructionMix mix = InstructionAnalyzer::analyze(final_sample);
                if (g_state.memory_monitor.is_initialized() && !g_state.memory_monitor.is_shutdown()) {
                    MemoryStats mem = g_state.memory_monitor.sample();
                    mix.gpu_mem_used = mem.gpu_used;
                    mix.gpu_mem_total = mem.gpu_total;
                    mix.vram_used = mem.vram_used;
                    mix.vram_total = mem.vram_total;
                    mix.gtt_used = mem.gtt_used;
                    mix.gtt_total = mem.gtt_total;
                    mix.cpu_mem_resident = mem.cpu_resident;
                    mix.cpu_mem_virtual = mem.cpu_vm_size;
                    mix.gpu_mem_hwm = mem.gpu_used_hwm;
                    mix.vram_hwm = mem.vram_used_hwm;
                    mix.gtt_hwm = mem.gtt_used_hwm;
                    mix.cpu_mem_hwm = mem.cpu_resident_hwm;
                    mix.cpu_vm_hwm = mem.cpu_vm_hwm;
                }
                auto elapsed = std::chrono::duration<double>(
                    final_sample.timestamp - g_state.start_time).count();
                {
                    std::lock_guard<std::mutex> lock(g_state.data_mutex);
                    g_state.collected_mixes.push_back(mix);
                }
                {
                    std::lock_guard<std::mutex> lock(g_state.output_mutex);
                    g_state.formatter->write_sample(*g_state.output, mix, elapsed);
                    g_state.output->flush();
                }
            }
            {
                std::lock_guard<std::mutex> lock(g_state.sampling_mutex);
                g_state.sampling.store(false);
            }
            g_state.sampling_cv.notify_all();
            break;  // Exit the loop
        }
        
        Sample sample = take_sample();
        
        InstructionMix mix;
        if (!sample.counters.empty()) {
            mix = InstructionAnalyzer::analyze(sample);
        }
        mix.sample_id = sample.sample_id;
        mix.gpu_id = sample.gpu_id;
        
        // Always collect memory statistics, even if counter sampling failed
        if (g_state.memory_monitor.is_initialized() && g_state.running.load()) {
            MemoryStats mem = g_state.memory_monitor.sample();
            
            // GPU memory from HIP
            mix.gpu_mem_used = mem.gpu_used;
            mix.gpu_mem_total = mem.gpu_total;
            
            // GPU memory from ROCm SMI (more detailed)
            mix.vram_used = mem.vram_used;
            mix.vram_total = mem.vram_total;
            mix.gtt_used = mem.gtt_used;
            mix.gtt_total = mem.gtt_total;
            
            // CPU memory
            mix.cpu_mem_resident = mem.cpu_resident;
            mix.cpu_mem_virtual = mem.cpu_vm_size;
            
            // High water marks
            mix.gpu_mem_hwm = mem.gpu_used_hwm;
            mix.vram_hwm = mem.vram_used_hwm;
            mix.gtt_hwm = mem.gtt_used_hwm;
            mix.cpu_mem_hwm = mem.cpu_resident_hwm;
            mix.cpu_vm_hwm = mem.cpu_vm_hwm;
        }
        
        auto elapsed = std::chrono::duration<double>(
            sample.timestamp - g_state.start_time).count();
        
        // Store for summary
        {
            std::lock_guard<std::mutex> lock(g_state.data_mutex);
            g_state.collected_mixes.push_back(mix);
        }
        
        // Write sample
        {
            std::lock_guard<std::mutex> lock(g_state.output_mutex);
            g_state.formatter->write_sample(*g_state.output, mix, elapsed);
            g_state.output->flush();
        }
        
        {
            std::lock_guard<std::mutex> lock(g_state.sampling_mutex);
            g_state.sampling.store(false);
        }
        g_state.sampling_cv.notify_all();
        std::this_thread::sleep_for(g_state.config.sampling_interval);
    }
    
    // Skip rocprofiler_stop_context during shutdown - it can hang or crash
    // when other profiling infrastructure is being torn down
    // rocprofiler_stop_context(g_state.context);
}

void set_profile_callback(rocprofiler_context_id_t ctx,
                          rocprofiler_compat::agent_set_profile_cb_t cb) {
    if (g_state.profile.handle != 0) {
        cb(ctx, g_state.profile);
    }
}

int tool_init(rocprofiler_client_finalize_t fini_func, void*) {
    g_state.finalize_func = fini_func;
    
    // Read environment variables
    int sampling_interval_ms = 1000;
    int gpu_id = 0;
    std::string output_format = "csv";  // Default to CSV for easier post-processing
    
    if (const char* interval = std::getenv("FP64_MONITOR_INTERVAL")) {
        // Validate: 1ms to 1 hour (3600000ms)
        sampling_interval_ms = static_cast<int>(safe_strtol(interval, 1000, 1, 3600000));
    }
    if (const char* gpu = std::getenv("FP64_MONITOR_GPU")) {
        // Validate: 0 to 255 (reasonable GPU count limit)
        gpu_id = static_cast<int>(safe_strtol(gpu, 0, 0, 255));
    }
    if (const char* verbose = std::getenv("FP64_MONITOR_VERBOSE")) {
        g_state.config.verbose = (safe_strtol(verbose, 0, 0, 1) != 0);
    }
    if (const char* format = std::getenv("FP64_MONITOR_FORMAT")) {
        output_format = format;
    }
    if (const char* usesmi = std::getenv("FP64_MONITOR_USE_ROCMSMI")) {
        g_state.config.use_rocm_smi = (safe_strtol(usesmi, 0, 0, 1) != 0);
    }
    if (const char* tmpdir = std::getenv("FP64_MONITOR_TMPDIR")) {
        g_state.config.tmpdir = tmpdir;
    }
    
    // Handle output file - default to unique filename per process
    std::string output_filename;
    if (const char* output = std::getenv("FP64_MONITOR_OUTPUT")) {
        std::string output_str(output);
        if (output_str == "stdout") {
            // Use stdout - no file
            output_filename.clear();
        } else {
            // Expand any patterns in the filename
            output_filename = expand_filename_pattern(output_str);
        }
    } else {
        // Generate default unique filename
        output_filename = generate_default_filename(output_format);
    }
    
    // Detect MPI rank early for controlling output
    int mpi_rank = get_mpi_rank();
    bool is_rank0 = (mpi_rank <= 0);  // rank 0 or non-MPI (-1)
    
    // Check if user requested ROCm SMI but it wasn't compiled in
#ifndef HAVE_ROCM_SMI
    if (g_state.config.use_rocm_smi) {
        if (is_rank0) {
            std::cerr << "[FP64 Monitor] Error: ROCm SMI support was not enabled at build time.\n"
                      << "[FP64 Monitor] Rebuild with: cmake -DENABLE_ROCM_SMI=ON ..\n"
                      << "[FP64 Monitor] Continuing without ROCm SMI memory monitoring.\n";
        }
        g_state.config.use_rocm_smi = false;
    }
#endif
    
    // Determine actual file path to open (temp file if tmpdir is set)
    std::string file_to_open = output_filename;
    if (!output_filename.empty() && !g_state.config.tmpdir.empty()) {
        // Store final destination and create temp file path
        g_state.final_filepath = output_filename;
        
        // Create unique temp filename: {tmpdir}/fp64_monitor_{pid}_{rank}.csv
        std::ostringstream temp_ss;
        temp_ss << g_state.config.tmpdir << "/fp64_monitor_" << getpid();
        if (mpi_rank >= 0) {
            temp_ss << "_" << mpi_rank;
        }
        // Preserve extension from original filename
        size_t dot_pos = output_filename.rfind('.');
        if (dot_pos != std::string::npos) {
            temp_ss << output_filename.substr(dot_pos);
        } else {
            temp_ss << ".csv";
        }
        g_state.temp_filepath = temp_ss.str();
        file_to_open = g_state.temp_filepath;
        
        if (g_state.config.verbose && is_rank0) {
            std::cerr << "[FP64 Monitor] Using temp file: " << g_state.temp_filepath << "\n";
            std::cerr << "[FP64 Monitor] Final destination: " << g_state.final_filepath << "\n";
        }
    }
    
    // Open output file if specified
    // Retry only for transient errors (ENOENT when dir being created)
    // Fail immediately for permanent errors (EACCES, ENOSPC, etc.)
    if (!file_to_open.empty()) {
        const int max_retries = 5;
        bool should_retry = true;
        
        for (int retry = 0; retry < max_retries && should_retry; ++retry) {
            errno = 0;
            g_state.output_file = std::make_unique<std::ofstream>(file_to_open);
            
            if (g_state.output_file->is_open()) {
                g_state.output = g_state.output_file.get();
                if (g_state.config.verbose && is_rank0) {
                    std::cerr << "[FP64 Monitor] Output file: " << file_to_open << "\n";
                }
                break;
            }
            
            // Check errno to decide if we should retry
            int open_errno = errno;
            switch (open_errno) {
                case ENOENT:  // Directory doesn't exist yet (being created by another rank)
                case EAGAIN:  // Resource temporarily unavailable
#if EWOULDBLOCK != EAGAIN
                case EWOULDBLOCK:
#endif
                    // Transient errors - retry after delay
                    if (retry < max_retries - 1) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    break;
                    
                case EACCES:  // Permission denied
                case EROFS:   // Read-only filesystem
                case ENOSPC:  // No space left on device
                case EDQUOT:  // Disk quota exceeded
                case ELOOP:   // Too many symbolic links
                case ENAMETOOLONG:  // Filename too long
                    // Permanent errors - don't retry
                    should_retry = false;
                    if (g_state.config.verbose && is_rank0) {
                        std::cerr << "[FP64 Monitor] File open failed (errno=" << open_errno 
                                  << "): " << strerror(open_errno) << "\n";
                    }
                    break;
                    
                default:
                    // Unknown error - retry once then give up
                    if (retry >= 1) {
                        should_retry = false;
                    } else if (retry < max_retries - 1) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                    break;
            }
        }
        
        if (!g_state.output_file || !g_state.output_file->is_open()) {
            std::cerr << "[FP64 Monitor] Warning: Could not open " << file_to_open 
                      << ", using stdout\n";
            g_state.output = &std::cout;
            g_state.output_file.reset();
            // Clear temp file paths since we couldn't open the file
            g_state.temp_filepath.clear();
            g_state.final_filepath.clear();
        }
    }
    
    g_state.config.sampling_interval = std::chrono::milliseconds(sampling_interval_ms);
    g_state.config.gpu_ids.push_back(gpu_id);
    g_state.config.output_format = output_format;
    
    // Create formatter
    g_state.formatter = OutputFormatter::create(output_format);
    
    // Get GPU agents
    auto gpus = CounterSampler::get_available_gpus();
    if (gpus.empty()) {
        std::cerr << "[FP64 Monitor] No GPUs found\n";
        return -1;
    }
    
    if (gpu_id >= static_cast<int>(gpus.size())) {
        std::cerr << "[FP64 Monitor] Invalid GPU ID " << gpu_id << "\n";
        return -1;
    }
    
    g_state.agent_id = gpus[gpu_id].id;
    
    // Log initialization info (verbose mode, rank 0 only)
    if (g_state.config.verbose && is_rank0) {
        std::cerr << "[FP64 Monitor] Initializing for GPU " << gpu_id
                  << " (" << gpus[gpu_id].name << ")";
        if (mpi_rank >= 0) {
            std::cerr << ", MPI rank " << mpi_rank;
        }
        std::cerr << "\n";
        std::cerr << "[FP64 Monitor] Output format: " << output_format << "\n";
    }
    
    // Initialize memory monitor
    if (!g_state.memory_monitor.initialize(gpu_id, g_state.config.use_rocm_smi)) {
        if (g_state.config.verbose && is_rank0) {
            std::cerr << "[FP64 Monitor] Warning: Memory monitoring initialization failed: "
                      << g_state.memory_monitor.get_last_error() << "\n";
        }
    } else if (g_state.config.verbose && is_rank0) {
        std::cerr << "[FP64 Monitor] Memory monitoring enabled\n";
    }
    
    // Populate counter map
    populate_counter_map();
    
    // Configure counters
    if (!configure_counters()) {
        return -1;
    }
    
    // Create context
    ROCP_CHECK(rocprofiler_create_context(&g_state.context), "create context");
    
    // Create buffer
    auto buffer_cb = [](rocprofiler_context_id_t,
                        rocprofiler_buffer_id_t,
                        rocprofiler_record_header_t**,
                        size_t,
                        void*,
                        uint64_t) {};
    
    ROCP_CHECK(rocprofiler_create_buffer(
                   g_state.context, 4096, 2048,
                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                   buffer_cb, nullptr, &g_state.buffer),
               "create buffer");
    
    // Create callback thread
    rocprofiler_callback_thread_t client_thread{};
    ROCP_CHECK(rocprofiler_create_callback_thread(&client_thread),
               "create callback thread");
    ROCP_CHECK(rocprofiler_assign_callback_thread(g_state.buffer, client_thread),
               "assign callback thread");
    
    // Configure device counting service
    auto device_cb = [](rocprofiler_context_id_t ctx,
                        rocprofiler_agent_id_t,
                        rocprofiler_compat::agent_set_profile_cb_t cb,
                        void*) {
        set_profile_callback(ctx, cb);
    };
    
    ROCP_CHECK(rocprofiler_configure_device_counting_service(
                   g_state.context, g_state.buffer,
                   g_state.agent_id, device_cb, nullptr),
               "configure device counting service");
    
    // Start sampling thread
    g_state.running.store(true);
    g_state.sampling_thread = std::make_unique<std::thread>(sampling_thread_func);
    
    // Register atexit handler LAST so it runs FIRST during cleanup (LIFO order).
    // This ensures we shutdown memory monitoring before other libraries
    // (HIP, HSA, rocm-smi) are torn down, preventing crashes when the
    // sampling thread tries to query already-unloaded libraries.
    std::atexit([]() {
        g_state.memory_monitor.shutdown();
        g_state.running.store(false);
    });
    
    return 0;
}

// Thread-safe guard for tool_fini (prevents double-execution)
std::once_flag g_fini_flag;

void tool_fini_impl() {
    // FIRST: Shutdown memory monitor to prevent rsmi/hip queries
    // This must happen before signaling the sampling thread to stop,
    // because the thread might be in the middle of a memory query.
    // The shutdown() call will wait for any in-progress query to complete,
    // then prevent future queries. This avoids crashes when rsmi/hip
    // libraries are torn down during atexit.
    g_state.memory_monitor.shutdown();
    
    // Check verbose and rank for output control
    int mpi_rank = get_mpi_rank();
    bool is_rank0 = (mpi_rank <= 0);
    
    if (g_state.config.verbose && is_rank0) {
        std::cerr << "[FP64 Monitor] Shutting down...\n";
    }
    
    // Signal thread to stop
    g_state.running.store(false);
    
    // Wait for sampling to complete using condition variable (not spin-wait)
    // Use a timeout to avoid hanging forever if something goes wrong
    {
        std::unique_lock<std::mutex> lock(g_state.sampling_mutex);
        bool stopped = g_state.sampling_cv.wait_for(
            lock,
            std::chrono::seconds(5),
            []{ return !g_state.sampling.load(); }
        );
        if (!stopped && g_state.config.verbose && is_rank0) {
            std::cerr << "[FP64 Monitor] Warning: Timeout waiting for sampling thread\n";
        }
    }
    
    // Check file state - if closed, state may be corrupted (e.g., rocBLAS cleanup)
    bool file_valid = g_state.output_file && g_state.output_file->is_open();
    
    // Write summary only if output file is still open
    if (file_valid) {
        std::lock_guard<std::mutex> data_lock(g_state.data_mutex);
        std::lock_guard<std::mutex> out_lock(g_state.output_mutex);
        
        if (g_state.formatter) {
            InstructionAnalyzer::Summary summary = 
                InstructionAnalyzer::compute_summary(g_state.collected_mixes);
            g_state.formatter->write_footer(*g_state.output_file, summary);
            g_state.output_file->flush();
            g_state.output_file->close();
        }
    }
    
    // Copy temp file to final destination if using tmpdir
    if (file_valid && !g_state.temp_filepath.empty() && !g_state.final_filepath.empty()) {
        // Ensure the final directory exists (create recursively if needed)
        size_t last_slash = g_state.final_filepath.rfind('/');
        if (last_slash != std::string::npos) {
            std::string final_dir = g_state.final_filepath.substr(0, last_slash);
            if (!mkdir_recursive(final_dir)) {
                std::cerr << "[FP64 Monitor] Warning: Could not create directory " 
                          << final_dir << " for final output\n";
            }
        }
        
        if (g_state.config.verbose && is_rank0) {
            std::cerr << "[FP64 Monitor] Copying " << g_state.temp_filepath 
                      << " to " << g_state.final_filepath << "\n";
        }
        
        if (copy_file(g_state.temp_filepath, g_state.final_filepath)) {
            // Delete temp file on success
            if (std::remove(g_state.temp_filepath.c_str()) != 0) {
                if (g_state.config.verbose && is_rank0) {
                    std::cerr << "[FP64 Monitor] Warning: Could not delete temp file "
                              << g_state.temp_filepath << "\n";
                }
            }
        } else {
            std::cerr << "[FP64 Monitor] Warning: Failed to copy temp file to "
                      << g_state.final_filepath << "\n";
            std::cerr << "[FP64 Monitor] Temp file remains at: " << g_state.temp_filepath << "\n";
        }
    }
    
    // Join the sampling thread properly (with timeout to avoid hanging)
    // This is safer than detach() which can leave the thread accessing freed memory
    if (g_state.sampling_thread && g_state.sampling_thread->joinable()) {
        // The thread should have exited by now since we waited for sampling to stop
        // Use a short additional wait just in case
        auto join_future = std::async(std::launch::async, []{
            if (g_state.sampling_thread && g_state.sampling_thread->joinable()) {
                g_state.sampling_thread->join();
            }
        });
        
        if (join_future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
            // Thread didn't join in time - detach as fallback to prevent crash
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] Warning: Thread join timeout, detaching\n";
            }
            // Can't detach after join attempt started, thread will be abandoned
        }
    }
    
    g_state.shutdown_complete.store(true);
    
    if (g_state.config.verbose && is_rank0) {
        std::cerr << "[FP64 Monitor] Done. Total samples: " << g_state.sample_count << "\n";
    }
}

void tool_fini(void*) {
    // Use std::call_once for thread-safe single execution
    std::call_once(g_fini_flag, tool_fini_impl);
}

}  // namespace

// External wrapper for preload.cpp to call during shutdown
// This runs after main() returns but before atexit handlers
extern "C" void fp64_monitor_fini() {
    tool_fini(nullptr);
}

extern "C" {

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id) {
    id->name = "FP64Monitor";
    g_state.client_id = id;
    
    // Check verbose setting early (before tool_init)
    bool verbose = false;
    if (const char* v = std::getenv("FP64_MONITOR_VERBOSE")) {
        verbose = (safe_strtol(v, 0, 0, 1) != 0);
    }
    
    // Only print version info if verbose and rank 0 (or non-MPI)
    int mpi_rank = get_mpi_rank();
    if (verbose && (mpi_rank <= 0)) {
        uint32_t major = version / 10000;
        uint32_t minor = (version % 10000) / 100;
        uint32_t patch = version % 100;
        
        std::cerr << "[FP64 Monitor] v" FP64_MONITOR_VERSION 
                  << " (built with ROCm " ROCM_VERSION_STRING ")\n";
        std::cerr << "[FP64 Monitor] Using rocprofiler-sdk v" << major << "."
                  << minor << "." << patch << " (" << runtime_version 
                  << "), priority=" << priority << "\n";
    }
    
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        &tool_init,
        &tool_fini,
        nullptr
    };
    
    return &cfg;
}

}  // extern "C"
