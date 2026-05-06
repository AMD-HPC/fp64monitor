// FP64 Monitor Tool - rocprofiler-sdk tool library
//
// This tool monitors FP64 instruction usage on AMD GPUs.
// Load via ROCP_TOOL_LIBRARIES to profile a HIP application.
//
// Usage:
//   ROCP_TOOL_LIBRARIES=/path/to/libfp64_monitor_tool_<rocm-version>.so ./your_hip_app
//
// Environment variables:
//   FP64_MONITOR_INTERVAL  - Sampling interval in ms (default: 1000)
//   FP64_MONITOR_OUTPUT    - Output file pattern (default: fp64_monitor_<rank>.csv)
//                            Supports: %r=MPI rank, %p=PID, %h=hostname
//                            Use "stdout" for console output
//   FP64_MONITOR_DIR       - Output directory (default: fp64_monitor_output)
//   FP64_MONITOR_GPUS      - Comma-separated logical GPU ids, or "all" for every visible GPU
//                            (default when unset: 0). One rocprofiler context per GPU; one output
//                            file per GPU by default. Set by fp64_monitor.sh from ROCR when applicable.
//   FP64_MONITOR_VERBOSE   - Set to 1 for verbose output
//   FP64_MONITOR_FORMAT    - Output format: text, csv, json (default: csv)
//   FP64_MONITOR_TMPDIR    - Write to fast local storage (e.g., /dev/shm), copy
//                            to final location at exit (reduces I/O on parallel FS)
//   FP64_MONITOR_USE_AMDSMI - Set to 1 to enable amd-smi memory monitoring

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
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <sstream>
#include <string>
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

void trim_gpu_token(std::string& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.erase(0, 1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
}

static bool fp64_gpu_env_is_all(std::string s) {
    trim_gpu_token(s);
    if (s.size() != 3) {
        return false;
    }
    return std::tolower(static_cast<unsigned char>(s[0])) == 'a' &&
           std::tolower(static_cast<unsigned char>(s[1])) == 'l' &&
           std::tolower(static_cast<unsigned char>(s[2])) == 'l';
}

struct Fp64MonitorGpuSelection {
    enum class Kind { kDefaultZero, kAll, kExplicit };
    Kind kind{Kind::kDefaultZero};
    std::vector<int> explicit_ids;
};

// Parses FP64_MONITOR_GPUS: unset -> default GPU 0; "all" -> every visible logical GPU;
// otherwise comma-separated ids (e.g. "0,1,2").
Fp64MonitorGpuSelection parse_fp64_monitor_gpu_env(bool is_rank0) {
    Fp64MonitorGpuSelection sel;
    const char* primary = std::getenv("FP64_MONITOR_GPUS");
    if (primary == nullptr || primary[0] == '\0') {
        sel.kind = Fp64MonitorGpuSelection::Kind::kDefaultZero;
        return sel;
    }
    std::string src(primary);
    trim_gpu_token(src);
    if (src.empty()) {
        sel.kind = Fp64MonitorGpuSelection::Kind::kDefaultZero;
        return sel;
    }
    if (fp64_gpu_env_is_all(src)) {
        sel.kind = Fp64MonitorGpuSelection::Kind::kAll;
        return sel;
    }

    std::vector<int> out;
    std::stringstream ss(src);
    std::string part;
    while (std::getline(ss, part, ',')) {
        trim_gpu_token(part);
        if (part.empty()) {
            continue;
        }
        char* endptr = nullptr;
        errno = 0;
        long v = std::strtol(part.c_str(), &endptr, 10);
        if (errno == ERANGE || endptr == part.c_str() ||
            (endptr != nullptr && *endptr != '\0')) {
            if (is_rank0) {
                std::cerr << "[FP64 Monitor] Ignoring invalid FP64_MONITOR_GPUS component: "
                          << part << "\n";
            }
            continue;
        }
        if (v < 0 || v > 255) {
            if (is_rank0) {
                std::cerr << "[FP64 Monitor] GPU id out of range (0-255): " << part << "\n";
            }
            continue;
        }
        out.push_back(static_cast<int>(v));
    }
    if (out.empty()) {
        if (is_rank0) {
            std::cerr << "[FP64 Monitor] No valid GPU ids in FP64_MONITOR_GPUS; defaulting to 0\n";
        }
        sel.kind = Fp64MonitorGpuSelection::Kind::kDefaultZero;
        return sel;
    }
    sel.kind = Fp64MonitorGpuSelection::Kind::kExplicit;
    sel.explicit_ids = std::move(out);
    return sel;
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

// Get short hostname (sanitized for safe use in filenames)
// Returns everything before the first '.' (e.g., "node01" from "node01.cluster.local")
std::string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[sizeof(hostname) - 1] = '\0';
        std::string full_hostname(hostname);
        // Extract short hostname (before first '.')
        size_t dot_pos = full_hostname.find('.');
        if (dot_pos != std::string::npos) {
            full_hostname = full_hostname.substr(0, dot_pos);
        }
        return sanitize_path_component(full_hostname);
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
//   %g = logical GPU index (when monitoring multiple GPUs; use pattern_with_gpu_if_multi)
// If pattern doesn't contain a path separator, prepend output directory
std::string expand_filename_pattern(const std::string& pattern, int gpu_index_for_g = -1) {
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

    while ((pos = result.find("%g")) != std::string::npos) {
        if (gpu_index_for_g >= 0) {
            result.replace(pos, 2, std::to_string(gpu_index_for_g));
        } else {
            result.replace(pos, 2, "0");
        }
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

// When multiple GPUs share one user pattern, ensure %g is present so each session gets a distinct path.
std::string pattern_with_gpu_if_multi(const std::string& user_pattern, size_t num_gpus) {
    if (num_gpus <= 1) {
        return user_pattern;
    }
    if (user_pattern.find("%g") != std::string::npos) {
        return user_pattern;
    }
    size_t dot = user_pattern.rfind('.');
    if (dot != std::string::npos && dot > 0) {
        return user_pattern.substr(0, dot) + "_gpu%g" + user_pattern.substr(dot);
    }
    return user_pattern + "_gpu%g";
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
// Generate default output filename with hostname, GPU ID, and rank/PID
// Format: fp64_monitor_<hostname>_gpu<gpu_id>_rank<rank>.csv
//     or: fp64_monitor_<hostname>_gpu<gpu_id>_pid<pid>.csv (if no MPI)
// Files are placed in the output directory
std::string generate_default_filename(const std::string& format, int gpu_id) {
    std::string extension;
    if (format == "json") {
        extension = ".json";
    } else if (format == "csv") {
        extension = ".csv";
    } else {
        extension = ".txt";
    }
    
    std::string dir = get_output_directory();
    std::string hostname = get_hostname();
    std::string basename;
    
    int mpi_rank = get_mpi_rank();
    if (mpi_rank >= 0) {
        // MPI mode: fp64_monitor_<hostname>_gpu<gpu_id>_rank<rank>.csv
        basename = "fp64_monitor_" + hostname + "_gpu" + std::to_string(gpu_id) 
                 + "_rank" + std::to_string(mpi_rank) + extension;
    } else {
        // Non-MPI mode: fp64_monitor_<hostname>_gpu<gpu_id>_pid<pid>.csv
        basename = "fp64_monitor_" + hostname + "_gpu" + std::to_string(gpu_id) 
                 + "_pid" + std::to_string(getpid()) + extension;
    }
    
    if (dir.empty()) {
        return basename;
    }
    return dir + "/" + basename;
}

void tool_fini(void*);

// Per-GPU rocprofiler device counter session (Phase 2: one context per monitored GPU).
struct GpuRocProfilerSession {
    int logical_gpu_id{-1};
    rocprofiler_agent_id_t agent_id{};
    rocprofiler_context_id_t context{};
    rocprofiler_buffer_id_t buffer{};
    rocprofiler_compat::config_id_t profile{.handle = 0};
    std::unordered_map<std::string, rocprofiler_counter_id_t> counter_name_to_id;
    std::unordered_map<uint64_t, std::string> counter_id_to_name;
    std::vector<std::string> configured_counters;
    size_t expected_record_count{0};
    uint64_t sample_count{0};
};

// Per-GPU output, samples, and memory (one file per GPU when multiple are monitored).
struct GpuMonitorSession {
    GpuRocProfilerSession roc;
    std::unique_ptr<std::ofstream> output_file;
    std::ostream* output{&std::cout};
    std::unique_ptr<OutputFormatter> formatter;
    std::vector<InstructionMix> collected_mixes;
    std::mutex output_mutex;
    std::mutex data_mutex;
    bool smi_error_logged{false};
    std::string temp_filepath;
    std::string final_filepath;
    MemoryMonitor memory_monitor;
};

struct ToolState {
    rocprofiler_client_id_t* client_id = nullptr;
    rocprofiler_client_finalize_t finalize_func = nullptr;

    // Serialize rocprofiler API usage (SDK not assumed thread-safe).
    std::mutex rocprofiler_api_mutex;
    // Serialize memory / AMD SMI queries across sessions (library not assumed thread-safe).
    std::mutex memory_query_mutex;

    std::vector<std::unique_ptr<GpuMonitorSession>> sessions;

    std::unique_ptr<std::thread> sampling_thread;
    std::atomic<bool> running{false};
    std::atomic<bool> sampling{false};
    std::atomic<bool> shutdown_complete{false};

    MonitorConfig config;

    std::chrono::steady_clock::time_point start_time;

    std::mutex sampling_mutex;
    std::condition_variable sampling_cv;

    ~ToolState() {
        tool_fini(nullptr);
    }
};

ToolState g_state;

void populate_counter_map(GpuRocProfilerSession& sess) {
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
        sess.agent_id, counter_cb, static_cast<void*>(&counter_ids));

    for (const auto& counter_id : counter_ids) {
        rocprofiler_counter_info_v0_t info;
        if (rocprofiler_query_counter_info(
                counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info) == 
            ROCPROFILER_STATUS_SUCCESS && info.name) {
            sess.counter_name_to_id[info.name] = counter_id;
            sess.counter_id_to_name[counter_id.handle] = info.name;
        }
    }
}

size_t get_counter_instance_count(const GpuRocProfilerSession& sess,
                                   rocprofiler_counter_id_t counter_id) {
#ifdef ROCPROFILER_SDK_V1
    (void)sess;
    rocprofiler_counter_info_v1_t info;
    info.size = sizeof(info);
    if (rocprofiler_query_counter_info(
            counter_id, ROCPROFILER_COUNTER_INFO_VERSION_1, &info) == 
        ROCPROFILER_STATUS_SUCCESS) {
        return info.dimensions_instances_count;
    }
#else
    size_t instance_count = 1;
    if (rocprofiler_query_counter_instance_count(
            sess.agent_id, counter_id, &instance_count) ==
        ROCPROFILER_STATUS_SUCCESS) {
        return instance_count;
    }
#endif
    return 1;
}

void resolve_counter_set_from_env(std::vector<std::string>* out_counters,
                                  std::string* out_counter_set_name) {
    out_counters->clear();
    *out_counter_set_name = "fp64";
    const char* counter_set = std::getenv("FP64_MONITOR_COUNTERS");
    if (counter_set) {
        std::string cs(counter_set);
        if (cs == "fp32") {
            *out_counters = CounterSets::fp32_minimal();
            *out_counter_set_name = "fp32";
        } else if (cs == "fp16") {
            *out_counters = CounterSets::fp16_minimal();
            *out_counter_set_name = "fp16";
        } else if (cs == "int") {
            *out_counters = CounterSets::int_minimal();
            *out_counter_set_name = "int";
        } else if (cs == "salu") {
            *out_counters = CounterSets::salu_minimal();
            *out_counter_set_name = "salu";
        } else if (cs == "highlevel") {
            *out_counters = CounterSets::highlevel();
            *out_counter_set_name = "highlevel";
        } else if (cs == "highlevel2") {
            *out_counters = CounterSets::highlevel2();
            *out_counter_set_name = "highlevel2";
        } else {
            *out_counters = CounterSets::fp64_minimal();
        }
    } else {
        *out_counters = CounterSets::fp64_minimal();
    }
}

bool configure_counters_for_session(GpuRocProfilerSession& sess,
                                    const std::vector<std::string>& counters,
                                    bool log_details) {
    sess.configured_counters.clear();
    std::vector<rocprofiler_counter_id_t> counter_ids;
    size_t total_instances = 0;
    
    for (const auto& name : counters) {
        auto it = sess.counter_name_to_id.find(name);
        if (it != sess.counter_name_to_id.end()) {
            counter_ids.push_back(it->second);
            total_instances += get_counter_instance_count(sess, it->second);
            sess.configured_counters.push_back(name);
        }
    }
    
    if (counter_ids.empty()) {
        std::cerr << "[FP64 Monitor] No counters available for GPU " << sess.logical_gpu_id << "\n";
        return false;
    }
    
    if (log_details && g_state.config.verbose && get_mpi_rank() <= 0) {
        std::cerr << "[FP64 Monitor] GPU " << sess.logical_gpu_id << ": configuring "
                  << counter_ids.size() << " counters (" << total_instances << " instances)\n";
    }
    
    rocprofiler_status_t status = rocprofiler_compat::create_config(
        sess.agent_id,
        counter_ids.data(),
        counter_ids.size(),
        &sess.profile);
    
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] Failed to create counter config for GPU "
                  << sess.logical_gpu_id << ": "
                  << rocprofiler_get_status_string(status) << std::endl;
        return false;
    }
    
    sess.expected_record_count = total_instances;
    return true;
}

std::string decode_counter_name(const GpuRocProfilerSession& sess,
                                const rocprofiler_compat::counter_record_t& rec) {
    rocprofiler_counter_id_t counter_id{};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);
    
    auto it = sess.counter_id_to_name.find(counter_id.handle);
    if (it != sess.counter_id_to_name.end()) {
        return it->second;
    }
    return "UNKNOWN";
}

Sample take_sample(GpuRocProfilerSession& sess) {
    Sample sample;
    sample.sample_id = ++sess.sample_count;
    sample.timestamp = std::chrono::steady_clock::now();
    sample.gpu_id = sess.logical_gpu_id;
    
    std::vector<rocprofiler_compat::counter_record_t> records(sess.expected_record_count);
    size_t record_count = records.size();
    
    rocprofiler_user_data_t user_data;
    user_data.value = sample.sample_id;
    
    rocprofiler_status_t status;
    {
        std::lock_guard<std::mutex> lock(g_state.rocprofiler_api_mutex);
        status = rocprofiler_sample_device_counting_service(
            sess.context,
            user_data,
            ROCPROFILER_COUNTER_FLAG_NONE,
            records.data(),
            &record_count);
    }
    
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        if (g_state.config.verbose && get_mpi_rank() <= 0) {
            std::cerr << "[FP64 Monitor] GPU " << sess.logical_gpu_id << " sample "
                      << sample.sample_id << " failed: "
                      << rocprofiler_get_status_string(status) << "\n";
        }
        return sample;
    }
    
    records.resize(record_count);
    
    for (const auto& rec : records) {
        CounterValue cv;
        cv.name = decode_counter_name(sess, rec);
        cv.value = rec.counter_value;
        cv.instance_id = rec.id;
        sample.counters.push_back(cv);
    }
    
    return sample;
}

void set_profile_callback_for_session(rocprofiler_context_id_t ctx,
                                      rocprofiler_compat::agent_set_profile_cb_t cb,
                                      void* user_data) {
    auto* sess = static_cast<GpuRocProfilerSession*>(user_data);
    if (sess != nullptr && sess->profile.handle != 0) {
        cb(ctx, sess->profile);
    }
}

bool start_all_contexts_with_retry() {
    int retries = 0;
    const int max_retries = 50;
    int mpi_rank = get_mpi_rank();
    bool is_rank0 = (mpi_rank <= 0);

    while (retries < max_retries && g_state.running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        rocprofiler_status_t last_status = ROCPROFILER_STATUS_SUCCESS;
        bool batch_ok = true;
        {
            std::lock_guard<std::mutex> lock(g_state.rocprofiler_api_mutex);
            for (auto& up : g_state.sessions) {
                GpuMonitorSession& ms = *up;
                last_status = rocprofiler_start_context(ms.roc.context);
                if (last_status != ROCPROFILER_STATUS_SUCCESS) {
                    batch_ok = false;
                    break;
                }
            }
        }
        if (batch_ok) {
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] All contexts started after " << retries << " retries\n";
            }
            return true;
        }
        if (last_status != ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED) {
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] Failed to start context: "
                          << rocprofiler_get_status_string(last_status) << std::endl;
            }
            return false;
        }
        retries++;
    }
    std::cerr << "[FP64 Monitor] Timeout waiting for HSA to load (running="
              << g_state.running.load() << ")\n";
    return false;
}

void attach_memory_to_mix(InstructionMix& mix, GpuMonitorSession& ms, bool is_rank0) {
    if (!ms.memory_monitor.is_initialized() || ms.memory_monitor.is_shutdown()) {
        return;
    }
    MemoryStats mem;
    {
        std::lock_guard<std::mutex> lock(g_state.memory_query_mutex);
        mem = ms.memory_monitor.sample();
    }
    if (g_state.config.use_amd_smi && !mem.smi_valid && !ms.smi_error_logged) {
        ms.smi_error_logged = true;
        if (is_rank0) {
            std::cerr << "[FP64 Monitor] Warning: AMD SMI memory query failed (GPU "
                      << ms.roc.logical_gpu_id << "): "
                      << ms.memory_monitor.get_last_error() << "\n";
        }
    }
    mix.vram_used = mem.vram_used;
    mix.vram_total = mem.vram_total;
    mix.gtt_used = mem.gtt_used;
    mix.gtt_total = mem.gtt_total;
    mix.vram_hwm = mem.vram_used_hwm;
    mix.gtt_hwm = mem.gtt_used_hwm;

    const auto& run_gpus = g_state.config.gpu_ids;
    const bool primary_for_cpu =
        run_gpus.empty() || ms.roc.logical_gpu_id == run_gpus.front();
    if (primary_for_cpu) {
        mix.cpu_mem_resident = mem.cpu_resident;
        mix.cpu_mem_virtual = mem.cpu_vm_size;
        mix.cpu_mem_hwm = mem.cpu_resident_hwm;
        mix.cpu_vm_hwm = mem.cpu_vm_hwm;
    } else {
        mix.cpu_mem_resident = 0;
        mix.cpu_mem_virtual = 0;
        mix.cpu_mem_hwm = 0;
        mix.cpu_vm_hwm = 0;
    }
}

void sampling_thread_func() {
    int mpi_rank = get_mpi_rank();
    bool is_rank0 = (mpi_rank <= 0);

    if (!start_all_contexts_with_retry()) {
        return;
    }

    for (auto& up : g_state.sessions) {
        GpuMonitorSession& ms = *up;
        MonitorConfig header_cfg = g_state.config;
        header_cfg.gpu_ids = {ms.roc.logical_gpu_id};
        if (g_state.config.gpu_ids.size() > 1) {
            header_cfg.metadata_monitored_gpu_ids = g_state.config.gpu_ids;
        } else {
            header_cfg.metadata_monitored_gpu_ids.clear();
        }
        std::lock_guard<std::mutex> lock(ms.output_mutex);
        ms.formatter->write_header(*ms.output, header_cfg);
    }

    g_state.start_time = std::chrono::steady_clock::now();

    while (true) {
        g_state.sampling.store(true);

        if (!g_state.running.load()) {
            for (auto& up : g_state.sessions) {
                GpuMonitorSession& ms = *up;
                Sample final_sample = take_sample(ms.roc);
                if (g_state.config.verbose && get_mpi_rank() <= 0) {
                    std::cerr << "[FP64 Monitor] GPU " << ms.roc.logical_gpu_id << " final sample: "
                              << final_sample.counters.size() << " counter records\n";
                }
                if (!final_sample.counters.empty()) {
                    InstructionMix mix = InstructionAnalyzer::analyze(final_sample);
                    if (ms.memory_monitor.is_initialized() && !ms.memory_monitor.is_shutdown()) {
                        attach_memory_to_mix(mix, ms, is_rank0);
                    }
                    auto elapsed = std::chrono::duration<double>(
                        final_sample.timestamp - g_state.start_time).count();
                    {
                        std::lock_guard<std::mutex> dlock(ms.data_mutex);
                        ms.collected_mixes.push_back(mix);
                    }
                    {
                        std::lock_guard<std::mutex> olock(ms.output_mutex);
                        ms.formatter->write_sample(*ms.output, mix, elapsed,
                                                    g_state.config.counter_set_name);
                        ms.output->flush();
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(g_state.sampling_mutex);
                g_state.sampling.store(false);
            }
            g_state.sampling_cv.notify_all();
            break;
        }

        for (auto& up : g_state.sessions) {
            GpuMonitorSession& ms = *up;
            Sample sample = take_sample(ms.roc);
            InstructionMix mix;
            if (!sample.counters.empty()) {
                mix = InstructionAnalyzer::analyze(sample);
            }
            mix.sample_id = sample.sample_id;
            mix.gpu_id = sample.gpu_id;
            attach_memory_to_mix(mix, ms, is_rank0);

            auto elapsed = std::chrono::duration<double>(
                sample.timestamp - g_state.start_time).count();
            {
                std::lock_guard<std::mutex> dlock(ms.data_mutex);
                ms.collected_mixes.push_back(mix);
            }
            {
                std::lock_guard<std::mutex> olock(ms.output_mutex);
                ms.formatter->write_sample(*ms.output, mix, elapsed,
                                            g_state.config.counter_set_name);
                ms.output->flush();
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_state.sampling_mutex);
            g_state.sampling.store(false);
        }
        g_state.sampling_cv.notify_all();
        std::this_thread::sleep_for(g_state.config.sampling_interval);
    }
}

bool open_output_for_session(GpuMonitorSession& ms,
                             const std::string& output_filename_final,
                             const std::string& tmpdir,
                             int mpi_rank,
                             int logical_gpu_id,
                             bool is_rank0) {
    if (output_filename_final.empty()) {
        ms.output = &std::cout;
        ms.output_file.reset();
        return true;
    }

    ms.final_filepath = output_filename_final;
    std::string file_to_open = output_filename_final;
    if (!tmpdir.empty()) {
        std::ostringstream temp_ss;
        temp_ss << tmpdir << "/fp64_monitor_" << getpid();
        if (mpi_rank >= 0) {
            temp_ss << "_" << mpi_rank;
        }
        temp_ss << "_gpu" << logical_gpu_id;
        size_t dot_pos = output_filename_final.rfind('.');
        if (dot_pos != std::string::npos) {
            temp_ss << output_filename_final.substr(dot_pos);
        } else {
            temp_ss << ".csv";
        }
        ms.temp_filepath = temp_ss.str();
        file_to_open = ms.temp_filepath;
        if (g_state.config.verbose && is_rank0) {
            std::cerr << "[FP64 Monitor] GPU " << logical_gpu_id << " temp file: " << ms.temp_filepath
                      << "\n[FP64 Monitor] GPU " << logical_gpu_id << " final: " << ms.final_filepath
                      << "\n";
        }
    }

    const int max_retries = 5;
    bool should_retry = true;
    for (int retry = 0; retry < max_retries && should_retry; ++retry) {
        errno = 0;
        ms.output_file = std::make_unique<std::ofstream>(file_to_open);
        if (ms.output_file->is_open()) {
            ms.output = ms.output_file.get();
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] GPU " << logical_gpu_id << " output: " << file_to_open
                          << "\n";
            }
            return true;
        }
        int open_errno = errno;
        switch (open_errno) {
            case ENOENT:
            case EAGAIN:
#if EWOULDBLOCK != EAGAIN
            case EWOULDBLOCK:
#endif
                if (retry < max_retries - 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                break;
            case EACCES:
            case EROFS:
            case ENOSPC:
            case EDQUOT:
            case ELOOP:
            case ENAMETOOLONG:
                should_retry = false;
                if (g_state.config.verbose && is_rank0) {
                    std::cerr << "[FP64 Monitor] File open failed (errno=" << open_errno
                              << "): " << strerror(open_errno) << "\n";
                }
                break;
            default:
                if (retry >= 1) {
                    should_retry = false;
                } else if (retry < max_retries - 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                break;
        }
    }
    std::cerr << "[FP64 Monitor] Warning: Could not open " << file_to_open
              << " for GPU " << logical_gpu_id << ", using stdout\n";
    ms.output = &std::cout;
    ms.output_file.reset();
    ms.temp_filepath.clear();
    ms.final_filepath.clear();
    return true;
}

bool create_rocprofiler_pipeline(GpuMonitorSession& ms) {
    std::lock_guard<std::mutex> lock(g_state.rocprofiler_api_mutex);
    rocprofiler_status_t st = rocprofiler_create_context(&ms.roc.context);
    if (st != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] create context (GPU " << ms.roc.logical_gpu_id << "): "
                  << rocprofiler_get_status_string(st) << std::endl;
        return false;
    }

    auto buffer_cb = [](rocprofiler_context_id_t,
                        rocprofiler_buffer_id_t,
                        rocprofiler_record_header_t**,
                        size_t,
                        void*,
                        uint64_t) {};

    st = rocprofiler_create_buffer(ms.roc.context, 4096, 2048,
                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                   buffer_cb, nullptr, &ms.roc.buffer);
    if (st != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] create buffer (GPU " << ms.roc.logical_gpu_id << "): "
                  << rocprofiler_get_status_string(st) << std::endl;
        return false;
    }

    rocprofiler_callback_thread_t client_thread{};
    st = rocprofiler_create_callback_thread(&client_thread);
    if (st != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] create callback thread (GPU " << ms.roc.logical_gpu_id
                  << "): " << rocprofiler_get_status_string(st) << std::endl;
        return false;
    }
    st = rocprofiler_assign_callback_thread(ms.roc.buffer, client_thread);
    if (st != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] assign callback thread (GPU " << ms.roc.logical_gpu_id
                  << "): " << rocprofiler_get_status_string(st) << std::endl;
        return false;
    }

    auto device_cb = [](rocprofiler_context_id_t ctx,
                        rocprofiler_agent_id_t,
                        rocprofiler_compat::agent_set_profile_cb_t cb,
                        void* user_data) {
        set_profile_callback_for_session(ctx, cb, user_data);
    };

    st = rocprofiler_configure_device_counting_service(
        ms.roc.context, ms.roc.buffer, ms.roc.agent_id, device_cb, &ms.roc);
    if (st != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "[FP64 Monitor] configure device counting (GPU " << ms.roc.logical_gpu_id
                  << "): " << rocprofiler_get_status_string(st) << std::endl;
        return false;
    }
    return true;
}

void destroy_session_profiles() {
    for (auto& up : g_state.sessions) {
        if (up->roc.profile.handle != 0) {
            rocprofiler_compat::destroy_config(up->roc.profile);
            up->roc.profile = rocprofiler_compat::config_id_t{.handle = 0};
        }
    }
    g_state.sessions.clear();
}

int tool_init(rocprofiler_client_finalize_t fini_func, void*) {
    g_state.finalize_func = fini_func;
    
    // Read environment variables
    int sampling_interval_ms = 1000;
    std::string output_format = "csv";  // Default to CSV for easier post-processing
    
    if (const char* interval = std::getenv("FP64_MONITOR_INTERVAL")) {
        // Validate: 1ms to 1 hour (3600000ms)
        sampling_interval_ms = static_cast<int>(safe_strtol(interval, 1000, 1, 3600000));
    }
    if (const char* verbose = std::getenv("FP64_MONITOR_VERBOSE")) {
        g_state.config.verbose = (safe_strtol(verbose, 0, 0, 1) != 0);
    }
    if (const char* format = std::getenv("FP64_MONITOR_FORMAT")) {
        output_format = format;
    }
    if (const char* usesmi = std::getenv("FP64_MONITOR_USE_AMDSMI")) {
        g_state.config.use_amd_smi = (safe_strtol(usesmi, 0, 0, 1) != 0);
    }
    if (const char* tmpdir = std::getenv("FP64_MONITOR_TMPDIR")) {
        g_state.config.tmpdir = tmpdir;
    }

    int mpi_rank = get_mpi_rank();
    bool is_rank0 = (mpi_rank <= 0);

    Fp64MonitorGpuSelection gpu_sel = parse_fp64_monitor_gpu_env(is_rank0);

    auto gpus = CounterSampler::get_available_gpus();
    if (gpus.empty()) {
        std::cerr << "[FP64 Monitor] No GPUs found\n";
        return -1;
    }

    std::vector<int> gpu_ids;
    if (gpu_sel.kind == Fp64MonitorGpuSelection::Kind::kAll) {
        gpu_ids.reserve(gpus.size());
        for (size_t i = 0; i < gpus.size(); ++i) {
            gpu_ids.push_back(static_cast<int>(i));
        }
    } else if (gpu_sel.kind == Fp64MonitorGpuSelection::Kind::kExplicit) {
        gpu_ids = gpu_sel.explicit_ids;
    } else {
        gpu_ids.push_back(0);
    }

    for (int gid : gpu_ids) {
        if (gid < 0 || gid >= static_cast<int>(gpus.size())) {
            std::cerr << "[FP64 Monitor] Invalid GPU ID " << gid << " ("
                      << gpus.size() << " GPU(s) visible)\n";
            return -1;
        }
    }

#ifndef HAVE_AMD_SMI
    if (g_state.config.use_amd_smi) {
        if (is_rank0) {
            std::cerr << "[FP64 Monitor] Error: AMD SMI support was not enabled at build time.\n"
                      << "[FP64 Monitor] Rebuild with: cmake -DENABLE_AMD_SMI=ON ..\n"
                      << "[FP64 Monitor] Continuing without AMD SMI memory monitoring.\n";
        }
        g_state.config.use_amd_smi = false;
    }
#endif

    const char* output_env = std::getenv("FP64_MONITOR_OUTPUT");
    bool use_stdout = false;
    std::string user_output_pattern;
    if (output_env != nullptr) {
        std::string os(output_env);
        if (os == "stdout") {
            use_stdout = true;
        } else {
            user_output_pattern = pattern_with_gpu_if_multi(os, gpu_ids.size());
        }
    }
    if (gpu_ids.size() > 1 && use_stdout) {
        std::cerr << "[FP64 Monitor] stdout output is not supported when monitoring multiple GPUs; "
                     "set FP64_MONITOR_OUTPUT to a file pattern (use %%g per GPU).\n";
        return -1;
    }

    std::vector<std::string> counter_names;
    std::string counter_set_name;
    resolve_counter_set_from_env(&counter_names, &counter_set_name);

    g_state.config.sampling_interval = std::chrono::milliseconds(sampling_interval_ms);
    g_state.config.gpu_ids = gpu_ids;
    g_state.config.output_format = output_format;
    g_state.config.counter_set_name = counter_set_name;

    if (g_state.config.verbose && is_rank0) {
        std::cerr << "[FP64 Monitor] Counter set: " << counter_set_name << "\n";
        std::cerr << "[FP64 Monitor] Monitoring " << gpu_ids.size() << " GPU(s), format "
                  << output_format << "\n";
    }

    std::vector<std::string> session_final_paths;
    session_final_paths.reserve(gpu_ids.size());
    std::unordered_set<std::string> used_final_paths;
    for (int gid : gpu_ids) {
        std::string final_path;
        if (use_stdout) {
            final_path.clear();
        } else if (output_env != nullptr) {
            final_path = expand_filename_pattern(user_output_pattern, gid);
        } else {
            final_path = generate_default_filename(output_format, gid);
        }
        if (!final_path.empty()) {
            if (!used_final_paths.insert(final_path).second) {
                std::cerr << "[FP64 Monitor] Duplicate output path after pattern expansion: "
                          << final_path << "\n"
                          << "[FP64 Monitor] Use %g in FP64_MONITOR_OUTPUT so each GPU gets a "
                             "distinct file.\n";
                return -1;
            }
        }
        session_final_paths.push_back(std::move(final_path));
    }

    g_state.sessions.clear();
    for (size_t idx = 0; idx < gpu_ids.size(); ++idx) {
        const int gid = gpu_ids[idx];
        auto ms = std::make_unique<GpuMonitorSession>();
        ms->roc.logical_gpu_id = gid;
        ms->roc.agent_id = gpus[static_cast<size_t>(gid)].id;

        const std::string& final_path = session_final_paths[idx];

        open_output_for_session(*ms, final_path, g_state.config.tmpdir, mpi_rank, gid, is_rank0);

        ms->formatter = OutputFormatter::create(output_format);

        if (g_state.config.verbose) {
            std::cerr << "[FP64 Monitor] Initializing GPU " << gid << " (" << gpus[gid].name << ")"
                      << ", MPI rank " << mpi_rank << "\n";
        }

        if (!ms->memory_monitor.initialize(gid, g_state.config.use_amd_smi)) {
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] Warning: Memory init failed for GPU " << gid << ": "
                          << ms->memory_monitor.get_last_error() << "\n";
            }
        } else if (g_state.config.verbose && is_rank0) {
            std::cerr << "[FP64 Monitor] Memory monitoring enabled for GPU " << gid << "\n";
        }

        populate_counter_map(ms->roc);
        if (!configure_counters_for_session(ms->roc, counter_names, idx == 0)) {
            if (ms->roc.profile.handle != 0) {
                rocprofiler_compat::destroy_config(ms->roc.profile);
                ms->roc.profile = rocprofiler_compat::config_id_t{.handle = 0};
            }
            destroy_session_profiles();
            return -1;
        }
        if (!create_rocprofiler_pipeline(*ms)) {
            if (ms->roc.profile.handle != 0) {
                rocprofiler_compat::destroy_config(ms->roc.profile);
                ms->roc.profile = rocprofiler_compat::config_id_t{.handle = 0};
            }
            destroy_session_profiles();
            return -1;
        }

        g_state.sessions.push_back(std::move(ms));
    }

    g_state.running.store(true);
    g_state.sampling_thread = std::make_unique<std::thread>(sampling_thread_func);

    std::atexit([]() {
        for (auto& up : g_state.sessions) {
            up->memory_monitor.shutdown();
        }
        g_state.running.store(false);
    });

    return 0;
}

// Thread-safe guard for tool_fini (prevents double-execution)
std::once_flag g_fini_flag;

void tool_fini_impl() {
    for (auto& up : g_state.sessions) {
        up->memory_monitor.shutdown();
    }

    int mpi_rank = get_mpi_rank();
    bool is_rank0 = (mpi_rank <= 0);

    if (g_state.config.verbose && is_rank0) {
        std::cerr << "[FP64 Monitor] Shutting down...\n";
    }

    g_state.running.store(false);

    {
        std::unique_lock<std::mutex> lock(g_state.sampling_mutex);
        bool stopped = g_state.sampling_cv.wait_for(
            lock,
            std::chrono::seconds(5),
            [] { return !g_state.sampling.load(); });
        if (!stopped && g_state.config.verbose && is_rank0) {
            std::cerr << "[FP64 Monitor] Warning: Timeout waiting for sampling thread\n";
        }
    }

    for (auto& up : g_state.sessions) {
        GpuMonitorSession& ms = *up;
        bool file_valid = ms.output_file && ms.output_file->is_open();

        if (file_valid && ms.formatter) {
            std::lock_guard<std::mutex> data_lock(ms.data_mutex);
            std::lock_guard<std::mutex> out_lock(ms.output_mutex);
            InstructionAnalyzer::Summary summary =
                InstructionAnalyzer::compute_summary(ms.collected_mixes);
            ms.formatter->write_footer(*ms.output_file, summary);
            ms.output_file->flush();
            ms.output_file->close();
        }

        if (file_valid && !ms.temp_filepath.empty() && !ms.final_filepath.empty()) {
            size_t last_slash = ms.final_filepath.rfind('/');
            if (last_slash != std::string::npos) {
                std::string final_dir = ms.final_filepath.substr(0, last_slash);
                if (!mkdir_recursive(final_dir)) {
                    std::cerr << "[FP64 Monitor] Warning: Could not create directory "
                              << final_dir << " for final output\n";
                }
            }
            if (g_state.config.verbose && is_rank0) {
                std::cerr << "[FP64 Monitor] Copying " << ms.temp_filepath << " to "
                          << ms.final_filepath << "\n";
            }
            if (copy_file(ms.temp_filepath, ms.final_filepath)) {
                if (std::remove(ms.temp_filepath.c_str()) != 0) {
                    if (g_state.config.verbose && is_rank0) {
                        std::cerr << "[FP64 Monitor] Warning: Could not delete temp file "
                                  << ms.temp_filepath << "\n";
                    }
                }
            } else {
                std::cerr << "[FP64 Monitor] Warning: Failed to copy temp file to "
                          << ms.final_filepath << "\n";
                std::cerr << "[FP64 Monitor] Temp file remains at: " << ms.temp_filepath << "\n";
            }
        }
    }

    if (g_state.sampling_thread && g_state.sampling_thread->joinable()) {
        g_state.sampling_thread->join();
    }

    g_state.shutdown_complete.store(true);

    if (g_state.config.verbose && is_rank0) {
        for (const auto& up : g_state.sessions) {
            std::cerr << "[FP64 Monitor] GPU " << up->roc.logical_gpu_id
                      << " samples: " << up->roc.sample_count << "\n";
        }
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
