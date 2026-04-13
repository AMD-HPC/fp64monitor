#include "measure_flops/memory_monitor.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>

// HIP includes
#include <hip/hip_runtime.h>

// ROCm SMI - optional, only compiled in if ENABLE_ROCM_SMI=ON at build time
#ifdef HAVE_ROCM_SMI
#include <rocm_smi/rocm_smi.h>
#endif

namespace measure_flops {

MemoryMonitor::MemoryMonitor() {
    page_size_ = sysconf(_SC_PAGESIZE);
    if (page_size_ <= 0) {
        page_size_ = 4096;
    }
}

MemoryMonitor::~MemoryMonitor() {
#ifdef HAVE_ROCM_SMI
    // Note: We intentionally don't call rsmi_shut_down() here.
    // When used as an ROCP_TOOL_LIBRARIES tool, the rocprofiler and HIP runtimes
    // may have already finalized by the time this destructor runs,
    // and calling rsmi_shut_down can cause a segfault.
    // The OS will clean up resources on process exit anyway.
    rsmi_initialized_ = false;
#endif
}

bool MemoryMonitor::initialize(int gpu_id, bool use_rocm_smi) {
    gpu_id_ = gpu_id;
    use_rocm_smi_ = use_rocm_smi;
    
    // Don't call hipSetDevice here - it may block if HIP isn't initialized yet
    // We'll do lazy initialization on first sample
    
#ifdef HAVE_ROCM_SMI
    // Defer rsmi_init to first query to avoid initialization order issues
    // with ROCP_TOOL_LIBRARIES tools where other libraries may not be ready yet
    rsmi_device_index_ = static_cast<uint32_t>(gpu_id);
#endif
    
    initialized_ = true;
    return true;
}

void MemoryMonitor::query_hip_memory(MemoryStats& stats) {
    size_t free_mem = 0, total_mem = 0;
    
    // Try to set device (may fail if HIP not ready)
    hipError_t err = hipSetDevice(gpu_id_);
    if (err != hipSuccess) {
        // Fallback to device 0
        err = hipSetDevice(0);
        if (err != hipSuccess) {
            stats.hip_valid = false;
            return;
        }
    }
    
    err = hipMemGetInfo(&free_mem, &total_mem);
    if (err == hipSuccess) {
        stats.gpu_total = total_mem;
        stats.gpu_free = free_mem;
        stats.gpu_used = total_mem - free_mem;
        stats.hip_valid = true;
    } else {
        stats.hip_valid = false;
    }
}

void MemoryMonitor::query_rsmi_memory(MemoryStats& stats) {
#ifdef HAVE_ROCM_SMI
    // Lazy initialization of ROCm SMI
    if (!rsmi_initialized_ && use_rocm_smi_) {
        rsmi_status_t init_status = rsmi_init(0);
        if (init_status == RSMI_STATUS_SUCCESS) {
            rsmi_initialized_ = true;
        } else {
            stats.rsmi_valid = false;
            return;
        }
    }
    
    uint64_t total = 0, used = 0;
    rsmi_status_t status;
    
    // Query VRAM (GPU dedicated memory)
    status = rsmi_dev_memory_total_get(rsmi_device_index_, RSMI_MEM_TYPE_VRAM, &total);
    if (status == RSMI_STATUS_SUCCESS) {
        stats.vram_total = total;
        status = rsmi_dev_memory_usage_get(rsmi_device_index_, RSMI_MEM_TYPE_VRAM, &used);
        if (status == RSMI_STATUS_SUCCESS) {
            stats.vram_used = used;
        }
    }
    
    // Query visible VRAM (GPU memory visible to CPU)
    status = rsmi_dev_memory_total_get(rsmi_device_index_, RSMI_MEM_TYPE_VIS_VRAM, &total);
    if (status == RSMI_STATUS_SUCCESS) {
        stats.vis_vram_total = total;
        status = rsmi_dev_memory_usage_get(rsmi_device_index_, RSMI_MEM_TYPE_VIS_VRAM, &used);
        if (status == RSMI_STATUS_SUCCESS) {
            stats.vis_vram_used = used;
        }
    }
    
    // Query GTT (Graphics Translation Table - system memory accessible by GPU)
    status = rsmi_dev_memory_total_get(rsmi_device_index_, RSMI_MEM_TYPE_GTT, &total);
    if (status == RSMI_STATUS_SUCCESS) {
        stats.gtt_total = total;
        status = rsmi_dev_memory_usage_get(rsmi_device_index_, RSMI_MEM_TYPE_GTT, &used);
        if (status == RSMI_STATUS_SUCCESS) {
            stats.gtt_used = used;
        }
    }
    
    stats.rsmi_valid = (stats.vram_total > 0);
#else
    (void)stats;  // Suppress unused parameter warning
    stats.rsmi_valid = false;
#endif
}

void MemoryMonitor::query_cpu_memory(MemoryStats& stats) {
    // Read /proc/self/statm
    // Format: size resident shared text lib data dt
    // All values are in pages
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        uint64_t size, resident, shared, text, lib, data, dt;
        if (statm >> size >> resident >> shared >> text >> lib >> data >> dt) {
            stats.cpu_vm_size = size * page_size_;
            stats.cpu_resident = resident * page_size_;
            stats.cpu_shared = shared * page_size_;
            stats.cpu_data = data * page_size_;
            stats.cpu_valid = true;
        }
        statm.close();
    }
    
    // Also try to get VmHWM from /proc/self/status for high water mark
    std::ifstream status_file("/proc/self/status");
    if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.compare(0, 6, "VmHWM:") == 0) {
                // Parse "VmHWM:    1234 kB"
                uint64_t hwm_kb = 0;
                std::istringstream iss(line.substr(6));
                if (iss >> hwm_kb) {
                    // The kernel tracks its own HWM, use it if higher than ours
                    uint64_t kernel_hwm = hwm_kb * 1024;
                    if (kernel_hwm > hwm_.cpu_resident_hwm) {
                        hwm_.cpu_resident_hwm = kernel_hwm;
                    }
                }
            } else if (line.compare(0, 7, "VmPeak:") == 0) {
                uint64_t peak_kb = 0;
                std::istringstream iss(line.substr(7));
                if (iss >> peak_kb) {
                    uint64_t kernel_peak = peak_kb * 1024;
                    if (kernel_peak > hwm_.cpu_vm_hwm) {
                        hwm_.cpu_vm_hwm = kernel_peak;
                    }
                }
            }
        }
        status_file.close();
    }
}

void MemoryMonitor::update_high_water_marks(const MemoryStats& stats) {
    if (stats.hip_valid && stats.gpu_used > hwm_.gpu_used_hwm) {
        hwm_.gpu_used_hwm = stats.gpu_used;
    }
    if (stats.rsmi_valid) {
        if (stats.vram_used > hwm_.vram_used_hwm) {
            hwm_.vram_used_hwm = stats.vram_used;
        }
        if (stats.gtt_used > hwm_.gtt_used_hwm) {
            hwm_.gtt_used_hwm = stats.gtt_used;
        }
    }
    if (stats.cpu_valid) {
        if (stats.cpu_resident > hwm_.cpu_resident_hwm) {
            hwm_.cpu_resident_hwm = stats.cpu_resident;
        }
        if (stats.cpu_vm_size > hwm_.cpu_vm_hwm) {
            hwm_.cpu_vm_hwm = stats.cpu_vm_size;
        }
    }
}

void MemoryMonitor::shutdown() {
    // Acquire mutex first to ensure no query is in progress
    std::lock_guard<std::mutex> lock(query_mutex_);
    shutdown_.store(true, std::memory_order_release);
    
    // Mark as not initialized to prevent future queries
    initialized_ = false;
    rsmi_initialized_ = false;
}

MemoryStats MemoryMonitor::sample() {
    MemoryStats stats;
    
    // Fast path: check shutdown without lock
    if (!initialized_ || shutdown_.load(std::memory_order_acquire)) {
        return stats;
    }
    
    // Acquire lock to synchronize with shutdown()
    std::lock_guard<std::mutex> lock(query_mutex_);
    
    // Re-check after acquiring lock (shutdown might have happened)
    if (!initialized_ || shutdown_.load(std::memory_order_acquire)) {
        return stats;
    }
    
    query_hip_memory(stats);
    query_rsmi_memory(stats);
    query_cpu_memory(stats);
    
    update_high_water_marks(stats);
    
    // Copy current HWM values to stats
    stats.gpu_used_hwm = hwm_.gpu_used_hwm;
    stats.vram_used_hwm = hwm_.vram_used_hwm;
    stats.gtt_used_hwm = hwm_.gtt_used_hwm;
    stats.cpu_resident_hwm = hwm_.cpu_resident_hwm;
    stats.cpu_vm_hwm = hwm_.cpu_vm_hwm;
    
    return stats;
}

void MemoryMonitor::reset_high_water_marks() {
    hwm_ = MemoryStats{};
}

std::string format_bytes(uint64_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    if (bytes >= 1024ULL * 1024 * 1024 * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024 * 1024 * 1024)) << " TB";
    } else if (bytes >= 1024ULL * 1024 * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024 * 1024)) << " GB";
    } else if (bytes >= 1024ULL * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024)) << " MB";
    } else if (bytes >= 1024) {
        oss << (static_cast<double>(bytes) / 1024.0) << " KB";
    } else {
        oss << bytes << " B";
    }
    
    return oss.str();
}

double bytes_to_gb(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024 * 1024);
}

double bytes_to_mb(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024);
}

}  // namespace measure_flops
