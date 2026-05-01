#include "measure_flops/memory_monitor.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <vector>
#include <iostream>

namespace measure_flops {

MemoryMonitor::MemoryMonitor() {
    page_size_ = sysconf(_SC_PAGESIZE);
    if (page_size_ <= 0) {
        page_size_ = 4096;
    }
}

MemoryMonitor::~MemoryMonitor() {
#ifdef HAVE_AMD_SMI
    // Note: We intentionally don't call amdsmi_shut_down() here.
    // When used as an ROCP_TOOL_LIBRARIES tool, the rocprofiler and HIP runtimes
    // may have already finalized by the time this destructor runs,
    // and calling amdsmi_shut_down can cause a segfault.
    // The OS will clean up resources on process exit anyway.
    smi_initialized_ = false;
#endif
}

bool MemoryMonitor::initialize(int gpu_id, bool use_amd_smi) {
    gpu_id_ = gpu_id;
    use_amd_smi_ = use_amd_smi;
    
    // AMD SMI initialization is deferred to first query to avoid initialization
    // order issues with ROCP_TOOL_LIBRARIES tools
    
    initialized_ = true;
    return true;
}

void MemoryMonitor::query_smi_memory(MemoryStats& stats) {
#ifdef HAVE_AMD_SMI
    // Lazy initialization of AMD SMI
    if (!smi_initialized_ && use_amd_smi_) {
        amdsmi_status_t init_status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
        if (init_status != AMDSMI_STATUS_SUCCESS) {
            last_error_ = "amdsmi_init failed with status " + std::to_string(init_status);
            stats.smi_valid = false;
            return;
        }
        
        // Get socket handles
        uint32_t socket_count = 0;
        init_status = amdsmi_get_socket_handles(&socket_count, nullptr);
        if (init_status != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
            last_error_ = "amdsmi_get_socket_handles failed: status=" + std::to_string(init_status) 
                         + ", count=" + std::to_string(socket_count);
            stats.smi_valid = false;
            return;
        }
        
        std::vector<amdsmi_socket_handle> sockets(socket_count);
        init_status = amdsmi_get_socket_handles(&socket_count, sockets.data());
        if (init_status != AMDSMI_STATUS_SUCCESS) {
            last_error_ = "amdsmi_get_socket_handles (data) failed: " + std::to_string(init_status);
            stats.smi_valid = false;
            return;
        }
        
        // Get processor (GPU) handles from first socket
        uint32_t proc_count = 0;
        init_status = amdsmi_get_processor_handles(sockets[0], &proc_count, nullptr);
        if (init_status != AMDSMI_STATUS_SUCCESS || proc_count == 0) {
            last_error_ = "amdsmi_get_processor_handles failed: status=" + std::to_string(init_status)
                         + ", count=" + std::to_string(proc_count);
            stats.smi_valid = false;
            return;
        }
        
        std::vector<amdsmi_processor_handle> processors(proc_count);
        init_status = amdsmi_get_processor_handles(sockets[0], &proc_count, processors.data());
        if (init_status != AMDSMI_STATUS_SUCCESS) {
            last_error_ = "amdsmi_get_processor_handles (data) failed: " + std::to_string(init_status);
            stats.smi_valid = false;
            return;
        }
        
        // Store the processor handle for the requested GPU
        if (static_cast<uint32_t>(gpu_id_) < proc_count) {
            processor_handle_ = processors[gpu_id_];
            smi_initialized_ = true;
            last_error_.clear();
        } else {
            processor_handle_ = processors[0];
            smi_initialized_ = true;
            last_error_.clear();
        /*
            last_error_ = "GPU " + std::to_string(gpu_id_) + " not found, only " 
                         + std::to_string(proc_count) + " GPUs available";
            stats.smi_valid = false;
            return;
            */
        }
    }
    
    if (!smi_initialized_) {
        if (last_error_.empty()) {
            last_error_ = "AMD SMI not initialized (use_amd_smi=" + std::to_string(use_amd_smi_) + ")";
        }
        stats.smi_valid = false;
        return;
    }
    
    uint64_t total = 0, used = 0;
    amdsmi_status_t status;
    
    // Query VRAM (GPU dedicated memory)
    status = amdsmi_get_gpu_memory_total(processor_handle_, AMDSMI_MEM_TYPE_VRAM, &total);
    if (status == AMDSMI_STATUS_SUCCESS) {
        stats.vram_total = total;
        status = amdsmi_get_gpu_memory_usage(processor_handle_, AMDSMI_MEM_TYPE_VRAM, &used);
        if (status == AMDSMI_STATUS_SUCCESS) {
            stats.vram_used = used;
        } else {
            last_error_ = "AMD SMI failed to read VRAM usage";
            //std::cerr << gpu_id_ << ": Failed to read VRAM: " << status << std::endl;
        }
    } else {
        //std::cerr << gpu_id_ << ": Failed to read VRAM: " << status << std::endl;
        last_error_ = "AMD SMI failed to read VRAM total";
    }
    //std::cout << gpu_id_ << ": VRAM: " << stats.vram_used << " of " << stats.vram_total << std::endl;
    
    // Query visible VRAM (GPU memory visible to CPU)
    status = amdsmi_get_gpu_memory_total(processor_handle_, AMDSMI_MEM_TYPE_VIS_VRAM, &total);
    if (status == AMDSMI_STATUS_SUCCESS) {
        stats.vis_vram_total = total;
        status = amdsmi_get_gpu_memory_usage(processor_handle_, AMDSMI_MEM_TYPE_VIS_VRAM, &used);
        if (status == AMDSMI_STATUS_SUCCESS) {
            stats.vis_vram_used = used;
        } else {
            //std::cerr << gpu_id_ << ": Failed to read VIS VRAM: " << status << std::endl;
        }
    } else {
        //std::cerr << gpu_id_ << ": Failed to read VIS VRAM: " << status << std::endl;
    }
    //std::cout << gpu_id_ << ": VIS VRAM: " << stats.vis_vram_used << " of " << stats.vis_vram_total << std::endl;
    
    /* Disabled because unreliable on CPX/TPX modes... 
    // Query GTT (Graphics Translation Table - system memory accessible by GPU)
    status = amdsmi_get_gpu_memory_total(processor_handle_, AMDSMI_MEM_TYPE_GTT, &total);
    if (status == AMDSMI_STATUS_SUCCESS) {
        stats.gtt_total = total;
        status = amdsmi_get_gpu_memory_usage(processor_handle_, AMDSMI_MEM_TYPE_GTT, &used);
        if (status == AMDSMI_STATUS_SUCCESS) {
            stats.gtt_used = used;
        } else {
            //std::cerr << gpu_id_ << ": Failed to read GTT: " << status << std::endl;
            last_error_ = "AMD SMI failed to read GTT usage";
        }
    } else {
        //std::cerr << gpu_id_ << ": Failed to read GTT: " << status << std::endl;
        last_error_ = "AMD SMI failed to read GTT total";
    }
    std::cout << gpu_id_ << ": GTT: " << stats.gtt_used << " of " << stats.gtt_total << std::endl;
    */
    
    stats.smi_valid = (stats.vram_total > 0);
#else
    (void)stats;  // Suppress unused parameter warning
    stats.smi_valid = false;
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
    if (stats.smi_valid) {
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
    smi_initialized_ = false;
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
    
    query_smi_memory(stats);
    if (use_amd_smi_ && last_error_.size() > 0) {
        static bool once = true;
        if (once) {
            std::cerr << gpu_id_ << ":" << last_error_ << std::endl;
            once = false;
        }
    }
    query_cpu_memory(stats);
    
    update_high_water_marks(stats);
    
    // Copy current HWM values to stats
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
