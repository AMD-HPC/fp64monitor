#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <mutex>

namespace measure_flops {

// Memory statistics in bytes
struct MemoryStats {
    // GPU memory from HIP API
    uint64_t gpu_total{0};
    uint64_t gpu_free{0};
    uint64_t gpu_used{0};
    
    // GPU memory from ROCm SMI (more detailed breakdown)
    uint64_t vram_total{0};
    uint64_t vram_used{0};
    uint64_t vis_vram_total{0};
    uint64_t vis_vram_used{0};
    uint64_t gtt_total{0};       // Graphics Translation Table (system memory for GPU)
    uint64_t gtt_used{0};
    
    // CPU memory from /proc/self/statm (in bytes)
    uint64_t cpu_vm_size{0};     // Total virtual memory
    uint64_t cpu_resident{0};    // Resident set size (physical memory)
    uint64_t cpu_shared{0};      // Shared memory
    uint64_t cpu_data{0};        // Data + stack
    
    // High water marks (peak usage during session)
    uint64_t gpu_used_hwm{0};
    uint64_t vram_used_hwm{0};
    uint64_t gtt_used_hwm{0};
    uint64_t cpu_resident_hwm{0};
    uint64_t cpu_vm_hwm{0};
    
    // Query success flags
    bool hip_valid{false};
    bool rsmi_valid{false};
    bool cpu_valid{false};
};

class MemoryMonitor {
public:
    MemoryMonitor();
    ~MemoryMonitor();
    
    // Initialize for specific GPU
    bool initialize(int gpu_id, bool use_rocm_smi);
    
    // Query current memory stats and update high water marks
    MemoryStats sample();
    
    // Get the current high water marks
    const MemoryStats& get_high_water_marks() const { return hwm_; }
    
    // Reset high water marks
    void reset_high_water_marks();
    
    // Check if initialized
    bool is_initialized() const { return initialized_; }
    
    // Get last error
    const std::string& get_last_error() const { return last_error_; }
    
    // Signal shutdown - prevents further rsmi/hip queries
    // Call this BEFORE tearing down other GPU resources to prevent crashes
    // when the sampling thread tries to query after libraries are unloaded
    void shutdown();
    
    // Check if shutdown has been called
    bool is_shutdown() const { return shutdown_.load(std::memory_order_acquire); }

private:
    bool initialized_{false};
    int gpu_id_{0};
    [[maybe_unused]] uint32_t rsmi_device_index_{0};
    [[maybe_unused]] bool rsmi_initialized_{false};
    MemoryStats hwm_;
    std::string last_error_;
    long page_size_{4096};
    bool use_rocm_smi_{false};
    
    // Shutdown synchronization - prevents queries after GPU libraries unload
    std::atomic<bool> shutdown_{false};
    mutable std::mutex query_mutex_;  // Protects rsmi/hip calls during shutdown
    
    // Query methods
    void query_hip_memory(MemoryStats& stats);
    void query_rsmi_memory(MemoryStats& stats);
    void query_cpu_memory(MemoryStats& stats);
    void update_high_water_marks(const MemoryStats& stats);
};

// Utility functions for formatting
std::string format_bytes(uint64_t bytes);
double bytes_to_gb(uint64_t bytes);
double bytes_to_mb(uint64_t bytes);

}  // namespace measure_flops
