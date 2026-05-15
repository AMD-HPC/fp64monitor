#include "measure_flops/counter_sampler.hpp"
#include "measure_flops/types.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

const char* VERSION = "1.0.0";

void print_version() {
    std::cout << "FP64 GPU Monitor v" << VERSION << "\n";
    std::cout << "A tool for monitoring FP64 instruction usage on AMD GPUs\n";
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\n"
              << "FP64 GPU Monitor - Monitor FP64 instruction usage on AMD GPUs\n"
              << "\n"
              << "This tool monitors hardware performance counters to measure the\n"
              << "percentage of FP64 (double-precision) operations in GPU workloads.\n"
              << "\n"
              << "Options:\n"
              << "  -h, --help              Show this help message\n"
              << "  -V, --version           Show version information\n"
              << "  -l, --list-gpus         List available GPUs and exit\n"
              << "  -c, --list-counters     List available counters for a GPU\n"
              << "  -g, --gpus ID           GPU id for list-counters (default: 0); --gpu accepted as alias\n"
              << "  -s, --show-usage        Show detailed usage instructions\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog_name << " --list-gpus\n"
              << "  " << prog_name << " --list-counters -g 0\n"
              << "  " << prog_name << " --show-usage\n"
              << "\n"
              << "For monitoring, use the tool library with ROCP_TOOL_LIBRARIES:\n"
              << "  ROCP_TOOL_LIBRARIES=libfp64_monitor_tool_<major>.so ./your_hip_app\n"
              << "\n";
}

void list_gpus() {
    std::cout << "Querying available GPUs...\n\n";
    
    auto gpus = measure_flops::CounterSampler::get_available_gpus();
    
    if (gpus.empty()) {
        std::cout << "No GPUs found.\n";
        return;
    }
    
    std::cout << "Available GPUs:\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << std::left
              << std::setw(4) << "ID"
              << std::setw(35) << "Name"
              << std::setw(12) << "GFX Version"
              << std::setw(8) << "CUs"
              << std::setw(15) << "Max Clock"
              << "\n";
    std::cout << std::string(78, '-') << "\n";
    
    for (const auto& gpu : gpus) {
        std::cout << std::left
                  << std::setw(4) << gpu.logical_id
                  << std::setw(35) << gpu.name.substr(0, 33)
                  << std::setw(12) << gpu.gfx_version
                  << std::setw(8) << gpu.compute_units
                  << std::setw(15) << std::to_string(gpu.max_clock_freq / 1000000) + " MHz"
                  << "\n";
    }
    std::cout << std::string(78, '-') << "\n";
    std::cout << "Total: " << gpus.size() << " GPU(s)\n";
}

void list_counters(int gpu_id) {
    auto gpus = measure_flops::CounterSampler::get_available_gpus();
    
    if (gpus.empty()) {
        std::cerr << "Error: No GPUs found.\n";
        return;
    }
    
    if (gpu_id < 0 || gpu_id >= static_cast<int>(gpus.size())) {
        std::cerr << "Error: Invalid GPU ID " << gpu_id << ". Valid range: 0-"
                  << (gpus.size() - 1) << "\n";
        return;
    }
    
    std::cout << "Querying counters for GPU " << gpu_id 
              << " (" << gpus[gpu_id].name << ")...\n\n";
    
    try {
        measure_flops::CounterSampler sampler(gpus[gpu_id]);
        auto counters = sampler.get_available_counters();
        
        if (counters.empty()) {
            std::cout << "No counters found (counter enumeration may require tool mode).\n";
            std::cout << "\nTo use monitoring, run with ROCP_TOOL_LIBRARIES:\n";
            std::cout << "  ROCP_TOOL_LIBRARIES=libfp64_monitor_tool_<major>.so ./your_app\n";
            return;
        }
        
        std::cout << "Available counters (" << counters.size() << " total):\n";
        std::cout << std::string(80, '-') << "\n";
        
        // Group counters by block
        std::map<std::string, std::vector<measure_flops::CounterInfo>> by_block;
        for (const auto& c : counters) {
            by_block[c.block.empty() ? "Other" : c.block].push_back(c);
        }
        
        for (const auto& kv : by_block) {
            std::cout << "\n[" << kv.first << "]\n";
            for (const auto& c : kv.second) {
                std::cout << "  " << std::left << std::setw(40) << c.name;
                if (c.is_derived) std::cout << " (derived)";
                if (c.is_constant) std::cout << " (constant)";
                std::cout << "\n";
                if (!c.description.empty()) {
                    std::string desc = c.description;
                    if (desc.length() > 70) {
                        desc = desc.substr(0, 67) + "...";
                    }
                    std::cout << "    " << desc << "\n";
                }
            }
        }
        
        std::cout << std::string(80, '-') << "\n";
        
        // Highlight FP64-related counters
        std::cout << "\nFP64-related counters:\n";
        for (const auto& c : counters) {
            if (c.name.find("F64") != std::string::npos ||
                c.name.find("f64") != std::string::npos) {
                std::cout << "  " << c.name << "\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

void show_usage_instructions() {
    std::cout << std::string(78, '=') << "\n";
    std::cout << "FP64 GPU Monitor - Usage Instructions\n";
    std::cout << std::string(78, '=') << "\n\n";
    
    std::cout << "BASIC USAGE\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << "Monitor a HIP application using ROCP_TOOL_LIBRARIES:\n\n";
    std::cout << "  ROCP_TOOL_LIBRARIES=/path/to/libfp64_monitor_tool_<major>.so ./your_hip_app\n\n";
    
    std::cout << "By default, output is written to a unique CSV file per process:\n";
    std::cout << "  - With MPI: fp64_monitor_<rank>.csv\n";
    std::cout << "  - Without MPI: fp64_monitor_<hostname>_<pid>.csv\n\n";
    
    std::cout << "ENVIRONMENT VARIABLES\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << std::left
              << std::setw(28) << "Variable" 
              << std::setw(15) << "Default" 
              << "Description\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << std::setw(28) << "FP64_MONITOR_INTERVAL"
              << std::setw(15) << "100"
              << "Sampling interval (ms)\n";
    std::cout << std::setw(28) << "FP64_MONITOR_GPUS"
              << std::setw(15) << "0"
              << "Comma-separated ids or \"all\" (tool library; see fp64_monitor.sh)\n";
    std::cout << std::setw(28) << "FP64_MONITOR_OUTPUT"
              << std::setw(15) << "auto"
              << "Output file (see patterns below)\n";
    std::cout << std::setw(28) << "FP64_MONITOR_FORMAT"
              << std::setw(15) << "csv"
              << "Format: text, csv, json\n";
    std::cout << std::setw(28) << "FP64_MONITOR_VERBOSE"
              << std::setw(15) << "0"
              << "Verbose output (1=enabled)\n";
    std::cout << "\n";
    
    std::cout << "OUTPUT FILE PATTERNS\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << "The FP64_MONITOR_OUTPUT variable supports patterns:\n";
    std::cout << "  %r  - MPI rank (auto-detected from environment)\n";
    std::cout << "  %p  - Process ID\n";
    std::cout << "  %h  - Hostname\n\n";
    std::cout << "Special value:\n";
    std::cout << "  stdout  - Write to console instead of file\n\n";
    
    std::cout << "EXAMPLES\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << "# Basic monitoring (auto-generates unique file)\n";
    std::cout << "ROCP_TOOL_LIBRARIES=libfp64_monitor_tool_<major>.so ./my_app\n\n";
    
    std::cout << "# MPI application (each rank writes to fp64_monitor_<rank>.csv)\n";
    std::cout << "mpirun -np 4 env ROCP_TOOL_LIBRARIES=libfp64_monitor_tool_<major>.so ./my_mpi_app\n\n";
    
    std::cout << "# Custom output filename pattern\n";
    std::cout << "FP64_MONITOR_OUTPUT='results/gpu_rank%r.csv' \\\n";
    std::cout << "  mpirun -np 4 env ROCP_TOOL_LIBRARIES=libfp64_monitor_tool_<major>.so ./my_app\n\n";
    
    std::cout << "# Console output with text format\n";
    std::cout << "FP64_MONITOR_OUTPUT=stdout FP64_MONITOR_FORMAT=text \\\n";
    std::cout << "  ROCP_TOOL_LIBRARIES=libfp64_monitor_tool_<major>.so ./my_app\n\n";
    
    std::cout << "# Fast sampling (50ms) with JSON output\n";
    std::cout << "FP64_MONITOR_INTERVAL=50 FP64_MONITOR_FORMAT=json \\\n";
    std::cout << "  ROCP_TOOL_LIBRARIES=libfp64_monitor_tool_<major>.so ./my_app\n\n";
    
    std::cout << "MPI RANK DETECTION\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << "The tool auto-detects MPI rank from these environment variables:\n";
    std::cout << "  OMPI_COMM_WORLD_RANK    (OpenMPI)\n";
    std::cout << "  PMI_RANK                (MPICH, Cray)\n";
    std::cout << "  SLURM_PROCID            (SLURM)\n";
    std::cout << "  MV2_COMM_WORLD_RANK     (MVAPICH2)\n";
    std::cout << "  PALS_RANKID             (HPE Cray PALS)\n\n";
    
    std::cout << "OUTPUT METRICS\n";
    std::cout << std::string(78, '-') << "\n";
    std::cout << "Key metrics in the output:\n";
    std::cout << "  fp64_valu_pct     - FP64 VALU as % of total VALU instructions\n";
    std::cout << "  fp64_total_pct    - FP64 as % of all instructions\n";
    std::cout << "  fp64_compute_pct  - FP64 as % of compute (excl. memory)\n";
    std::cout << "  mfma_fp64_pct     - FP64 MFMA as % of all MFMA\n\n";
    
    std::cout << "Note: Counter values are per-wavefront (64 threads), not per-thread.\n";
    std::cout << std::string(78, '=') << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    bool list_gpus_flag = false;
    bool list_counters_flag = false;
    bool show_usage_flag = false;
    int gpu_id = 0;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-V" || arg == "--version") {
            print_version();
            return 0;
        } else if (arg == "-l" || arg == "--list-gpus") {
            list_gpus_flag = true;
        } else if (arg == "-c" || arg == "--list-counters") {
            list_counters_flag = true;
        } else if (arg == "-s" || arg == "--show-usage") {
            show_usage_flag = true;
        } else if ((arg == "-g" || arg == "--gpu" || arg == "--gpus") && i + 1 < argc) {
            gpu_id = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (list_gpus_flag) {
        list_gpus();
        return 0;
    }
    
    if (list_counters_flag) {
        list_counters(gpu_id);
        return 0;
    }
    
    if (show_usage_flag) {
        show_usage_instructions();
        return 0;
    }
    
    // Default: show brief usage
    print_usage(argv[0]);
    return 0;
}
