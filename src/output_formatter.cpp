#include "measure_flops/output_formatter.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>

namespace measure_flops {

std::unique_ptr<OutputFormatter> OutputFormatter::create(const std::string& format) {
    if (format == "csv") {
        return std::make_unique<CsvFormatter>();
    } else if (format == "json") {
        return std::make_unique<JsonFormatter>();
    } else {
        return std::make_unique<TextFormatter>();
    }
}

// ============================================================================
// TextFormatter
// ============================================================================

void TextFormatter::write_header(std::ostream& os, const MonitorConfig& config) {
    os << "\n";
    os << std::string(78, '=') << "\n";
    os << "FP64 GPU Monitor\n";
    os << std::string(78, '=') << "\n";
    os << "Sampling interval: " << config.sampling_interval.count() << " ms\n";
    os << "Output format:     " << config.output_format << "\n";
    if (!config.gpu_ids.empty()) {
        os << "GPU IDs:           ";
        for (size_t i = 0; i < config.gpu_ids.size(); ++i) {
            if (i > 0) os << ", ";
            os << config.gpu_ids[i];
        }
        os << "\n";
    }
    if (config.metadata_monitored_gpu_ids.size() > 1) {
        os << "Run GPU set:       ";
        for (size_t i = 0; i < config.metadata_monitored_gpu_ids.size(); ++i) {
            if (i > 0) os << ", ";
            os << config.metadata_monitored_gpu_ids[i];
        }
        os << " (this file is one GPU of the run)\n";
    }
    os << std::string(78, '=') << "\n\n";

    os << std::left
       << std::setw(8) << "Sample"
       << std::setw(10) << "Time(s)"
       << std::setw(12) << "GPU"
       << std::setw(14) << "FP64 VALU"
       << std::setw(14) << "Total VALU"
       << std::setw(10) << "FP64%"
       << std::setw(10) << "MFMA64%"
       << "\n";
    os << std::string(78, '-') << "\n";
    
    header_written_ = true;
}

void TextFormatter::write_sample(std::ostream& os, const InstructionMix& mix,
                                  double elapsed_seconds,
                                  const std::string& counter_set) {
    (void)counter_set;  // Not displayed in text format
    double fp64_valu = mix.fp64_add + mix.fp64_mul + mix.fp64_fma + mix.fp64_trans;

    os << std::left << std::fixed
       << std::setw(8) << mix.sample_id
       << std::setw(10) << std::setprecision(2) << elapsed_seconds
       << std::setw(12) << mix.gpu_id
       << std::setw(14) << static_cast<int64_t>(fp64_valu)
       << std::setw(14) << static_cast<int64_t>(mix.valu_instructions)
       << std::setw(10) << std::setprecision(2) << mix.fp64_valu_pct
       << std::setw(10) << std::setprecision(2) << mix.mfma_fp64_pct
       << "\n";
}

void TextFormatter::write_footer(std::ostream& os,
                                  const InstructionAnalyzer::Summary& summary) {
    os << std::string(78, '-') << "\n";
    os << "\n";
    os << std::string(78, '=') << "\n";
    os << "Summary\n";
    os << std::string(78, '=') << "\n";
    os << std::fixed << std::setprecision(2);
    os << "Total samples:      " << summary.total_samples << "\n";
    os << "Duration:           " << summary.duration_seconds << " seconds\n";
    os << "\n";
    os << "FP64 VALU Percentage:\n";
    os << "  Average:          " << summary.avg_fp64_valu_pct << "%\n";
    os << "  Min:              " << summary.min_fp64_valu_pct << "%\n";
    os << "  Max:              " << summary.max_fp64_valu_pct << "%\n";
    os << "\n";
    os << "FP64 Total Percentage (of all instructions):\n";
    os << "  Average:          " << summary.avg_fp64_total_pct << "%\n";
    os << "  Min:              " << summary.min_fp64_total_pct << "%\n";
    os << "  Max:              " << summary.max_fp64_total_pct << "%\n";
    os << "\n";
    os << "Cumulative Counts:\n";
    os << "  Total FP64 VALU:  " << static_cast<int64_t>(summary.total_fp64_valu) << "\n";
    os << "  Total FP64 MFMA:  " << static_cast<int64_t>(summary.total_fp64_mfma) << "\n";
    os << "  Total VALU:       " << static_cast<int64_t>(summary.total_valu) << "\n";
    os << "  Total Insts:      " << static_cast<int64_t>(summary.total_instructions) << "\n";
    os << std::string(78, '=') << "\n";
}

// ============================================================================
// CsvFormatter
// ============================================================================

void CsvFormatter::write_header(std::ostream& os, const MonitorConfig& config) {
    (void)config;  // Not used in CSV header
    
    os << "sample_id,elapsed_seconds,gpu_id,counter_set,"
       << "total_instructions,valu_instructions,salu_total,smem_total,vmem_total,vmem_rd,vmem_wr,vmem_rd_cycles,vmem_wr_cycles,lds_total,flat_total,gds_total,exp_gds_total,sendmsg_total,vskipped_total,"
       << "fp64_add,fp64_mul,fp64_fma,fp64_trans,fp64_mfma,"
       << "fp32_add,fp32_mul,fp32_fma,fp32_trans,fp32_mfma,"
       << "fp16_total,bf16_mfma,int8_mfma,int32_total,int64_total,"
       << "mfma_total,branch_total,"
       << "fp64_valu_pct,fp64_total_pct,fp64_compute_pct,mfma_fp64_pct,"
       << "vram_used_mb,vram_total_mb,gtt_used_mb,gtt_total_mb,"
       << "cpu_resident_mb,cpu_virtual_mb,"
       << "vram_hwm_mb,gtt_hwm_mb,cpu_resident_hwm_mb,cpu_vm_hwm_mb"
       << "\n";
}

void CsvFormatter::write_sample(std::ostream& os, const InstructionMix& mix,
                                 double elapsed_seconds,
                                 const std::string& counter_set) {
    os << std::fixed << std::setprecision(6);
    
    // Helper lambda to convert bytes to MB
    auto to_mb = [](uint64_t bytes) -> double {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };
    
    os << mix.sample_id << ","
       << elapsed_seconds << ","
       << mix.gpu_id << ","
       << counter_set << ","
       << static_cast<int64_t>(mix.total_instructions) << ","
       << static_cast<int64_t>(mix.valu_instructions) << ","
       << static_cast<int64_t>(mix.salu_total) << ","
       << static_cast<int64_t>(mix.smem_total) << ","
       << static_cast<int64_t>(mix.vmem_total) << ","
       << static_cast<int64_t>(mix.vmem_rd) << ","
       << static_cast<int64_t>(mix.vmem_wr) << ","
       << static_cast<int64_t>(mix.vmem_rd_cycles) << ","
       << static_cast<int64_t>(mix.vmem_wr_cycles) << ","
       << static_cast<int64_t>(mix.lds_total) << ","
       << static_cast<int64_t>(mix.flat_total) << ","
       << static_cast<int64_t>(mix.gds_total) << ","
       << static_cast<int64_t>(mix.exp_gds_total) << ","
       << static_cast<int64_t>(mix.sendmsg_total) << ","
       << static_cast<int64_t>(mix.vskipped_total) << ","
       << static_cast<int64_t>(mix.fp64_add) << ","
       << static_cast<int64_t>(mix.fp64_mul) << ","
       << static_cast<int64_t>(mix.fp64_fma) << ","
       << static_cast<int64_t>(mix.fp64_trans) << ","
       << static_cast<int64_t>(mix.fp64_mfma) << ","
       << static_cast<int64_t>(mix.fp32_add) << ","
       << static_cast<int64_t>(mix.fp32_mul) << ","
       << static_cast<int64_t>(mix.fp32_fma) << ","
       << static_cast<int64_t>(mix.fp32_trans) << ","
       << static_cast<int64_t>(mix.fp32_mfma) << ","
       << static_cast<int64_t>(mix.fp16_total) << ","
       << static_cast<int64_t>(mix.bf16_mfma) << ","
       << static_cast<int64_t>(mix.int8_mfma) << ","
       << static_cast<int64_t>(mix.int32_total) << ","
       << static_cast<int64_t>(mix.int64_total) << ","
       << static_cast<int64_t>(mix.mfma_total) << ","
       << static_cast<int64_t>(mix.branch_total) << ","
       << std::setprecision(4)
       << mix.fp64_valu_pct << ","
       << mix.fp64_total_pct << ","
       << mix.fp64_compute_pct << ","
       << mix.mfma_fp64_pct << ","
       << std::setprecision(2)
       << to_mb(mix.vram_used) << ","
       << to_mb(mix.vram_total) << ","
       << to_mb(mix.gtt_used) << ","
       << to_mb(mix.gtt_total) << ","
       << to_mb(mix.cpu_mem_resident) << ","
       << to_mb(mix.cpu_mem_virtual) << ","
       << to_mb(mix.vram_hwm) << ","
       << to_mb(mix.gtt_hwm) << ","
       << to_mb(mix.cpu_mem_hwm) << ","
       << to_mb(mix.cpu_vm_hwm)
       << "\n";
}

void CsvFormatter::write_footer(std::ostream& os,
                                 const InstructionAnalyzer::Summary& summary) {
    // CSV typically doesn't have a footer, but we can add a comment
    os << "# Summary: samples=" << summary.total_samples
       << ",duration=" << summary.duration_seconds
       << ",avg_fp64_valu_pct=" << summary.avg_fp64_valu_pct
       << ",avg_fp64_total_pct=" << summary.avg_fp64_total_pct
       << "\n";
}

// ============================================================================
// JsonFormatter
// ============================================================================

void JsonFormatter::write_header(std::ostream& os, const MonitorConfig& config) {
    // Get current time
    auto now = std::time(nullptr);
    char time_buf[64];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    os << "{\n";
    os << "  \"metadata\": {\n";
    os << "    \"tool\": \"FP64 GPU Monitor\",\n";
    os << "    \"version\": \"1.0.0\",\n";
    os << "    \"start_time\": \"" << time_buf << "\",\n";
    os << "    \"sampling_interval_ms\": " << config.sampling_interval.count() << ",\n";
    os << "    \"gpu_ids\": [";
    for (size_t i = 0; i < config.gpu_ids.size(); ++i) {
        if (i > 0) os << ", ";
        os << config.gpu_ids[i];
    }
    os << "]";
    if (!config.metadata_monitored_gpu_ids.empty()) {
        os << ",\n    \"monitored_gpu_ids\": [";
        for (size_t i = 0; i < config.metadata_monitored_gpu_ids.size(); ++i) {
            if (i > 0) os << ", ";
            os << config.metadata_monitored_gpu_ids[i];
        }
        os << "]";
    }
    os << "\n  },\n";
    os << "  \"samples\": [\n";
    
    first_sample_ = true;
}

void JsonFormatter::write_sample(std::ostream& os, const InstructionMix& mix,
                                  double elapsed_seconds,
                                  const std::string& counter_set) {
    if (!first_sample_) {
        os << ",\n";
    }
    first_sample_ = false;

    double fp64_valu = mix.fp64_add + mix.fp64_mul + mix.fp64_fma + mix.fp64_trans;

    os << std::fixed;
    os << "    {\n";
    os << "      \"sample_id\": " << mix.sample_id << ",\n";
    os << "      \"elapsed_seconds\": " << std::setprecision(3) << elapsed_seconds << ",\n";
    os << "      \"gpu_id\": " << mix.gpu_id << ",\n";
    if (!counter_set.empty()) {
        os << "      \"counter_set\": \"" << counter_set << "\",\n";
    }
    os << "      \"counters\": {\n";
    os << "        \"total_instructions\": " << static_cast<int64_t>(mix.total_instructions) << ",\n";
    os << "        \"valu_instructions\": " << static_cast<int64_t>(mix.valu_instructions) << ",\n";
    os << "        \"salu_total\": " << static_cast<int64_t>(mix.salu_total) << ",\n";
    os << "        \"vmem_total\": " << static_cast<int64_t>(mix.vmem_total) << ",\n";
    os << "        \"mfma_total\": " << static_cast<int64_t>(mix.mfma_total) << ",\n";
    os << "        \"fp64_valu\": " << static_cast<int64_t>(fp64_valu) << ",\n";
    os << "        \"fp64_mfma\": " << static_cast<int64_t>(mix.fp64_mfma) << "\n";
    os << "      },\n";
    os << "      \"analysis\": {\n";
    os << "        \"fp64_valu_pct\": " << std::setprecision(4) << mix.fp64_valu_pct << ",\n";
    os << "        \"fp64_total_pct\": " << mix.fp64_total_pct << ",\n";
    os << "        \"fp64_compute_pct\": " << mix.fp64_compute_pct << ",\n";
    os << "        \"mfma_fp64_pct\": " << mix.mfma_fp64_pct << "\n";
    os << "      }\n";
    os << "    }";
}

void JsonFormatter::write_footer(std::ostream& os,
                                  const InstructionAnalyzer::Summary& summary) {
    os << "\n  ],\n";
    os << "  \"summary\": {\n";
    os << "    \"total_samples\": " << summary.total_samples << ",\n";
    os << "    \"duration_seconds\": " << std::fixed << std::setprecision(3) 
       << summary.duration_seconds << ",\n";
    os << "    \"fp64_valu_pct\": {\n";
    os << "      \"average\": " << std::setprecision(4) << summary.avg_fp64_valu_pct << ",\n";
    os << "      \"min\": " << summary.min_fp64_valu_pct << ",\n";
    os << "      \"max\": " << summary.max_fp64_valu_pct << "\n";
    os << "    },\n";
    os << "    \"fp64_total_pct\": {\n";
    os << "      \"average\": " << summary.avg_fp64_total_pct << ",\n";
    os << "      \"min\": " << summary.min_fp64_total_pct << ",\n";
    os << "      \"max\": " << summary.max_fp64_total_pct << "\n";
    os << "    },\n";
    os << "    \"cumulative\": {\n";
    os << "      \"total_fp64_valu\": " << static_cast<int64_t>(summary.total_fp64_valu) << ",\n";
    os << "      \"total_fp64_mfma\": " << static_cast<int64_t>(summary.total_fp64_mfma) << ",\n";
    os << "      \"total_valu\": " << static_cast<int64_t>(summary.total_valu) << ",\n";
    os << "      \"total_instructions\": " << static_cast<int64_t>(summary.total_instructions) << "\n";
    os << "    }\n";
    os << "  }\n";
    os << "}\n";
}

}  // namespace measure_flops
