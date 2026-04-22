#include "measure_flops/instruction_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace measure_flops {

double InstructionAnalyzer::sum_counter(const Sample& sample, const std::string& name) {
    double total = 0.0;
    for (const auto& cv : sample.counters) {
        if (cv.name == name) {
            total += cv.value;
        }
    }
    return total;
}

double InstructionAnalyzer::safe_percentage(double numerator, double denominator) {
    if (denominator <= 0.0 || !std::isfinite(denominator)) {
        return 0.0;
    }
    double result = (numerator / denominator) * 100.0;
    if (!std::isfinite(result)) {
        return 0.0;
    }
    return result;
}

InstructionMix InstructionAnalyzer::analyze(const Sample& sample, bool aggregate_dimensions) {
    InstructionMix mix;
    mix.timestamp = sample.timestamp;
    mix.sample_id = sample.sample_id;
    mix.gpu_id = sample.gpu_id;

    if (aggregate_dimensions) {
        // Sum all counter values by name
        mix.total_instructions = sum_counter(sample, "SQ_INSTS");
        mix.valu_instructions = sum_counter(sample, "SQ_INSTS_VALU");
        mix.salu_total = sum_counter(sample, "SQ_INSTS_SALU");
        mix.vmem_total = sum_counter(sample, "SQ_INSTS_VMEM");
        mix.vmem_rd = sum_counter(sample, "SQ_INSTS_VMEM_RD");
        mix.vmem_wr = sum_counter(sample, "SQ_INSTS_VMEM_WR");
        mix.vmem_rd_cycles = sum_counter(sample, "SQ_INST_CYCLES_VMEM_RD");
        mix.vmem_wr_cycles = sum_counter(sample, "SQ_INST_CYCLES_VMEM_WR");
        mix.smem_total = sum_counter(sample, "SQ_INSTS_SMEM");
        mix.lds_total = sum_counter(sample, "SQ_INSTS_LDS");
        mix.flat_total = sum_counter(sample, "SQ_INSTS_FLAT");
        mix.salu_total = sum_counter(sample, "SQ_INSTS_SALU");
        mix.branch_total = sum_counter(sample, "SQ_INSTS_BRANCH");
        mix.mfma_total = sum_counter(sample, "SQ_INSTS_MFMA");
        mix.gds_total = sum_counter(sample, "SQ_INSTS_GDS");
        mix.exp_gds_total = sum_counter(sample, "SQ_INSTS_EXP_GDS");
        mix.sendmsg_total = sum_counter(sample, "SQ_INSTS_SENDMSG");
        mix.vskipped_total = sum_counter(sample, "SQ_INSTS_VSKIPPED");

        // FP64
        mix.fp64_add = sum_counter(sample, "SQ_INSTS_VALU_ADD_F64");
        mix.fp64_mul = sum_counter(sample, "SQ_INSTS_VALU_MUL_F64");
        mix.fp64_fma = sum_counter(sample, "SQ_INSTS_VALU_FMA_F64");
        mix.fp64_trans = sum_counter(sample, "SQ_INSTS_VALU_TRANS_F64");
        mix.fp64_mfma = sum_counter(sample, "SQ_INSTS_VALU_MFMA_F64");
        mix.fp64_mfma_ops = sum_counter(sample, "SQ_INSTS_VALU_MFMA_MOPS_F64") * 512.0;

        // FP32
        mix.fp32_add = sum_counter(sample, "SQ_INSTS_VALU_ADD_F32");
        mix.fp32_mul = sum_counter(sample, "SQ_INSTS_VALU_MUL_F32");
        mix.fp32_fma = sum_counter(sample, "SQ_INSTS_VALU_FMA_F32");
        mix.fp32_trans = sum_counter(sample, "SQ_INSTS_VALU_TRANS_F32");
        mix.fp32_mfma = sum_counter(sample, "SQ_INSTS_VALU_MFMA_F32");

        // FP16 (sum all FP16 counters)
        mix.fp16_total = sum_counter(sample, "SQ_INSTS_VALU_ADD_F16") +
                         sum_counter(sample, "SQ_INSTS_VALU_MUL_F16") +
                         sum_counter(sample, "SQ_INSTS_VALU_FMA_F16") +
                         sum_counter(sample, "SQ_INSTS_VALU_MFMA_F16");

        // Other types
        mix.bf16_mfma = sum_counter(sample, "SQ_INSTS_VALU_MFMA_BF16");
        mix.int8_mfma = sum_counter(sample, "SQ_INSTS_VALU_MFMA_I8");
        mix.int32_total = sum_counter(sample, "SQ_INSTS_VALU_INT32");
        mix.int64_total = sum_counter(sample, "SQ_INSTS_VALU_INT64");
    } else {
        // For non-aggregated mode, we'd need a different data structure
        // For now, just use the same aggregation
        return analyze(sample, true);
    }

    // Compute derived values
    // Include MFMA in fp64_valu_total since SQ_INSTS_VALU includes MFMA instructions
    double fp64_valu_total = mix.fp64_add + mix.fp64_mul + mix.fp64_fma + mix.fp64_trans + mix.fp64_mfma;
    double fp64_all = fp64_valu_total;  // Now the same since MFMA is included above
    double compute_total = mix.valu_instructions + mix.salu_total;

    // Compute percentages
    mix.fp64_valu_pct = safe_percentage(fp64_valu_total, mix.valu_instructions);
    mix.fp64_total_pct = safe_percentage(fp64_all, mix.total_instructions);
    mix.fp64_compute_pct = safe_percentage(fp64_all, compute_total);
    mix.mfma_fp64_pct = safe_percentage(mix.fp64_mfma, mix.mfma_total);

    return mix;
}

InstructionMix InstructionAnalyzer::compute_delta(const InstructionMix& current,
                                                   const InstructionMix& previous) {
    InstructionMix delta;
    delta.timestamp = current.timestamp;
    delta.sample_id = current.sample_id;
    delta.gpu_id = current.gpu_id;

    // Compute deltas for all raw counts
    delta.total_instructions = current.total_instructions - previous.total_instructions;
    delta.valu_instructions = current.valu_instructions - previous.valu_instructions;
    delta.salu_total = current.salu_total - previous.salu_total;
    delta.vmem_total = current.vmem_total - previous.vmem_total;
    delta.vmem_rd = current.vmem_rd - previous.vmem_rd;
    delta.vmem_wr = current.vmem_wr - previous.vmem_wr;
    delta.vmem_rd_cycles = current.vmem_rd_cycles - previous.vmem_rd_cycles;
    delta.vmem_wr_cycles = current.vmem_wr_cycles - previous.vmem_wr_cycles;
    delta.smem_total = current.smem_total - previous.smem_total;
    delta.lds_total = current.lds_total - previous.lds_total;
    delta.flat_total = current.flat_total - previous.flat_total;
    delta.salu_total = current.salu_total - previous.salu_total;
    delta.branch_total = current.branch_total - previous.branch_total;
    delta.mfma_total = current.mfma_total - previous.mfma_total;
    delta.gds_total = current.gds_total - previous.gds_total;
    delta.exp_gds_total = current.exp_gds_total - previous.exp_gds_total;
    delta.sendmsg_total = current.sendmsg_total - previous.sendmsg_total;
    delta.vskipped_total = current.vskipped_total - previous.vskipped_total;

    delta.fp64_add = current.fp64_add - previous.fp64_add;
    delta.fp64_mul = current.fp64_mul - previous.fp64_mul;
    delta.fp64_fma = current.fp64_fma - previous.fp64_fma;
    delta.fp64_trans = current.fp64_trans - previous.fp64_trans;
    delta.fp64_mfma = current.fp64_mfma - previous.fp64_mfma;
    delta.fp64_mfma_ops = current.fp64_mfma_ops - previous.fp64_mfma_ops;

    delta.fp32_add = current.fp32_add - previous.fp32_add;
    delta.fp32_mul = current.fp32_mul - previous.fp32_mul;
    delta.fp32_fma = current.fp32_fma - previous.fp32_fma;
    delta.fp32_trans = current.fp32_trans - previous.fp32_trans;
    delta.fp32_mfma = current.fp32_mfma - previous.fp32_mfma;

    delta.fp16_total = current.fp16_total - previous.fp16_total;
    delta.bf16_mfma = current.bf16_mfma - previous.bf16_mfma;
    delta.int8_mfma = current.int8_mfma - previous.int8_mfma;
    delta.int32_total = current.int32_total - previous.int32_total;
    delta.int64_total = current.int64_total - previous.int64_total;

    // Recompute percentages from deltas
    // Include MFMA in fp64_valu_total since SQ_INSTS_VALU includes MFMA instructions
    double fp64_valu_total = delta.fp64_add + delta.fp64_mul + delta.fp64_fma + delta.fp64_trans + delta.fp64_mfma;
    double fp64_all = fp64_valu_total;
    double compute_total = delta.valu_instructions + delta.salu_total;

    delta.fp64_valu_pct = safe_percentage(fp64_valu_total, delta.valu_instructions);
    delta.fp64_total_pct = safe_percentage(fp64_all, delta.total_instructions);
    delta.fp64_compute_pct = safe_percentage(fp64_all, compute_total);
    delta.mfma_fp64_pct = safe_percentage(delta.fp64_mfma, delta.mfma_total);

    return delta;
}

InstructionAnalyzer::Summary InstructionAnalyzer::compute_summary(
    const std::vector<InstructionMix>& mixes) {
    Summary summary;

    if (mixes.empty()) {
        return summary;
    }

    summary.total_samples = mixes.size();

    // Compute duration
    if (mixes.size() >= 2) {
        auto duration = mixes.back().timestamp - mixes.front().timestamp;
        summary.duration_seconds = std::chrono::duration<double>(duration).count();
    }

    // Initialize min/max
    summary.min_fp64_valu_pct = std::numeric_limits<double>::max();
    summary.max_fp64_valu_pct = std::numeric_limits<double>::lowest();
    summary.min_fp64_total_pct = std::numeric_limits<double>::max();
    summary.max_fp64_total_pct = std::numeric_limits<double>::lowest();

    double sum_fp64_valu_pct = 0.0;
    double sum_fp64_total_pct = 0.0;

    for (const auto& mix : mixes) {
        // Accumulate totals (include MFMA since SQ_INSTS_VALU includes MFMA)
        double fp64_valu = mix.fp64_add + mix.fp64_mul + mix.fp64_fma + mix.fp64_trans + mix.fp64_mfma;
        summary.total_fp64_valu += fp64_valu;
        summary.total_fp64_mfma += mix.fp64_mfma;  // Track separately for reporting
        summary.total_valu += mix.valu_instructions;
        summary.total_instructions += mix.total_instructions;

        // Track percentages
        sum_fp64_valu_pct += mix.fp64_valu_pct;
        sum_fp64_total_pct += mix.fp64_total_pct;

        summary.min_fp64_valu_pct = std::min(summary.min_fp64_valu_pct, mix.fp64_valu_pct);
        summary.max_fp64_valu_pct = std::max(summary.max_fp64_valu_pct, mix.fp64_valu_pct);
        summary.min_fp64_total_pct = std::min(summary.min_fp64_total_pct, mix.fp64_total_pct);
        summary.max_fp64_total_pct = std::max(summary.max_fp64_total_pct, mix.fp64_total_pct);
    }

    // Compute averages
    summary.avg_fp64_valu_pct = sum_fp64_valu_pct / static_cast<double>(mixes.size());
    summary.avg_fp64_total_pct = sum_fp64_total_pct / static_cast<double>(mixes.size());

    // Handle edge case where no valid percentages were computed
    if (summary.min_fp64_valu_pct == std::numeric_limits<double>::max()) {
        summary.min_fp64_valu_pct = 0.0;
    }
    if (summary.max_fp64_valu_pct == std::numeric_limits<double>::lowest()) {
        summary.max_fp64_valu_pct = 0.0;
    }
    if (summary.min_fp64_total_pct == std::numeric_limits<double>::max()) {
        summary.min_fp64_total_pct = 0.0;
    }
    if (summary.max_fp64_total_pct == std::numeric_limits<double>::lowest()) {
        summary.max_fp64_total_pct = 0.0;
    }

    return summary;
}

std::vector<std::string> InstructionAnalyzer::get_required_counters() {
    return CounterSets::fp64_minimal();
}

std::vector<std::string> InstructionAnalyzer::get_fp64_counters() {
    return CounterSets::fp64_minimal();
}

}  // namespace measure_flops
