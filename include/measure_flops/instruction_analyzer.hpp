#pragma once

#include "measure_flops/types.hpp"

#include <string>
#include <vector>

namespace measure_flops {

class InstructionAnalyzer {
public:
    // Analyze a sample to produce instruction mix breakdown
    // If aggregate_dimensions is true, sum counter values across all instances
    static InstructionMix analyze(const Sample& sample, bool aggregate_dimensions = true);

    // Compute delta between two instruction mixes (for rate calculation)
    // Returns the difference: current - previous
    static InstructionMix compute_delta(const InstructionMix& current,
                                        const InstructionMix& previous);

    // Compute summary statistics from a collection of samples
    struct Summary {
        size_t total_samples;
        double duration_seconds;
        
        // FP64 percentages
        double avg_fp64_valu_pct;
        double min_fp64_valu_pct;
        double max_fp64_valu_pct;
        
        double avg_fp64_total_pct;
        double min_fp64_total_pct;
        double max_fp64_total_pct;
        
        // Total instruction counts
        double total_fp64_valu;
        double total_fp64_mfma;
        double total_valu;
        double total_instructions;
        
        Summary()
            : total_samples(0),
              duration_seconds(0),
              avg_fp64_valu_pct(0),
              min_fp64_valu_pct(0),
              max_fp64_valu_pct(0),
              avg_fp64_total_pct(0),
              min_fp64_total_pct(0),
              max_fp64_total_pct(0),
              total_fp64_valu(0),
              total_fp64_mfma(0),
              total_valu(0),
              total_instructions(0) {}
    };

    static Summary compute_summary(const std::vector<InstructionMix>& mixes);

    // Get the list of counters needed for full instruction mix analysis
    static std::vector<std::string> get_required_counters();

    // Get minimal counters for FP64 percentage only
    static std::vector<std::string> get_fp64_counters();

private:
    // Helper to sum counter values by name (across dimensions/instances)
    static double sum_counter(const Sample& sample, const std::string& name);

    // Helper to compute percentages safely (avoid division by zero)
    static double safe_percentage(double numerator, double denominator);
};

}  // namespace measure_flops
