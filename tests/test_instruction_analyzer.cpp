// Unit tests for InstructionAnalyzer
//
// Build: make test_instruction_analyzer
// Run:   ./tests/test_instruction_analyzer

#include "measure_flops/instruction_analyzer.hpp"
#include "measure_flops/types.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace measure_flops;

// Helper to check floating point equality with tolerance
bool approx_equal(double a, double b, double tolerance = 0.0001) {
    return std::abs(a - b) < tolerance;
}

// Helper to create a sample with specific counter values
Sample create_sample(uint64_t id, int gpu_id,
                     double total_insts, double valu_insts,
                     double fp64_add, double fp64_mul, double fp64_fma, double fp64_trans,
                     double fp64_mfma = 0.0) {
    Sample sample;
    sample.sample_id = id;
    sample.gpu_id = gpu_id;
    sample.timestamp = std::chrono::steady_clock::now();
    
    sample.counters.push_back({"SQ_INSTS", total_insts, 0, 0, {}});
    sample.counters.push_back({"SQ_INSTS_VALU", valu_insts, 1, 0, {}});
    sample.counters.push_back({"SQ_INSTS_VALU_ADD_F64", fp64_add, 2, 0, {}});
    sample.counters.push_back({"SQ_INSTS_VALU_MUL_F64", fp64_mul, 3, 0, {}});
    sample.counters.push_back({"SQ_INSTS_VALU_FMA_F64", fp64_fma, 4, 0, {}});
    sample.counters.push_back({"SQ_INSTS_VALU_TRANS_F64", fp64_trans, 5, 0, {}});
    sample.counters.push_back({"SQ_INSTS_VALU_MFMA_F64", fp64_mfma, 6, 0, {}});
    sample.counters.push_back({"SQ_INSTS_SALU", 0, 7, 0, {}});
    sample.counters.push_back({"SQ_INSTS_MFMA", fp64_mfma, 8, 0, {}});
    
    return sample;
}

void test_basic_analysis() {
    std::cout << "Test: Basic analysis... ";
    
    // Create a sample with known values
    // 1000 total instructions, 800 VALU, 100 FP64 FMA
    Sample sample = create_sample(1, 0, 1000, 800, 0, 0, 100, 0);
    
    [[maybe_unused]] InstructionMix mix = InstructionAnalyzer::analyze(sample);
    
    assert(mix.sample_id == 1);
    assert(mix.gpu_id == 0);
    assert(approx_equal(mix.total_instructions, 1000));
    assert(approx_equal(mix.valu_instructions, 800));
    assert(approx_equal(mix.fp64_fma, 100));
    
    // FP64 VALU percentage: 100/800 * 100 = 12.5%
    assert(approx_equal(mix.fp64_valu_pct, 12.5));
    
    // FP64 total percentage: 100/1000 * 100 = 10%
    assert(approx_equal(mix.fp64_total_pct, 10.0));
    
    std::cout << "PASSED\n";
}

void test_multiple_fp64_types() {
    std::cout << "Test: Multiple FP64 instruction types... ";
    
    // 1000 total, 500 VALU
    // FP64: 50 ADD + 30 MUL + 100 FMA + 20 TRANS = 200
    Sample sample = create_sample(1, 0, 1000, 500, 50, 30, 100, 20);
    
    [[maybe_unused]] InstructionMix mix = InstructionAnalyzer::analyze(sample);
    
    assert(approx_equal(mix.fp64_add, 50));
    assert(approx_equal(mix.fp64_mul, 30));
    assert(approx_equal(mix.fp64_fma, 100));
    assert(approx_equal(mix.fp64_trans, 20));
    
    // FP64 VALU percentage: 200/500 * 100 = 40%
    assert(approx_equal(mix.fp64_valu_pct, 40.0));
    
    // FP64 total percentage: 200/1000 * 100 = 20%
    assert(approx_equal(mix.fp64_total_pct, 20.0));
    
    std::cout << "PASSED\n";
}

void test_with_mfma() {
    std::cout << "Test: FP64 with MFMA... ";
    
    // 10000 total, 5000 VALU
    // FP64 VALU: 100 FMA
    // FP64 MFMA: 50
    Sample sample = create_sample(1, 0, 10000, 5000, 0, 0, 100, 0, 50);
    
    [[maybe_unused]] InstructionMix mix = InstructionAnalyzer::analyze(sample);
    
    assert(approx_equal(mix.fp64_fma, 100));
    assert(approx_equal(mix.fp64_mfma, 50));
    
    // FP64 VALU percentage: (100+50)/5000 * 100 = 3% (MFMA included in VALU)
    assert(approx_equal(mix.fp64_valu_pct, 3.0));
    
    // FP64 total percentage: (100+50)/10000 * 100 = 1.5%
    assert(approx_equal(mix.fp64_total_pct, 1.5));
    
    // MFMA FP64 percentage: 50/50 * 100 = 100% (all MFMA is FP64)
    assert(approx_equal(mix.mfma_fp64_pct, 100.0));
    
    std::cout << "PASSED\n";
}

void test_zero_values() {
    std::cout << "Test: Zero values... ";
    
    // All zeros
    Sample sample = create_sample(1, 0, 0, 0, 0, 0, 0, 0);
    
    [[maybe_unused]] InstructionMix mix = InstructionAnalyzer::analyze(sample);
    
    // Should handle division by zero gracefully
    assert(approx_equal(mix.fp64_valu_pct, 0.0));
    assert(approx_equal(mix.fp64_total_pct, 0.0));
    assert(approx_equal(mix.fp64_compute_pct, 0.0));
    
    std::cout << "PASSED\n";
}

void test_empty_sample() {
    std::cout << "Test: Empty sample... ";
    
    Sample sample;
    sample.sample_id = 1;
    sample.gpu_id = 0;
    sample.timestamp = std::chrono::steady_clock::now();
    // No counters
    
    [[maybe_unused]] InstructionMix mix = InstructionAnalyzer::analyze(sample);
    
    // Should return zeros without crashing
    assert(approx_equal(mix.total_instructions, 0.0));
    assert(approx_equal(mix.valu_instructions, 0.0));
    assert(approx_equal(mix.fp64_valu_pct, 0.0));
    
    std::cout << "PASSED\n";
}

void test_compute_delta() {
    std::cout << "Test: Compute delta... ";
    
    // First sample: 1000 total, 500 VALU, 50 FP64 FMA
    Sample sample1 = create_sample(1, 0, 1000, 500, 0, 0, 50, 0);
    InstructionMix mix1 = InstructionAnalyzer::analyze(sample1);
    
    // Second sample: 2000 total, 1000 VALU, 150 FP64 FMA (delta = 100 FMA)
    Sample sample2 = create_sample(2, 0, 2000, 1000, 0, 0, 150, 0);
    InstructionMix mix2 = InstructionAnalyzer::analyze(sample2);
    
    [[maybe_unused]] InstructionMix delta = InstructionAnalyzer::compute_delta(mix2, mix1);
    
    assert(approx_equal(delta.total_instructions, 1000));
    assert(approx_equal(delta.valu_instructions, 500));
    assert(approx_equal(delta.fp64_fma, 100));
    
    // FP64 VALU percentage of delta: 100/500 * 100 = 20%
    assert(approx_equal(delta.fp64_valu_pct, 20.0));
    
    std::cout << "PASSED\n";
}

void test_compute_summary() {
    std::cout << "Test: Compute summary... ";
    
    std::vector<InstructionMix> mixes;
    
    // Create several mixes with different FP64 percentages
    for (int i = 0; i < 5; i++) {
        InstructionMix mix;
        mix.sample_id = i;
        mix.gpu_id = 0;
        mix.timestamp = std::chrono::steady_clock::now();
        
        mix.total_instructions = 1000;
        mix.valu_instructions = 500;
        mix.fp64_add = 10 * (i + 1);  // 10, 20, 30, 40, 50
        mix.fp64_mul = 0;
        mix.fp64_fma = 0;
        mix.fp64_trans = 0;
        mix.fp64_mfma = 0;
        
        // FP64 VALU percentages: 2%, 4%, 6%, 8%, 10%
        mix.fp64_valu_pct = (10.0 * (i + 1)) / 500.0 * 100.0;
        mix.fp64_total_pct = (10.0 * (i + 1)) / 1000.0 * 100.0;
        
        mixes.push_back(mix);
    }
    
    [[maybe_unused]] auto summary = InstructionAnalyzer::compute_summary(mixes);
    
    assert(summary.total_samples == 5);
    
    // Total FP64 VALU: 10+20+30+40+50 = 150
    assert(approx_equal(summary.total_fp64_valu, 150.0));
    
    // Min/max FP64 VALU %: 2% to 10%
    assert(approx_equal(summary.min_fp64_valu_pct, 2.0));
    assert(approx_equal(summary.max_fp64_valu_pct, 10.0));
    
    // Average FP64 VALU %: (2+4+6+8+10)/5 = 6%
    assert(approx_equal(summary.avg_fp64_valu_pct, 6.0));
    
    std::cout << "PASSED\n";
}

void test_empty_summary() {
    std::cout << "Test: Empty summary... ";
    
    std::vector<InstructionMix> mixes;
    
    [[maybe_unused]] auto summary = InstructionAnalyzer::compute_summary(mixes);
    
    assert(summary.total_samples == 0);
    assert(approx_equal(summary.duration_seconds, 0.0));
    assert(approx_equal(summary.avg_fp64_valu_pct, 0.0));
    
    std::cout << "PASSED\n";
}

void test_aggregation_across_dimensions() {
    std::cout << "Test: Aggregation across dimensions (XCCs)... ";
    
    Sample sample;
    sample.sample_id = 1;
    sample.gpu_id = 0;
    sample.timestamp = std::chrono::steady_clock::now();
    
    // Simulate 8 XCCs, each reporting counters
    for (int xcc = 0; xcc < 8; xcc++) {
        sample.counters.push_back({"SQ_INSTS", 1000.0, static_cast<uint64_t>(xcc * 10), 0, {}});
        sample.counters.push_back({"SQ_INSTS_VALU", 500.0, static_cast<uint64_t>(xcc * 10 + 1), 0, {}});
        sample.counters.push_back({"SQ_INSTS_VALU_FMA_F64", 50.0, static_cast<uint64_t>(xcc * 10 + 2), 0, {}});
    }
    
    [[maybe_unused]] InstructionMix mix = InstructionAnalyzer::analyze(sample);
    
    // Should sum across all XCCs
    assert(approx_equal(mix.total_instructions, 8000.0));
    assert(approx_equal(mix.valu_instructions, 4000.0));
    assert(approx_equal(mix.fp64_fma, 400.0));
    
    // FP64 VALU percentage: 400/4000 * 100 = 10%
    assert(approx_equal(mix.fp64_valu_pct, 10.0));
    
    std::cout << "PASSED\n";
}

void test_100_percent_fp64() {
    std::cout << "Test: 100% FP64 workload... ";
    
    // All VALU instructions are FP64 FMA
    Sample sample = create_sample(1, 0, 1000, 1000, 0, 0, 1000, 0);
    
    [[maybe_unused]] InstructionMix mix = InstructionAnalyzer::analyze(sample);
    
    // Should be 100% FP64 VALU
    assert(approx_equal(mix.fp64_valu_pct, 100.0));
    assert(approx_equal(mix.fp64_total_pct, 100.0));
    
    std::cout << "PASSED\n";
}

void test_get_counters() {
    std::cout << "Test: Get counter lists... ";
    
    auto required = InstructionAnalyzer::get_required_counters();
    auto fp64_only = InstructionAnalyzer::get_fp64_counters();
    
    // Should have reasonable sizes
    assert(required.size() > 0);
    assert(fp64_only.size() > 0);
    assert(fp64_only.size() <= required.size());
    
    // FP64 counters should include the essential FP64 counters
    [[maybe_unused]] bool has_fma = false;
    for (const auto& c : fp64_only) {
        if (c == "SQ_INSTS_VALU_FMA_F64") {
            has_fma = true;
            break;
        }
    }
    assert(has_fma);
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== InstructionAnalyzer Unit Tests ===\n\n";
    
    try {
        test_basic_analysis();
        test_multiple_fp64_types();
        test_with_mfma();
        test_zero_values();
        test_empty_sample();
        test_compute_delta();
        test_compute_summary();
        test_empty_summary();
        test_aggregation_across_dimensions();
        test_100_percent_fp64();
        test_get_counters();
        
        std::cout << "\n=== All tests PASSED ===\n\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << "\n";
        return 1;
    }
}
