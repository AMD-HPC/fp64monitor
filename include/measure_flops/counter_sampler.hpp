/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE in the project root for license information.
 */

#pragma once

#include "measure_flops/rocprofiler_compat.hpp"
#include "measure_flops/types.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace measure_flops {

// GPU agent information
struct GpuAgent {
    rocprofiler_agent_id_t id;
    std::string name;
    std::string gfx_version;
    int logical_id;
    uint32_t compute_units;
    uint64_t max_clock_freq;
};

// Counter information
struct CounterInfo {
    rocprofiler_counter_id_t id;
    std::string name;
    std::string description;
    std::string block;
    bool is_derived;
    bool is_constant;
};

class CounterSampler {
public:
    // Initialize sampler for a specific GPU agent
    explicit CounterSampler(const GpuAgent& agent);
    ~CounterSampler();

    // Non-copyable
    CounterSampler(const CounterSampler&) = delete;
    CounterSampler& operator=(const CounterSampler&) = delete;

    // Movable
    CounterSampler(CounterSampler&& other) noexcept;
    CounterSampler& operator=(CounterSampler&& other) noexcept;

    // Get available GPU agents on the system
    static std::vector<GpuAgent> get_available_gpus();

    // Get list of all available counter info for this agent
    std::vector<CounterInfo> get_available_counters() const;

    // Get list of counter names only
    std::vector<std::string> get_available_counter_names() const;

    // Check if a specific counter is available
    bool is_counter_available(const std::string& name) const;

    // Configure which counters to collect
    // Returns false if any counter is unavailable
    bool configure_counters(const std::vector<std::string>& counter_names);

    // Get the list of currently configured counters
    const std::vector<std::string>& get_configured_counters() const {
        return configured_counters_;
    }

    // Take a single sample of all configured counters
    // Returns empty Sample on error
    Sample take_sample();

    // Get the GPU agent this sampler is configured for
    const GpuAgent& get_agent() const { return agent_; }

    // Get last error message
    const std::string& get_last_error() const { return last_error_; }

    // Check if sampler is ready to take samples
    bool is_ready() const { return context_initialized_ && !configured_counters_.empty(); }

private:
    GpuAgent agent_;
    rocprofiler_context_id_t context_{};
    rocprofiler_buffer_id_t buffer_{};
    rocprofiler_compat::config_id_t profile_{.handle = 0};

    // Counter name <-> ID mappings
    std::unordered_map<std::string, rocprofiler_counter_id_t> counter_name_to_id_;
    std::unordered_map<uint64_t, std::string> counter_id_to_name_;

    // Currently configured counters
    std::vector<std::string> configured_counters_;
    
    // Cached counter configs for quick switching
    std::map<std::vector<std::string>, rocprofiler_compat::config_id_t> cached_profiles_;
    std::map<uint64_t, size_t> profile_sizes_;  // profile handle -> expected record count

    size_t expected_record_count_{0};
    uint64_t sample_counter_{0};
    std::string last_error_;
    bool context_initialized_{false};

    // Internal initialization
    void initialize_context();
    void populate_counter_map();
    void cleanup();

    // Counter info helpers
    size_t get_counter_instance_count(rocprofiler_counter_id_t counter_id) const;
    
    // Callback for setting profile during context start
    void set_profile_callback(rocprofiler_context_id_t ctx,
                              rocprofiler_compat::agent_set_profile_cb_t cb);

    // Decode counter name from record
    std::string decode_counter_name(const rocprofiler_compat::counter_record_t& rec) const;
};

}  // namespace measure_flops
