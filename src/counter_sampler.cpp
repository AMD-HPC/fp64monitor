/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE in the project root for license information.
 */

#include "measure_flops/counter_sampler.hpp"
#include "measure_flops/rocprofiler_compat.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace measure_flops {

namespace {

// Helper macro for checking rocprofiler status
#define ROCP_CHECK(result, msg)                                                 \
    do {                                                                        \
        rocprofiler_status_t status_ = (result);                                \
        if (status_ != ROCPROFILER_STATUS_SUCCESS) {                            \
            std::ostringstream oss;                                             \
            oss << msg << ": " << rocprofiler_get_status_string(status_);       \
            throw std::runtime_error(oss.str());                                \
        }                                                                       \
    } while (0)

}  // namespace

// Static method to get available GPUs - this works without full initialization
std::vector<GpuAgent> CounterSampler::get_available_gpus() {
    std::vector<GpuAgent> agents;

    auto iterate_cb = [](rocprofiler_agent_version_t version,
                         const void** agents_arr,
                         size_t num_agents,
                         void* user_data) -> rocprofiler_status_t {
        if (version != ROCPROFILER_AGENT_INFO_VERSION_0) {
            return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
        }

        auto* result = static_cast<std::vector<GpuAgent>*>(user_data);
        int logical_id = 0;

        for (size_t i = 0; i < num_agents; ++i) {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            
            // Only include GPU agents
            if (agent->type == ROCPROFILER_AGENT_TYPE_GPU) {
                GpuAgent gpu;
                gpu.id = agent->id;
                gpu.name = agent->name ? agent->name : "Unknown";
                
                // Convert gfx_target_version (uint32_t) to string like "gfx942"
                uint32_t ver = agent->gfx_target_version;
                if (ver > 0) {
                    uint32_t major = (ver / 10000) % 100;
                    uint32_t minor = (ver / 100) % 100;
                    uint32_t patch = ver % 100;
                    std::ostringstream oss;
                    oss << "gfx" << major << minor << patch;
                    gpu.gfx_version = oss.str();
                } else {
                    gpu.gfx_version = "";
                }
                
                gpu.logical_id = logical_id++;
                gpu.compute_units = agent->cu_count;
                gpu.max_clock_freq = agent->max_engine_clk_fcompute;
                result->push_back(gpu);
            }
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    rocprofiler_status_t status = rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        iterate_cb,
        sizeof(rocprofiler_agent_v0_t),
        static_cast<void*>(&agents));

    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "Warning: Failed to query available agents: "
                  << rocprofiler_get_status_string(status) << std::endl;
    }

    return agents;
}

CounterSampler::CounterSampler(const GpuAgent& agent)
    : agent_(agent) {
    // Populate counter map - this can be done before full initialization
    populate_counter_map();
    // Context initialization will be done lazily when needed
    context_initialized_ = false;
}

CounterSampler::~CounterSampler() {
    cleanup();
}

CounterSampler::CounterSampler(CounterSampler&& other) noexcept
    : agent_(std::move(other.agent_)),
      context_(other.context_),
      buffer_(other.buffer_),
      profile_(other.profile_),
      counter_name_to_id_(std::move(other.counter_name_to_id_)),
      counter_id_to_name_(std::move(other.counter_id_to_name_)),
      configured_counters_(std::move(other.configured_counters_)),
      cached_profiles_(std::move(other.cached_profiles_)),
      profile_sizes_(std::move(other.profile_sizes_)),
      expected_record_count_(other.expected_record_count_),
      sample_counter_(other.sample_counter_),
      last_error_(std::move(other.last_error_)),
      context_initialized_(other.context_initialized_) {
    other.context_initialized_ = false;
    other.context_ = {};
    other.buffer_ = {};
    other.profile_.handle = 0;
}

CounterSampler& CounterSampler::operator=(CounterSampler&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        agent_ = std::move(other.agent_);
        context_ = other.context_;
        buffer_ = other.buffer_;
        profile_ = other.profile_;
        counter_name_to_id_ = std::move(other.counter_name_to_id_);
        counter_id_to_name_ = std::move(other.counter_id_to_name_);
        configured_counters_ = std::move(other.configured_counters_);
        cached_profiles_ = std::move(other.cached_profiles_);
        profile_sizes_ = std::move(other.profile_sizes_);
        expected_record_count_ = other.expected_record_count_;
        sample_counter_ = other.sample_counter_;
        last_error_ = std::move(other.last_error_);
        context_initialized_ = other.context_initialized_;

        other.context_initialized_ = false;
        other.context_ = {};
        other.buffer_ = {};
        other.profile_.handle = 0;
    }
    return *this;
}

void CounterSampler::cleanup() {
    if (context_initialized_) {
        rocprofiler_stop_context(context_);
        
        for (auto& kv : cached_profiles_) {
            rocprofiler_compat::destroy_config(kv.second);
        }
        cached_profiles_.clear();
        
        context_initialized_ = false;
    }
}

void CounterSampler::populate_counter_map() {
    counter_name_to_id_.clear();
    counter_id_to_name_.clear();

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

    rocprofiler_status_t status = rocprofiler_iterate_agent_supported_counters(
        agent_.id, counter_cb, static_cast<void*>(&counter_ids));

    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::cerr << "Warning: Failed to iterate counters: " 
                  << rocprofiler_get_status_string(status) << std::endl;
        return;
    }

    for (const auto& counter_id : counter_ids) {
        rocprofiler_counter_info_v0_t info;
        rocprofiler_status_t qstatus = rocprofiler_query_counter_info(
            counter_id,
            ROCPROFILER_COUNTER_INFO_VERSION_0,
            static_cast<void*>(&info));

        if (qstatus == ROCPROFILER_STATUS_SUCCESS && info.name) {
            counter_name_to_id_[info.name] = counter_id;
            counter_id_to_name_[counter_id.handle] = info.name;
        }
    }
}

void CounterSampler::initialize_context() {
    if (context_initialized_) {
        return;
    }

    // Create context
    ROCP_CHECK(rocprofiler_create_context(&context_), "Failed to create context");

    // Create buffer
    auto buffer_cb = [](rocprofiler_context_id_t,
                        rocprofiler_buffer_id_t,
                        rocprofiler_record_header_t**,
                        size_t,
                        void*,
                        uint64_t) {};

    ROCP_CHECK(
        rocprofiler_create_buffer(
            context_,
            4096,
            2048,
            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
            buffer_cb,
            nullptr,
            &buffer_),
        "Failed to create buffer");

    // Create callback thread
    rocprofiler_callback_thread_t client_thread{};
    ROCP_CHECK(rocprofiler_create_callback_thread(&client_thread),
               "Failed to create callback thread");
    ROCP_CHECK(rocprofiler_assign_callback_thread(buffer_, client_thread),
               "Failed to assign callback thread");

    // Configure device counting service
    auto device_cb = [](rocprofiler_context_id_t ctx_id,
                        rocprofiler_agent_id_t,
                        rocprofiler_compat::agent_set_profile_cb_t set_config,
                        void* user_data) {
        auto* sampler = static_cast<CounterSampler*>(user_data);
        sampler->set_profile_callback(ctx_id, set_config);
    };

    ROCP_CHECK(
        rocprofiler_configure_device_counting_service(
            context_, buffer_, agent_.id, device_cb, this),
        "Failed to configure device counting service");

    context_initialized_ = true;
}

void CounterSampler::set_profile_callback(
    rocprofiler_context_id_t ctx,
    rocprofiler_compat::agent_set_profile_cb_t cb) {
    if (profile_.handle != 0) {
        cb(ctx, profile_);
    }
}

std::vector<CounterInfo> CounterSampler::get_available_counters() const {
    std::vector<CounterInfo> result;
    result.reserve(counter_name_to_id_.size());

    for (const auto& kv : counter_name_to_id_) {
        rocprofiler_counter_info_v0_t info;
        rocprofiler_status_t status = rocprofiler_query_counter_info(
            kv.second,
            ROCPROFILER_COUNTER_INFO_VERSION_0,
            static_cast<void*>(&info));

        if (status == ROCPROFILER_STATUS_SUCCESS) {
            CounterInfo ci;
            ci.id = kv.second;
            ci.name = info.name ? info.name : "";
            ci.description = info.description ? info.description : "";
            ci.block = info.block ? info.block : "";
            ci.is_derived = info.is_derived;
            ci.is_constant = info.is_constant;
            result.push_back(ci);
        }
    }

    std::sort(result.begin(), result.end(),
              [](const CounterInfo& a, const CounterInfo& b) {
                  return a.name < b.name;
              });

    return result;
}

std::vector<std::string> CounterSampler::get_available_counter_names() const {
    std::vector<std::string> names;
    names.reserve(counter_name_to_id_.size());
    
    for (const auto& kv : counter_name_to_id_) {
        names.push_back(kv.first);
    }
    
    std::sort(names.begin(), names.end());
    return names;
}

bool CounterSampler::is_counter_available(const std::string& name) const {
    return counter_name_to_id_.find(name) != counter_name_to_id_.end();
}

size_t CounterSampler::get_counter_instance_count(
    rocprofiler_counter_id_t counter_id) const {
#ifdef ROCPROFILER_SDK_V1
    // ROCm 7.x - use v1 counter info structure
    rocprofiler_counter_info_v1_t info;
    info.size = sizeof(info);
    
    rocprofiler_status_t status = rocprofiler_query_counter_info(
        counter_id,
        ROCPROFILER_COUNTER_INFO_VERSION_1,
        static_cast<void*>(&info));

    if (status == ROCPROFILER_STATUS_SUCCESS) {
        return info.dimensions_instances_count;
    }
#else
    // ROCm 6.x - use deprecated query function
    size_t instance_count = 1;
    rocprofiler_status_t status = rocprofiler_query_counter_instance_count(
        agent_.id, counter_id, &instance_count);
    
    if (status == ROCPROFILER_STATUS_SUCCESS) {
        return instance_count;
    }
#endif
    
    return 1;
}

bool CounterSampler::configure_counters(const std::vector<std::string>& counter_names) {
    if (counter_names.empty()) {
        last_error_ = "No counters specified";
        return false;
    }

    // Check if we have a cached profile for this counter set
    auto cached = cached_profiles_.find(counter_names);
    if (cached != cached_profiles_.end()) {
        profile_ = cached->second;
        expected_record_count_ = profile_sizes_[profile_.handle];
        configured_counters_ = counter_names;
        return true;
    }

    // Build list of counter IDs
    std::vector<rocprofiler_counter_id_t> counter_ids;
    size_t total_instances = 0;

    for (const auto& name : counter_names) {
        auto it = counter_name_to_id_.find(name);
        if (it == counter_name_to_id_.end()) {
            last_error_ = "Counter not available: " + name;
            return false;
        }
        counter_ids.push_back(it->second);
        total_instances += get_counter_instance_count(it->second);
    }

    // Create new profile
    rocprofiler_compat::config_id_t new_profile{};
    rocprofiler_status_t status = rocprofiler_compat::create_config(
        agent_.id,
        counter_ids.data(),
        counter_ids.size(),
        &new_profile);

    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "Failed to create counter config: "
            << rocprofiler_get_status_string(status);
        last_error_ = oss.str();
        return false;
    }

    // Cache the profile
    cached_profiles_[counter_names] = new_profile;
    profile_sizes_[new_profile.handle] = total_instances;

    profile_ = new_profile;
    expected_record_count_ = total_instances;
    configured_counters_ = counter_names;

    return true;
}

std::string CounterSampler::decode_counter_name(
    const rocprofiler_compat::counter_record_t& rec) const {
    rocprofiler_counter_id_t counter_id{};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);

    auto it = counter_id_to_name_.find(counter_id.handle);
    if (it != counter_id_to_name_.end()) {
        return it->second;
    }
    return "UNKNOWN";
}

Sample CounterSampler::take_sample() {
    Sample sample;
    sample.timestamp = std::chrono::steady_clock::now();
    sample.sample_id = ++sample_counter_;
    sample.gpu_id = agent_.logical_id;

    // Try to initialize context lazily
    if (!context_initialized_) {
        try {
            initialize_context();
        } catch (const std::exception& e) {
            last_error_ = std::string("Context init failed: ") + e.what();
            return sample;
        }
    }

    if (configured_counters_.empty() || profile_.handle == 0) {
        last_error_ = "No counters configured";
        return sample;
    }

    // Start context
    rocprofiler_status_t status = rocprofiler_start_context(context_);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        if (status == ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED) {
            last_error_ = "HSA runtime not loaded yet";
            return sample;
        }
        last_error_ = std::string("Failed to start context: ") + 
                      rocprofiler_get_status_string(status);
        return sample;
    }

    // Small delay to allow counters to accumulate
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // Read counter values
    std::vector<rocprofiler_compat::counter_record_t> records(expected_record_count_);
    size_t record_count = records.size();

    rocprofiler_user_data_t user_data;
    user_data.value = sample.sample_id;

    status = rocprofiler_sample_device_counting_service(
        context_,
        user_data,
        ROCPROFILER_COUNTER_FLAG_NONE,
        records.data(),
        &record_count);

    // Stop context
    rocprofiler_stop_context(context_);

    if (status != ROCPROFILER_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "Failed to sample counters: " << rocprofiler_get_status_string(status);
        last_error_ = oss.str();
        return sample;
    }

    // Convert records to CounterValues
    records.resize(record_count);
    sample.counters.reserve(record_count);

    for (const auto& rec : records) {
        CounterValue cv;
        cv.name = decode_counter_name(rec);
        cv.value = rec.counter_value;
        cv.instance_id = rec.id;
        
        rocprofiler_counter_id_t counter_id{};
        rocprofiler_query_record_counter_id(rec.id, &counter_id);
        cv.counter_id = counter_id.handle;

        sample.counters.push_back(cv);
    }

    last_error_.clear();
    return sample;
}

}  // namespace measure_flops
