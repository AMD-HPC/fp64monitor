/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE in the project root for license information.
 */

// Compatibility header for rocprofiler-sdk across ROCm versions
//
// This header provides compatibility between:
//   - ROCm 6.x (rocprofiler-sdk 0.x)
//   - ROCm 7.x (rocprofiler-sdk 1.x)
//
// API changes between versions:
//   ROCm 6:                                    ROCm 7:
//   rocprofiler_profile_config_id_t         -> rocprofiler_counter_config_id_t
//   rocprofiler_create_profile_config       -> rocprofiler_create_counter_config
//   rocprofiler_destroy_profile_config      -> rocprofiler_destroy_counter_config
//   rocprofiler_record_counter_t            -> rocprofiler_counter_record_t
//   rocprofiler_agent_set_profile_callback_t -> rocprofiler_device_counting_agent_cb_t
//   rocprofiler_device_counting_service_callback_t -> rocprofiler_device_counting_service_cb_t

#pragma once

#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/fwd.h>

// Detect ROCm version based on rocprofiler-sdk version
// ROCm 6.x uses rocprofiler-sdk 0.x
// ROCm 7.x uses rocprofiler-sdk 1.x
#if defined(ROCPROFILER_VERSION_MAJOR) && ROCPROFILER_VERSION_MAJOR >= 1
    #define ROCPROFILER_SDK_V1 1
#else
    #define ROCPROFILER_SDK_V0 1
#endif

// Include appropriate headers
#ifdef ROCPROFILER_SDK_V1
    // ROCm 7.x - use new API with deprecated wrappers for compatibility
    #include <rocprofiler-sdk/counter_config.h>
    #include <rocprofiler-sdk/deprecated/profile_config.h>
#else
    // ROCm 6.x - use original API
    #include <rocprofiler-sdk/profile_config.h>
#endif

#include <rocprofiler-sdk/device_counting_service.h>

// Unified type aliases that work across both versions
namespace rocprofiler_compat {

// Config ID type - use the old name which has typedefs in both versions
using config_id_t = rocprofiler_profile_config_id_t;

// Counter record type
#ifdef ROCPROFILER_SDK_V1
    using counter_record_t = rocprofiler_counter_record_t;
#else
    using counter_record_t = rocprofiler_record_counter_t;
#endif

// Callback types for device counting service
#ifdef ROCPROFILER_SDK_V1
    // ROCm 7 has the old typedef as deprecated alias
    using agent_set_profile_cb_t = rocprofiler_agent_set_profile_callback_t;
    using device_counting_service_cb_t = rocprofiler_device_counting_service_cb_t;
#else
    // ROCm 6 uses original names
    using agent_set_profile_cb_t = rocprofiler_agent_set_profile_callback_t;
    using device_counting_service_cb_t = rocprofiler_device_counting_service_callback_t;
#endif

// Wrapper functions for config creation/destruction
inline rocprofiler_status_t create_config(
    rocprofiler_agent_id_t agent_id,
    rocprofiler_counter_id_t* counters_list,
    size_t counters_count,
    config_id_t* config_id)
{
    return rocprofiler_create_profile_config(agent_id, counters_list, counters_count, config_id);
}

inline rocprofiler_status_t destroy_config(config_id_t config_id)
{
    return rocprofiler_destroy_profile_config(config_id);
}

} // namespace rocprofiler_compat
