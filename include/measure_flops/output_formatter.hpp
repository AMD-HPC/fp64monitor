/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE in the project root for license information.
 */

#pragma once

#include "measure_flops/instruction_analyzer.hpp"
#include "measure_flops/types.hpp"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace measure_flops {

class OutputFormatter {
public:
    virtual ~OutputFormatter() = default;

    // Write header (called once at start)
    virtual void write_header(std::ostream& os, const MonitorConfig& config) = 0;

    // Write a single instruction mix sample
    virtual void write_sample(std::ostream& os, const InstructionMix& mix,
                              double elapsed_seconds,
                              const std::string& counter_set = "") = 0;

    // Write footer/summary (called once at end)
    virtual void write_footer(std::ostream& os,
                              const InstructionAnalyzer::Summary& summary) = 0;

    // Factory method to create formatter by name
    static std::unique_ptr<OutputFormatter> create(const std::string& format);
};

class TextFormatter : public OutputFormatter {
public:
    void write_header(std::ostream& os, const MonitorConfig& config) override;
    void write_sample(std::ostream& os, const InstructionMix& mix,
                      double elapsed_seconds,
                      const std::string& counter_set = "") override;
    void write_footer(std::ostream& os,
                      const InstructionAnalyzer::Summary& summary) override;

private:
    bool header_written_ = false;
};

class CsvFormatter : public OutputFormatter {
public:
    void write_header(std::ostream& os, const MonitorConfig& config) override;
    void write_sample(std::ostream& os, const InstructionMix& mix,
                      double elapsed_seconds,
                      const std::string& counter_set = "") override;
    void write_footer(std::ostream& os,
                      const InstructionAnalyzer::Summary& summary) override;
};

class JsonFormatter : public OutputFormatter {
public:
    void write_header(std::ostream& os, const MonitorConfig& config) override;
    void write_sample(std::ostream& os, const InstructionMix& mix,
                      double elapsed_seconds,
                      const std::string& counter_set = "") override;
    void write_footer(std::ostream& os,
                      const InstructionAnalyzer::Summary& summary) override;

private:
    bool first_sample_ = true;
};

}  // namespace measure_flops
