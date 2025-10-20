#pragma once

#include <sycl/sycl.hpp>

#include "models.hpp"
#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"


class TransientTable
{
private:
    bool *flags_host, *flags_gpu;
    sycl::queue gpu_queue, cpu_queue;
    std::vector<Column *> current_columns;
    std::vector<Column> materialized_columns;
public:
    TransientTable(Table *base_table, sycl::queue &gpu_queue, sycl::queue &cpu_queue)
        : gpu_queue(gpu_queue), cpu_queue(cpu_queue)
    {
        uint64_t nrows = base_table->get_nrows();

        flags_gpu = sycl::malloc_device<bool>(nrows, gpu_queue);
        auto e1 = gpu_queue.fill<bool>(flags_gpu, true, nrows);

        flags_host = sycl::malloc_host<bool>(nrows, gpu_queue);
        auto e2 = cpu_queue.fill<bool>(flags_host, true, nrows);

        const std::vector<Column> &base_columns = base_table->get_columns();

        current_columns.reserve(base_columns.size());
        for (const Column &col : base_columns)
            current_columns.push_back(const_cast<Column *>(&col));

        e1.wait();
        e2.wait();
    }

    ~TransientTable()
    {
        sycl::free(flags_gpu, gpu_queue);
        sycl::free(flags_host, gpu_queue);
    }

    std::vector<sycl::event> apply_filter(
        const ExprType &expr,
        std::string parent_op,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        const std::vector<sycl::event> &dependencies)
    {
        // Recursive parsing of EXPR types. LITERAL and COLUMN are handled in parent EXPR type.
        if (expr.exprType != ExprOption::EXPR)
        {
            std::cerr << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
            return {};
        }

        std::vector<sycl::event> events;

        if (expr.op == "SEARCH")
        {
            int col_index = expr.operands[0].input;
            const std::vector<Segment> &segments = current_columns[col_index]->get_segments();

            events.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                events.push_back(
                    segment.search_operator(
                        expr,
                        parent_op,
                        gpu_allocator,
                        cpu_allocator,
                        flags_gpu + segment_number * SEGMENT_SIZE,
                        flags_host + segment_number * SEGMENT_SIZE,
                        dependencies
                    )
                );
            }
        }
        else if (is_filter_logical(expr.op))
        {
            // Logical operation between other expressions. Pass parent op to the first then use the current op.
            // TODO: check if passing parent logic is correct in general
            bool parent_op_used = false;
            std::vector<sycl::event> child_deps(dependencies);
            for (const ExprType &operand : expr.operands)
            {
                child_deps = apply_filter(
                    expr,
                    parent_op_used ? expr.op : parent_op,
                    gpu_allocator,
                    cpu_allocator,
                    child_deps
                );
                parent_op_used = true;
            }
            events.insert(events.end(), child_deps.begin(), child_deps.end());
        }
        else
        {
            if (expr.operands.size() != 2)
            {
                std::cerr << "Filter condition: Unsupported number of operands for EXPR" << std::endl;
                return {};
            }

            Column *cols[2];
            bool literal = false;
            int literal_value;

            for (int i = 0; i < 2; i++)
            {
                switch (expr.operands[i].exprType)
                {
                case ExprOption::COLUMN:
                    cols[i] = current_columns[expr.operands[i].input];
                    break;
                case ExprOption::LITERAL:
                    literal = true;
                    literal_value = expr.operands[i].literal.value;
                    break;
                default:
                    std::cerr << "Filter condition: Unsupported parsing ExprType "
                        << expr.operands[i].exprType
                        << " for comparison operand"
                        << std::endl;
                    return {};
                }
            }

            const std::vector<Segment> &segments = cols[0]->get_segments();
            events.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                events.push_back(
                    literal ?
                    segment.filter_operator(
                        expr.op,
                        parent_op,
                        literal_value,
                        flags_gpu + segment_number * SEGMENT_SIZE,
                        flags_host + segment_number * SEGMENT_SIZE,
                        dependencies
                    ) :
                    segment.filter_operator(
                        expr.op,
                        parent_op,
                        cols[1]->get_segments()[segment_number],
                        flags_gpu + segment_number * SEGMENT_SIZE,
                        flags_host + segment_number * SEGMENT_SIZE,
                        dependencies
                    )
                );
            }
        }

        return events;
    }
};