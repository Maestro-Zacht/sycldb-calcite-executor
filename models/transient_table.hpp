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
    uint64_t nrows;
    Column *group_by_column;
public:
    TransientTable(Table *base_table, sycl::queue &gpu_queue, sycl::queue &cpu_queue, memory_manager &gpu_allocator, memory_manager &cpu_allocator)
        : gpu_queue(gpu_queue), cpu_queue(cpu_queue), nrows(base_table->get_nrows()), group_by_column(nullptr)
    {
        flags_gpu = gpu_allocator.alloc<bool>(nrows);
        auto e1 = gpu_queue.fill<bool>(flags_gpu, true, nrows);

        flags_host = cpu_allocator.alloc<bool>(nrows);
        auto e2 = cpu_queue.fill<bool>(flags_host, true, nrows);

        const std::vector<Column> &base_columns = base_table->get_columns();

        current_columns.reserve(base_columns.size() + 100);
        for (const Column &col : base_columns)
            current_columns.push_back(const_cast<Column *>(&col));

        materialized_columns.reserve(50);

        e1.wait();
        e2.wait();
    }

    std::vector<Column *> get_columns() const { return current_columns; }
    uint64_t get_nrows() const { return nrows; }
    const Column *get_group_by_column() const { return group_by_column; }

    void set_group_by_column(uint64_t col) { group_by_column = current_columns[col]; }

    friend std::ostream &operator<<(std::ostream &out, const TransientTable &table)
    {
        for (uint64_t i = 0; i < table.nrows; i++)
        {
            if (table.flags_host[i]) // TODO ensure flags are sync'd
            {
                for (uint64_t j = 0; j < table.current_columns.size(); j++)
                {
                    const Column *col = table.current_columns[j];
                    out << (col->get_is_aggregate_result() ? col->get_aggregate_value(i) : col->operator[](i)) << ((j < table.current_columns.size() - 1) ? " " : "");
                }
                out << "\n";
            }
        }

        return out;
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
                    operand,
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

    std::vector<sycl::event> apply_project(
        const std::vector<ExprType> &exprs,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        const std::vector<sycl::event> &dependencies)
    {
        std::vector<sycl::event> events;
        std::vector<Column *> new_columns;
        new_columns.reserve(exprs.size() + 50);

        for (size_t i = 0; i < exprs.size(); i++)
        {
            const ExprType &expr = exprs[i];
            switch (expr.exprType)
            {
            case ExprOption::COLUMN:
                new_columns.push_back(current_columns[expr.input]);
                break;
            case ExprOption::LITERAL:
            {
                Column &new_col = materialized_columns.emplace_back(
                    nrows,
                    gpu_queue,
                    cpu_queue,
                    gpu_allocator,
                    cpu_allocator,
                    true,
                    false
                );

                auto fill_events = new_col.fill_with_literal((int)expr.literal.value);
                events.insert(events.end(), fill_events.begin(), fill_events.end());

                new_columns.push_back(&new_col);
                break;
            }
            case ExprOption::EXPR:
            {
                if (expr.operands.size() != 2)
                {
                    std::cerr << "Project operation: Unsupported number of operands for EXPR" << std::endl;
                    return {};
                }

                Column &new_col = materialized_columns.emplace_back(
                    nrows,
                    gpu_queue,
                    cpu_queue,
                    gpu_allocator,
                    cpu_allocator,
                    true,
                    false
                );

                if (expr.operands[0].exprType == ExprOption::COLUMN &&
                    expr.operands[1].exprType == ExprOption::COLUMN)
                {
                    const std::vector<Segment> &segments_a = current_columns[expr.operands[0].input]->get_segments();
                    const std::vector<Segment> &segments_b = current_columns[expr.operands[1].input]->get_segments();
                    std::vector<Segment> &segments_result = new_col.get_segments();

                    if (segments_a.size() != segments_b.size())
                    {
                        std::cerr << "Project operation: Mismatched segment sizes between columns" << std::endl;
                        return {};
                    }

                    for (size_t segment_number = 0; segment_number < segments_a.size(); segment_number++)
                    {
                        const Segment &segment_a = segments_a[segment_number];
                        const Segment &segment_b = segments_b[segment_number];
                        Segment &segment_result = segments_result[segment_number];

                        bool on_device = segment_a.is_on_device();
                        if (segment_b.is_on_device() != on_device || segment_result.is_on_device() != on_device)
                        {
                            std::cerr << "Project operation: Mismatched segment locations between columns" << std::endl;
                            return {};
                        }

                        events.push_back(
                            segment_result.perform_operator(
                                segment_a,
                                segment_b,
                                (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE,
                                expr.op,
                                dependencies
                            )
                        );
                    }
                }
                else if (expr.operands[0].exprType == ExprOption::LITERAL &&
                    expr.operands[1].exprType == ExprOption::COLUMN)
                {
                    const std::vector<Segment> &segments = current_columns[expr.operands[1].input]->get_segments();
                    std::vector<Segment> &segments_result = new_col.get_segments();

                    for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
                    {
                        const Segment &segment = segments[segment_number];
                        Segment &segment_result = segments_result[segment_number];

                        bool on_device = segment.is_on_device();

                        if (segment_result.is_on_device() != on_device)
                        {
                            std::cerr << "Project operation: Mismatched segment locations between columns" << std::endl;
                            return {};
                        }

                        events.push_back(
                            segment_result.perform_operator(
                                (int)expr.operands[0].literal.value,
                                segment,
                                (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE,
                                expr.op,
                                dependencies
                            )
                        );
                    }
                }
                else if (expr.operands[0].exprType == ExprOption::COLUMN &&
                    expr.operands[1].exprType == ExprOption::LITERAL)
                {
                    const std::vector<Segment> &segments = current_columns[expr.operands[0].input]->get_segments();
                    std::vector<Segment> &segments_result = new_col.get_segments();

                    for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
                    {
                        const Segment &segment = segments[segment_number];
                        Segment &segment_result = segments_result[segment_number];

                        bool on_device = segment.is_on_device();

                        if (segment_result.is_on_device() != on_device)
                        {
                            std::cerr << "Project operation: Mismatched segment locations between columns" << std::endl;
                            return {};
                        }

                        events.push_back(
                            segment_result.perform_operator(
                                segment,
                                (int)expr.operands[1].literal.value,
                                (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE,
                                expr.op,
                                dependencies
                            )
                        );
                    }
                }
                else
                {
                    std::cerr << "Project operation: Unsupported parsing ExprType "
                        << expr.operands[0].exprType << " and "
                        << expr.operands[1].exprType
                        << " for EXPR" << std::endl;
                    return {};
                }

                new_columns.push_back(&new_col);
                break;
            }
            }
        }

        current_columns = new_columns;
        return events;
    }

    std::vector<sycl::event> apply_aggregate(
        const AggType &agg,
        const std::vector<long> &group,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        const std::vector<sycl::event> &dependencies)
    {
        std::vector<sycl::event> events;

        if (group.size() == 0)
        {
            Column &result_column = materialized_columns.emplace_back(
                1,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator,
                true,
                true
            );

            const std::vector<Segment> &input_segments = current_columns[agg.operands[0]]->get_segments();

            uint64_t *temp_results = gpu_allocator.alloc<uint64_t>(input_segments.size());

            events.reserve(input_segments.size() + 1);

            for (int i = 0; i < input_segments.size(); i++)
            {
                const Segment &input_segment = input_segments[i];
                events.push_back(
                    input_segment.aggregate_operator(
                        temp_results + i,
                        (input_segment.is_on_device() ? flags_gpu : flags_host) + i * SEGMENT_SIZE,
                        agg.agg,
                        dependencies
                    )
                );
            }

            uint64_t *final_result = result_column.get_segments()[0].get_aggregate_data(true);

            auto e = gpu_queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(events);
                    cgh.parallel_for(
                        sycl::range<1>(input_segments.size()),
                        sycl::reduction(final_result, sycl::plus<>()),
                        [=](sycl::id<1> idx, auto &sum)
                        {
                            sum.combine(temp_results[idx]);
                        }
                    );
                }
            );

            events.clear();
            events.push_back(e);

            bool *new_gpu_flags = gpu_allocator.alloc<bool>(1),
                *new_cpu_flags = cpu_allocator.alloc<bool>(1);

            events.push_back(gpu_queue.fill<bool>(new_gpu_flags, true, 1));
            new_cpu_flags[0] = true;

            flags_gpu = new_gpu_flags;
            flags_host = new_cpu_flags;

            nrows = 1;

            current_columns.clear();
            current_columns.push_back(&result_column);
        }
        else
        {
            // TODO
            std::cerr << "Aggregate operation: GROUP BY not yet supported" << std::endl;
        }

        return events;
    }

    std::vector<sycl::event> apply_join(
        const TransientTable &right_table,
        const RelNode &rel,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        const std::vector<sycl::event> &dependencies)
    {
        std::vector<sycl::event> events;
        int left_column = rel.condition.operands[0].input,
            right_column = rel.condition.operands[1].input - current_columns.size();

        if (left_column < 0 ||
            left_column >= current_columns.size() ||
            right_column < 0 ||
            right_column >= right_table.current_columns.size())
        {
            std::cerr << "Join operation: Invalid column indices in join condition." << std::endl;
            return {};
        }

        if (rel.joinType == "semi")
        {
            auto ht_data = right_table.current_columns[right_column]->build_keys_hash_table(
                right_table.flags_gpu,
                gpu_allocator,
                cpu_allocator,
                dependencies
            );
            bool *ht = std::get<0>(ht_data);
            int build_min_value = std::get<1>(ht_data),
                build_max_value = std::get<2>(ht_data);
            std::vector<sycl::event> ht_events = std::get<3>(ht_data);

            events = current_columns[left_column]->semi_join(
                flags_gpu,
                build_min_value,
                build_max_value,
                ht,
                ht_events
            );

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);
        }
        else
        {
            auto ht_data = right_table.current_columns[right_column]->build_key_vals_hash_table(
                right_table.group_by_column,
                right_table.flags_gpu,
                gpu_allocator,
                cpu_allocator,
                dependencies
            );
            int *ht = std::get<0>(ht_data);
            int build_min_value = std::get<1>(ht_data),
                build_max_value = std::get<2>(ht_data);
            std::vector<sycl::event> ht_events = std::get<3>(ht_data);

            auto join_data = current_columns[left_column]->full_join_operation(
                flags_gpu,
                build_min_value,
                build_max_value,
                ht,
                gpu_allocator,
                cpu_allocator,
                gpu_queue,
                cpu_queue,
                ht_events
            );

            events = join_data.first;

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);

            materialized_columns.push_back(std::move(join_data.second));
            Column *new_col = &materialized_columns[materialized_columns.size() - 1];

            current_columns[rel.condition.operands[1].input] = new_col;
        }

        return events;
    }
};