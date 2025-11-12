#pragma once

#include <sycl/sycl.hpp>

#include "../common.hpp"

#include "models.hpp"
#include "execution.hpp"
#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/common.hpp"


uint64_t count_true_flags(
    const bool *flags,
    int len,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies = {})
{
    uint64_t *count = sycl::malloc_shared<uint64_t>(1, queue);
    *count = 0;

    queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                sycl::range<1>(len),
                sycl::reduction(count, sycl::plus<>()),
                [=](sycl::id<1> idx, auto &sum)
                {
                    if (flags[idx[0]])
                        sum.combine(1);
                }
            );
        }
    ).wait();
    uint64_t result = *count;
    sycl::free(count, queue);
    return result;
}

class TransientTable
{
private:
    bool *flags_host, *flags_gpu;
    sycl::queue gpu_queue, cpu_queue;
    #if USE_FUSION
    sycl::ext::codeplay::experimental::fusion_wrapper fw_gpu, fw_cpu;
    #endif
    std::vector<Column *> current_columns;
    std::vector<Column> materialized_columns;
    uint64_t nrows;
    Column *group_by_column;
    uint64_t group_by_column_index;
    std::vector<std::vector<KernelBundle>> pending_kernels;
    std::vector<sycl::event> pending_kernels_dependencies;

    std::vector<sycl::event> execute_pending_kernels()
    {
        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0 ? 1 : 0);
        std::vector<sycl::event> events;

        if (pending_kernels.size() == 0)
            return pending_kernels_dependencies;

        for (const auto &phases : pending_kernels)
        {
            if (phases.size() != segment_num)
            {
                std::cerr << "Pending kernels segment number mismatch: expected " << segment_num << ", got " << phases.size() << std::endl;
                throw std::runtime_error("Pending kernels segment number mismatch.");
            }
        }

        for (uint64_t segment_index = 0; segment_index < segment_num; segment_index++)
        {
            std::vector<sycl::event> deps = pending_kernels_dependencies;
            sycl::event e;

            #if USE_FUSION
            fw_gpu.start_fusion();
            #endif

            for (const auto &phases : pending_kernels)
            {
                const KernelBundle &bundle = phases[segment_index];
                e = bundle.execute(gpu_queue, deps);
                deps.clear();
                deps.push_back(e);
            }

            #if USE_FUSION
            fw_gpu.complete_fusion(sycl::ext::codeplay::experimental::property::no_barriers {});
            #endif

            events.push_back(e);
        }

        pending_kernels.clear();
        pending_kernels_dependencies.clear();

        return events;
    }
public:
    TransientTable(Table *base_table,
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue,
        #if USE_FUSION
        sycl::ext::codeplay::experimental::fusion_wrapper fw_gpu,
        sycl::ext::codeplay::experimental::fusion_wrapper fw_cpu,
        #endif
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator
    )
        : gpu_queue(gpu_queue),
        cpu_queue(cpu_queue),
        #if USE_FUSION
        fw_gpu(fw_gpu),
        fw_cpu(fw_cpu),
        #endif
        nrows(base_table->get_nrows()),
        group_by_column(nullptr),
        group_by_column_index(0)
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

    void set_group_by_column(uint64_t col)
    {
        group_by_column = current_columns[col];
        group_by_column_index = col;
    }

    void copy_flags_to_host()
    {
        gpu_queue.copy(flags_gpu, flags_host, nrows).wait();
    }

    uint64_t count_flags_true(std::vector<sycl::event> dependencies = {})
    {
        return count_true_flags(flags_gpu, nrows, gpu_queue, dependencies);
    }


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

    std::tuple<bool *, int, int> build_keys_hash_table(int column, memory_manager &gpu_allocator, memory_manager &cpu_allocator)
    {
        auto ht_res = current_columns[column]->build_keys_hash_table(
            flags_gpu,
            gpu_allocator,
            cpu_allocator
        );

        std::vector<KernelBundle> ht_kernels = std::get<3>(ht_res);
        pending_kernels.push_back(ht_kernels);

        return std::make_tuple(
            std::get<0>(ht_res),
            std::get<1>(ht_res),
            std::get<2>(ht_res)
        );
    }

    void apply_filter(
        const ExprType &expr,
        std::string parent_op,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator)
    {
        // Recursive parsing of EXPR types. LITERAL and COLUMN are handled in parent EXPR type.
        if (expr.exprType != ExprOption::EXPR)
        {
            std::cerr << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
            return;
        }

        std::vector<KernelBundle> ops;

        if (expr.op == "SEARCH")
        {
            int col_index = expr.operands[0].input;
            const std::vector<Segment> &segments = current_columns[col_index]->get_segments();

            ops.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                ops.push_back(
                    segment.search_operator(
                        expr,
                        parent_op,
                        gpu_allocator,
                        cpu_allocator,
                        flags_gpu + segment_number * SEGMENT_SIZE,
                        flags_host + segment_number * SEGMENT_SIZE
                    )
                );
            }

            pending_kernels.push_back(ops);
        }
        else if (is_filter_logical(expr.op))
        {
            // Logical operation between other expressions. Pass parent op to the first then use the current op.
            // TODO: check if passing parent logic is correct in general
            bool parent_op_used = false;
            for (const ExprType &operand : expr.operands)
            {
                apply_filter(
                    operand,
                    parent_op_used ? expr.op : parent_op,
                    gpu_allocator,
                    cpu_allocator
                );
                parent_op_used = true;
            }
        }
        else
        {
            if (expr.operands.size() != 2)
            {
                std::cerr << "Filter condition: Unsupported number of operands for EXPR" << std::endl;
                return;
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
                    return;
                }
            }

            const std::vector<Segment> &segments = cols[0]->get_segments();
            ops.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                KernelBundle bundle;

                bundle.add_kernel(
                    KernelData(
                        literal ? KernelType::SelectionKernelLiteral : KernelType::SelectionKernelColumns,
                        literal ?
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                literal_value,
                                flags_gpu + segment_number * SEGMENT_SIZE,
                                flags_host + segment_number * SEGMENT_SIZE
                            )
                            ) :
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                cols[1]->get_segments()[segment_number],
                                flags_gpu + segment_number * SEGMENT_SIZE,
                                flags_host + segment_number * SEGMENT_SIZE
                            )
                            )
                    )
                );
                ops.push_back(bundle);
            }

            pending_kernels.push_back(ops);
        }
    }

    void apply_project(
        const std::vector<ExprType> &exprs,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator)
    {
        std::vector<Column *> new_columns;
        new_columns.reserve(exprs.size() + 50);

        for (size_t i = 0; i < exprs.size(); i++)
        {
            const ExprType &expr = exprs[i];
            switch (expr.exprType)
            {
            case ExprOption::COLUMN:
            {
                new_columns.push_back(current_columns[expr.input]);
                if (expr.input == group_by_column_index)
                    group_by_column_index = i;
                break;
            }
            case ExprOption::LITERAL:
            {
                int literal_value = (int)expr.literal.value;
                Column &new_col = materialized_columns.emplace_back(
                    nrows,
                    gpu_queue,
                    cpu_queue,
                    gpu_allocator,
                    cpu_allocator,
                    true,
                    false
                );

                std::vector<KernelBundle> fill_bundles = new_col.fill_with_literal(literal_value);

                pending_kernels.push_back(fill_bundles);
                new_columns.push_back(&new_col);
                break;
            }
            case ExprOption::EXPR:
            {
                if (expr.operands.size() != 2)
                {
                    std::cerr << "Project operation: Unsupported number of operands for EXPR" << std::endl;
                    return;
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

                std::vector<KernelBundle> ops;

                std::vector<Segment> &segments_result = new_col.get_segments();
                ops.reserve(segments_result.size());

                if (expr.operands[0].exprType == ExprOption::COLUMN &&
                    expr.operands[1].exprType == ExprOption::COLUMN)
                {
                    const std::vector<Segment> &segments_a = current_columns[expr.operands[0].input]->get_segments();
                    const std::vector<Segment> &segments_b = current_columns[expr.operands[1].input]->get_segments();

                    if (segments_a.size() != segments_b.size())
                    {
                        std::cerr << "Project operation: Mismatched segment sizes between columns" << std::endl;
                        return;
                    }

                    for (size_t segment_number = 0; segment_number < segments_a.size(); segment_number++)
                    {
                        const Segment &segment_a = segments_a[segment_number];
                        const Segment &segment_b = segments_b[segment_number];
                        Segment &segment_result = segments_result[segment_number];
                        KernelBundle bundle;

                        bool on_device = segment_a.is_on_device();
                        if (segment_b.is_on_device() != on_device || segment_result.is_on_device() != on_device)
                        {
                            std::cerr << "Project operation: Mismatched segment locations between columns" << std::endl;
                            return;
                        }

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelColumns,
                                segment_result.perform_operator(
                                    segment_a,
                                    segment_b,
                                    (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE,
                                    expr.op
                                )
                            )
                        );
                        ops.push_back(bundle);
                    }
                }
                else if (expr.operands[0].exprType == ExprOption::LITERAL &&
                    expr.operands[1].exprType == ExprOption::COLUMN)
                {
                    const std::vector<Segment> &segments = current_columns[expr.operands[1].input]->get_segments();

                    for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
                    {
                        const Segment &segment = segments[segment_number];
                        Segment &segment_result = segments_result[segment_number];
                        KernelBundle bundle;

                        bool on_device = segment.is_on_device();

                        if (segment_result.is_on_device() != on_device)
                        {
                            std::cerr << "Project operation: Mismatched segment locations between columns" << std::endl;
                            return;
                        }

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralFirst,
                                segment_result.perform_operator(
                                    (int)expr.operands[0].literal.value,
                                    segment,
                                    (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE,
                                    expr.op
                                )
                            )
                        );
                        ops.push_back(bundle);
                    }
                }
                else if (expr.operands[0].exprType == ExprOption::COLUMN &&
                    expr.operands[1].exprType == ExprOption::LITERAL)
                {
                    const std::vector<Segment> &segments = current_columns[expr.operands[0].input]->get_segments();

                    for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
                    {
                        const Segment &segment = segments[segment_number];
                        Segment &segment_result = segments_result[segment_number];
                        KernelBundle bundle;

                        bool on_device = segment.is_on_device();

                        if (segment_result.is_on_device() != on_device)
                        {
                            std::cerr << "Project operation: Mismatched segment locations between columns" << std::endl;
                            return;
                        }

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralSecond,
                                segment_result.perform_operator(
                                    segment,
                                    (int)expr.operands[1].literal.value,
                                    (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE,
                                    expr.op
                                )
                            )
                        );
                        ops.push_back(bundle);
                    }
                }
                else
                {
                    std::cerr << "Project operation: Unsupported parsing ExprType "
                        << expr.operands[0].exprType << " and "
                        << expr.operands[1].exprType
                        << " for EXPR" << std::endl;
                    return;
                }

                pending_kernels.push_back(ops);
                new_columns.push_back(&new_col);
                break;
            }
            }
        }

        current_columns = new_columns;
    }

    std::vector<sycl::event> apply_aggregate(
        const AggType &agg,
        const std::vector<long> &group,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator)
    {
        std::vector<sycl::event> events;

        if (group.size() == 0)
        {
            std::vector<sycl::event> dependencies = execute_pending_kernels();

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

            uint64_t *final_result = result_column.get_segments()[0].get_aggregate_data(true);

            events.reserve(input_segments.size());

            auto agg_op = sycl::reduction(final_result, sycl::plus<>());

            for (int i = 0; i < input_segments.size(); i++)
            {
                const Segment &input_segment = input_segments[i];
                events.push_back(
                    input_segment.aggregate_operator(
                        (input_segment.is_on_device() ? flags_gpu : flags_host) + i * SEGMENT_SIZE,
                        agg_op,
                        dependencies
                    )
                );
            }

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
            std::vector<sycl::event> dependencies;
            uint64_t prod_ranges = 1;
            int *min = cpu_allocator.alloc<int>(group.size()),
                *max = cpu_allocator.alloc<int>(group.size());

            for (int i = 0; i < group.size(); i++)
            {
                auto min_max = current_columns[group[i]]->get_min_max();
                min[i] = min_max.first;
                max[i] = min_max.second;
                prod_ranges *= max[i] - min[i] + 1;
            }

            uint64_t *aggregate_result = gpu_allocator.alloc<uint64_t>(prod_ranges);
            bool *new_gpu_flags = gpu_allocator.alloc<bool>(prod_ranges),
                *new_cpu_flags = cpu_allocator.alloc<bool>(prod_ranges);
            unsigned *temp_flags = gpu_allocator.alloc<unsigned>(prod_ranges);

            int **results = cpu_allocator.alloc<int *>(group.size());
            const int **contents = cpu_allocator.alloc<const int *>(group.size());

            for (int i = 0; i < group.size(); i++)
                results[i] = gpu_allocator.alloc<int>(prod_ranges);

            int num_segments = current_columns[agg.operands[0]]->get_segments().size();

            events.reserve(num_segments);

            for (int i = 0; i < num_segments; i++)
            {
                for (int j = 0; j < group.size(); j++)
                {
                    const Segment &segment = current_columns[group[j]]->get_segments()[i];
                    contents[j] = segment.get_data(true);
                }
                const Segment &agg_segment = current_columns[agg.operands[0]]->get_segments()[i];

                auto e = group_by_aggregate(
                    contents,
                    agg_segment.get_data(true),
                    max,
                    min,
                    flags_gpu + i * SEGMENT_SIZE,
                    agg_segment.get_nrows(),
                    group.size(),
                    results,
                    aggregate_result,
                    temp_flags,
                    prod_ranges,
                    agg.agg,
                    gpu_queue,
                    dependencies
                );
                events.push_back(e);
                e.wait();
            }

            auto e1 = gpu_queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(events);
                    cgh.parallel_for(
                        prod_ranges,
                        [=](sycl::id<1> idx)
                        {
                            new_gpu_flags[idx] = temp_flags[idx] != 0;
                        }
                    );
                }
            );

            auto e2 = gpu_queue.memcpy(
                new_cpu_flags,
                new_gpu_flags,
                sizeof(bool) * prod_ranges,
                e1
            );

            events.clear();
            events.push_back(e2);

            current_columns.clear();

            for (int i = 0; i < group.size(); i++)
            {
                Column &new_col = materialized_columns.emplace_back(
                    results[i],
                    true,
                    gpu_queue,
                    cpu_queue,
                    gpu_allocator,
                    cpu_allocator,
                    prod_ranges
                );
                current_columns.push_back(&new_col);
            }

            Column &agg_col = materialized_columns.emplace_back(
                aggregate_result,
                true,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator,
                prod_ranges
            );
            current_columns.push_back(&agg_col);

            flags_gpu = new_gpu_flags;
            flags_host = new_cpu_flags;

            nrows = prod_ranges;
        }

        return events;
    }

    void apply_join(
        TransientTable &right_table,
        const RelNode &rel,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator)
    {
        int left_column = rel.condition.operands[0].input,
            right_column = rel.condition.operands[1].input - current_columns.size();

        if (left_column < 0 ||
            left_column >= current_columns.size() ||
            right_column < 0 ||
            right_column >= right_table.current_columns.size())
        {
            std::cerr << "Join operation: Invalid column indices in join condition: " << left_column << "/" << current_columns.size() << " and " << right_column << "/" << right_table.current_columns.size() << " ( " << rel.condition.operands[1].input << " )" << std::endl;
            throw std::invalid_argument("Invalid column indices in join condition.");
        }

        if (rel.joinType == "semi")
        {
            auto ht_data = right_table.build_keys_hash_table(
                right_column,
                gpu_allocator,
                cpu_allocator
            );
            std::vector<sycl::event> ht_events = right_table.execute_pending_kernels();

            pending_kernels_dependencies.insert(
                pending_kernels_dependencies.end(),
                ht_events.begin(),
                ht_events.end()
            );

            bool *ht = std::get<0>(ht_data);
            int build_min_value = std::get<1>(ht_data),
                build_max_value = std::get<2>(ht_data);

            pending_kernels.push_back(
                current_columns[left_column]->semi_join(
                    flags_gpu,
                    build_min_value,
                    build_max_value,
                    ht
                )
            );

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);
        }
        else
        {
            std::vector<sycl::event> dependencies;
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
            auto min_max_gb = right_table.group_by_column->get_min_max();

            auto join_data = current_columns[left_column]->full_join_operation(
                flags_gpu,
                build_min_value,
                build_max_value,
                min_max_gb.first,
                min_max_gb.second,
                ht,
                gpu_allocator,
                cpu_allocator,
                gpu_queue,
                cpu_queue,
                ht_events
            );

            // events = join_data.first;

            uint64_t group_by_col_index = current_columns.size() + right_table.group_by_column_index;

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);

            materialized_columns.push_back(std::move(join_data.second));

            current_columns[group_by_col_index] = &materialized_columns[materialized_columns.size() - 1];
        }
    }
};