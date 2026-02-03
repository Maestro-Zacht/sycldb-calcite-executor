#pragma once

#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include "../common.hpp"

#include "models.hpp"
#include "execution.hpp"
#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/common.hpp"

class TransientTable
{
private:
    bool *flags_host;
    std::vector<bool *> flags_devices;
    std::vector<bool> flags_modified_host;
    std::vector<std::vector<bool>> flags_modified_devices;
    sycl::queue &cpu_queue;
    std::vector<sycl::queue> &device_queues;
    #if USE_FUSION
    // sycl::ext::codeplay::experimental::fusion_wrapper &fw_cpu;
    std::vector<sycl::ext::codeplay::experimental::fusion_wrapper> &fw_devices;
    #endif
    std::vector<Column *> current_columns;
    std::vector<Column> materialized_columns;
    uint64_t nrows;
    Column *group_by_column;
    uint64_t group_by_column_index;
    std::vector<std::vector<KernelBundle>> pending_kernels;
    std::vector<sycl::event> pending_kernels_dependencies_cpu;
    std::vector<std::vector<sycl::event>> pending_kernels_dependencies_devices;
public:
    TransientTable(Table *base_table,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        #if USE_FUSION
        sycl::ext::codeplay::experimental::fusion_wrapper &fw_cpu,
        std::vector<sycl::ext::codeplay::experimental::fusion_wrapper> &fw_devices,
        #endif
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators
    )
        :
        flags_modified_devices(device_queues.size()),
        cpu_queue(cpu_queue),
        device_queues(device_queues),
        #if USE_FUSION
        // fw_cpu(fw_cpu),
        fw_devices(fw_devices),
        #endif
        nrows(base_table->get_nrows()),
        group_by_column(nullptr),
        group_by_column_index(0),
        pending_kernels_dependencies_devices(device_queues.size())
    {
        // std::cout << "Creating transient table with " << nrows << " rows." << std::endl;

        flags_host = cpu_allocator.alloc<bool>(nrows, true);
        auto e1 = cpu_queue.fill<bool>(flags_host, true, nrows);

        std::vector<sycl::event> gpu_events;
        gpu_events.reserve(device_queues.size());
        flags_devices.reserve(device_queues.size());
        for (int d = 0; d < device_queues.size(); d++)
        {
            flags_devices.push_back(device_allocators[d].alloc<bool>(nrows, true));
            gpu_events.push_back(
                device_queues[d].fill<bool>(flags_devices[d], true, nrows)
            );
        }

        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);

        flags_modified_host.resize(segment_num, false);
        for (int d = 0; d < device_queues.size(); d++)
            flags_modified_devices[d].resize(segment_num, false);

        const std::vector<Column> &base_columns = base_table->get_columns();

        current_columns.reserve(base_columns.size() + 100);
        for (const Column &col : base_columns)
            current_columns.push_back(const_cast<Column *>(&col));

        materialized_columns.reserve(50);

        sycl::event::wait(gpu_events);
        e1.wait();
        // std::cout << "Transient table created." << std::endl;
    }

    std::vector<Column *> get_columns() const { return current_columns; }
    uint64_t get_nrows() const { return nrows; }
    const Column *get_group_by_column() const { return group_by_column; }

    void set_group_by_column(uint64_t col)
    {
        group_by_column = current_columns[col];
        group_by_column_index = col;
    }

    friend std::ostream &operator<<(std::ostream &out, const TransientTable &table)
    {
        for (uint64_t i = 0; i < table.nrows; i++)
        {
            if (table.flags_host[i])
            {
                for (uint64_t j = 0; j < table.current_columns.size(); j++)
                {
                    const Column *col = table.current_columns[j];
                    if (col != nullptr && col->get_segments().size() > 0)
                        out << (col->get_is_aggregate_result() ? col->get_aggregate_value(i) : col->operator[](i)) << ((j < table.current_columns.size() - 1) ? " " : "");
                }
                out << "\n";
            }
        }

        return out;
    }

    std::pair<std::vector<sycl::event>, std::vector<std::vector<sycl::event>>> execute_pending_kernels(
        #if USE_FUSION
        bool fuse = true
        #endif
    )
    {
        // std::cout << "start execute" << std::endl;
        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
        std::vector<sycl::event> events_cpu;
        std::vector<std::vector<sycl::event>> events_devices(device_queues.size());
        bool executed_cpu = false;
        std::vector<bool> executed_devices(device_queues.size(), false);

        if (pending_kernels.size() == 0)
        {
            events_cpu = pending_kernels_dependencies_cpu;
            for (int d = 0; d < device_queues.size(); d++)
            {
                events_devices[d] = pending_kernels_dependencies_devices[d];
                pending_kernels_dependencies_devices[d].clear();
            }
            pending_kernels_dependencies_cpu.clear();

            // std::cout << "No pending kernels to execute." << std::endl;

            return { events_cpu, events_devices };
        }

        for (const auto &phases : pending_kernels)
        {
            if (phases.size() != segment_num)
            {
                std::cerr << "Pending kernels segment number mismatch: expected " << segment_num << ", got " << phases.size() << std::endl;
                throw std::runtime_error("Pending kernels segment number mismatch.");
            }
        }

        for (int d = 0; d < device_queues.size(); d++)
            events_devices[d].reserve(segment_num);
        events_cpu.reserve(segment_num);

        for (uint64_t segment_index = 0; segment_index < segment_num; segment_index++)
        {
            std::vector<sycl::event> deps_cpu, tmp;
            std::vector<std::vector<sycl::event>> deps_devices(device_queues.size());

            for (int d = 0; d < device_queues.size(); d++)
            {
                if (pending_kernels_dependencies_devices[d].size() == segment_num)
                    deps_devices[d].push_back(pending_kernels_dependencies_devices[d][segment_index]);
                else
                    deps_devices[d] = pending_kernels_dependencies_devices[d];
            }

            if (pending_kernels_dependencies_cpu.size() == segment_num)
                deps_cpu.push_back(pending_kernels_dependencies_cpu[segment_index]);
            else
                deps_cpu = pending_kernels_dependencies_cpu;


            for (int d = -1; d < (int)device_queues.size(); d++)
            {
                bool kernel_present = false;

                #if USE_FUSION
                if (d >= 0 && fuse)
                {
                    fw_devices[d].start_fusion();
                }
                #endif

                for (const auto &phases : pending_kernels)
                {
                    const KernelBundle &bundle = phases[segment_index];
                    bool on_device = bundle.is_on_device();
                    int device_index = bundle.get_device_index();

                    if ((d == -1 && !on_device) || (d >= 0 && d == device_index))
                    {
                        tmp = bundle.execute(
                            cpu_queue,
                            device_queues,
                            deps_cpu,
                            deps_devices
                        );

                        if (on_device)
                            deps_devices[d] = std::move(tmp);
                        else
                            deps_cpu = std::move(tmp);

                        kernel_present = true;
                    }
                }

                #if USE_FUSION
                if (d == -1 && kernel_present)
                {
                    events_cpu.insert(
                        events_cpu.end(),
                        deps_cpu.begin(),
                        deps_cpu.end()
                    );
                    executed_cpu = true;
                }
                else if (d >= 0)
                {
                    if (!kernel_present)
                    {
                        fw_devices[d].cancel_fusion();
                    }
                    else if (fuse)
                    {
                        events_devices[d].push_back(
                            fw_devices[d].complete_fusion(sycl::ext::codeplay::experimental::property::no_barriers {})
                        );
                        executed_devices[d] = true;
                    }
                    else
                    {
                        events_devices[d].insert(
                            events_devices[d].end(),
                            deps_devices[d].begin(),
                            deps_devices[d].end()
                        );
                        executed_devices[d] = true;
                    }
                }
                #else
                if (kernel_present)
                {
                    if (d >= 0)
                    {
                        events_devices[d].insert(
                            events_devices[d].end(),
                            deps_devices[d].begin(),
                            deps_devices[d].end()
                        );
                        executed_devices[d] = true;
                    }
                    else
                    {
                        events_cpu.insert(
                            events_cpu.end(),
                            deps_cpu.begin(),
                            deps_cpu.end()
                        );
                        executed_cpu = true;
                    }
                }
                #endif
            }
        }

        // std::cout << "All segments executed." << std::endl;

        for (int d = 0; d < device_queues.size(); d++)
        {
            if (!executed_devices[d])
                events_devices[d] = pending_kernels_dependencies_devices[d];

            pending_kernels_dependencies_devices[d].clear();
        }

        if (!executed_cpu)
            events_cpu = pending_kernels_dependencies_cpu;

        pending_kernels.clear();
        pending_kernels_dependencies_cpu.clear();

        // std::cout << "end execute" << std::endl;

        return { events_cpu, events_devices };
    }

    void assert_flags_to_cpu()
    {
        for (auto &flags_modified_gpu : flags_modified_devices)
        {
            for (bool modified : flags_modified_gpu)
            {
                if (modified)
                {
                    std::cerr << "Flags on GPU modified but expected to be on CPU." << std::endl;
                    throw std::runtime_error("Flags on GPU modified but expected to be on CPU.");
                }
            }
        }
    }

    uint64_t count_flags_true(bool on_device, int device_index)
    {
        auto dependencies = execute_pending_kernels();

        return count_true_flags(
            on_device ? flags_devices[device_index] : flags_host,
            nrows,
            on_device ? device_queues[device_index] : cpu_queue,
            on_device ? dependencies.second[device_index] : dependencies.first
        );
    }

    // This function is a sync point due to oneDPL algorithms and needs dependencies to be waited manually before calling it
    std::tuple<int *, uint64_t> build_row_ids(int segment_n, int segment_size, memory_manager &gpu_allocator, int device_index)
    {
        bool *flags = flags_devices[device_index] + segment_n * SEGMENT_SIZE;
        int *row_ids_gpu = gpu_allocator.alloc<int>(segment_size, true);

        auto policy = oneapi::dpl::execution::make_device_policy(device_queues[device_index]);
        auto start_index = oneapi::dpl::counting_iterator<int>(0);
        auto end_index = oneapi::dpl::counting_iterator<int>(segment_size);

        auto end_ptr = oneapi::dpl::copy_if(
            policy,
            start_index,
            end_index,
            row_ids_gpu,
            [=](int i)
            {
                return flags[i];
            }
        );

        uint64_t n_selected = end_ptr - row_ids_gpu;

        return { row_ids_gpu, n_selected };
    }

    // This is a sync point
    void compress_and_sync(memory_manager &cpu_allocator, memory_manager &device_allocator, int device_index, const std::vector<Column *> &columns_to_sync = {})
    {
        if (device_index < 0 || device_index >= device_queues.size())
        {
            std::cerr << "compress_and_sync: invalid device index " << device_index << std::endl;
            throw std::runtime_error("compress_and_sync: invalid device index.");
        }

        int num_segments = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
        std::vector<uint64_t> n_rows_new(num_segments);
        execute_pending_kernels();

        device_queues[device_index].wait();

        for (int i = 0; i < num_segments; i++)
        {
            int *row_ids_gpu = nullptr, *row_ids_host = nullptr;
            uint64_t segment_size = (i == num_segments - 1) ? (nrows - i * SEGMENT_SIZE) : SEGMENT_SIZE;
            sycl::event e_row_ids_host;

            if (flags_modified_devices[device_index][i])
            {
                auto row_id_res = build_row_ids(i, segment_size, device_allocator, device_index);
                row_ids_gpu = std::get<0>(row_id_res);
                n_rows_new[i] = std::get<1>(row_id_res);
                row_ids_host = device_allocator.alloc<int>(n_rows_new[i], false);
                e_row_ids_host = device_queues[device_index].memcpy(
                    row_ids_host,
                    row_ids_gpu,
                    n_rows_new[i] * sizeof(int)
                );
            }

            for (const Column *col : columns_to_sync)
            {
                Segment &seg = const_cast<Segment &>(col->get_segments()[i]);
                if (seg.is_on_device(device_index) && seg.needs_copy_on(false, -1))
                {
                    if (row_ids_gpu == nullptr)
                    {
                        auto row_id_res = build_row_ids(i, segment_size, device_allocator, device_index);
                        row_ids_gpu = std::get<0>(row_id_res);
                        n_rows_new[i] = std::get<1>(row_id_res);
                        row_ids_host = device_allocator.alloc<int>(n_rows_new[i], false);
                        e_row_ids_host = device_queues[device_index].memcpy(
                            row_ids_host,
                            row_ids_gpu,
                            n_rows_new[i] * sizeof(int)
                        );
                    }

                    seg.compress_sync(
                        row_ids_gpu,
                        row_ids_host,
                        e_row_ids_host,
                        n_rows_new[i],
                        device_allocator,
                        device_index
                    ).wait(); // TODO: need to find why sometimes segfault if not wait here
                }
            }

            if (flags_modified_devices[device_index][i])
            {
                bool *flags = flags_host + i * SEGMENT_SIZE;
                cpu_queue.submit(
                    [&](sycl::handler &cgh)
                    {
                        cgh.depends_on(e_row_ids_host);
                        cgh.parallel_for(
                            n_rows_new[i] - 1,
                            [=](sycl::id<1> idx)
                            {
                                auto i = idx[0];
                                int row_id = row_ids_host[i],
                                    next_row_id = row_ids_host[i + 1];

                                for (int r = row_id + 1; r < next_row_id; r++)
                                    flags[r] = false;
                            }
                        );
                    }
                );

                e_row_ids_host.wait();

                int first_row_id = row_ids_host[0];
                if (first_row_id > 0)
                {
                    cpu_queue.memset(
                        flags,
                        0,
                        first_row_id * sizeof(bool)
                    );
                }

                int last_row_id = row_ids_host[n_rows_new[i] - 1];
                if (last_row_id < segment_size - 1)
                {
                    cpu_queue.memset(
                        flags + last_row_id + 1,
                        0,
                        (segment_size - 1 - last_row_id) * sizeof(bool)
                    );
                }

                flags_modified_devices[device_index][i] = false;
            }
        }

        device_queues[device_index].wait_and_throw();
        cpu_queue.wait_and_throw();
    }

    std::tuple<bool *, int, int> build_keys_hash_table(int column, memory_manager &cpu_allocator, memory_manager &device_allocator, bool on_device, int device_index)
    {
        // Assumption: flags do not need to be synced before building hash table

        auto ht_res = current_columns[column]->build_keys_hash_table(
            (on_device ? flags_devices[device_index] : flags_host),
            (on_device ? device_allocator : cpu_allocator),
            on_device,
            device_index
        );

        std::vector<KernelBundle> ht_kernels = std::get<3>(ht_res);
        pending_kernels.push_back(ht_kernels);

        return std::make_tuple(
            std::get<0>(ht_res),
            std::get<1>(ht_res),
            std::get<2>(ht_res)
        );
    }

    std::tuple<int *, int, int> build_key_vals_hash_table(int column, bool on_device, int device_index, memory_manager &cpu_allocator, memory_manager &device_allocator)
    {
        // Assumption: flags do not need to be synced before building hash table

        auto ht_res = current_columns[column]->build_key_vals_hash_table(
            group_by_column,
            (on_device ? flags_devices[device_index] : flags_host),
            (on_device ? device_allocator : cpu_allocator),
            on_device,
            device_index
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
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
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
            const std::vector<Segment> &segments = current_columns[expr.operands[0].input]->get_segments();

            ops.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                std::vector<bool *> segments_flags_devices;
                segments_flags_devices.reserve(device_queues.size());
                std::transform(
                    flags_devices.begin(),
                    flags_devices.end(),
                    std::back_inserter(segments_flags_devices),
                    [segment_number](bool *flags_device)
                    {
                        return flags_device + segment_number * SEGMENT_SIZE;
                    }
                );
                KernelBundle bundle = segment.search_operator(
                    expr,
                    parent_op,
                    cpu_allocator,
                    device_allocators,
                    flags_host + segment_number * SEGMENT_SIZE,
                    segments_flags_devices
                );

                if (bundle.is_on_device())
                {
                    int device_index = bundle.get_device_index();
                    flags_modified_devices[device_index][segment_number] = true;
                }
                else
                    flags_modified_host[segment_number] = true;

                ops.push_back(bundle);
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
                    cpu_allocator,
                    device_allocators
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
                bool on_device =
                    segment.is_on_device() &&
                    (literal ||
                        (cols[1]->get_segments()[segment_number].is_on_device()
                            && cols[1]->get_segments()[segment_number].get_device_index() == segment.get_device_index()
                            ));
                int device_index = segment.get_device_index();
                KernelBundle bundle(on_device, device_index);

                bundle.add_kernel(
                    KernelData(
                        literal ? KernelType::SelectionKernelLiteral : KernelType::SelectionKernelColumns,
                        literal ?
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                literal_value,
                                (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
                                device_index
                            )
                            ) :
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                cols[1]->get_segments()[segment_number],
                                (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
                                device_index
                            )
                            )
                    )
                );

                if (on_device)
                    flags_modified_devices[device_index][segment_number] = true;
                else
                    flags_modified_host[segment_number] = true;

                ops.push_back(bundle);
            }

            pending_kernels.push_back(ops);
        }
    }

    void apply_project(
        const std::vector<ExprType> &exprs,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
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
                    cpu_queue,
                    device_queues,
                    cpu_allocator,
                    device_allocators[0]
                );

                // TODO better way

                std::vector<KernelBundle> fill_bundles_cpu = new_col.fill_with_literal(literal_value, false, -1, cpu_allocator);
                pending_kernels.push_back(fill_bundles_cpu);

                // std::vector<KernelBundle> fill_bundles_gpu = new_col.fill_with_literal(literal_value, true, 0, device_allocators[0]);
                // pending_kernels.push_back(fill_bundles_gpu);

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

                // TODO: pass the correct allocator in order to do host malloc, to speed up transfers.
                Column &new_col = materialized_columns.emplace_back(
                    nrows,
                    cpu_queue,
                    device_queues,
                    cpu_allocator,
                    device_allocators[0]
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
                        bool on_device = segment_a.is_on_device() && segment_b.is_on_device()
                            && segment_a.get_device_index() == segment_b.get_device_index();
                        int device_index = segment_a.get_device_index();
                        KernelBundle bundle(on_device, device_index);

                        if (on_device)
                            segment_result.build_on_device(device_allocators[device_index], device_index);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelColumns,
                                segment_result.perform_operator(
                                    segment_a,
                                    segment_b,
                                    on_device,
                                    device_index,
                                    (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
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
                        bool on_device = segment.is_on_device();
                        int device_index = segment.get_device_index();
                        KernelBundle bundle(on_device, device_index);

                        if (on_device)
                            segment_result.build_on_device(device_allocators[device_index], device_index);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralFirst,
                                segment_result.perform_operator(
                                    (int)expr.operands[0].literal.value,
                                    segment,
                                    on_device,
                                    device_index,
                                    (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
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
                        bool on_device = segment.is_on_device();
                        int device_index = segment.get_device_index();
                        KernelBundle bundle(on_device, device_index);

                        if (on_device)
                            segment_result.build_on_device(device_allocators[device_index], device_index);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralSecond,
                                segment_result.perform_operator(
                                    segment,
                                    (int)expr.operands[1].literal.value,
                                    on_device,
                                    device_index,
                                    (on_device ? flags_devices[device_index] : flags_host) + segment_number * SEGMENT_SIZE,
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

    void apply_aggregate(
        const AggType &agg,
        const std::vector<long> &group,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
    {
        std::vector<KernelBundle> agg_bundles;

        if (group.size() == 0)
        {
            const std::vector<Segment> &input_segments = current_columns[agg.operands[0]]->get_segments();

            bool on_device = current_columns[agg.operands[0]]->is_all_on_same_device(),
                need_sync = false;
            int device_index = input_segments[0].get_device_index();

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying aggregate on "
                << (on_device ? "GPU" : "CPU")
                << " with " << input_segments.size() << " segments." << std::endl;
            #endif

            if (on_device)
            {
                for (bool modified : flags_modified_host)
                {
                    if (modified)
                    {
                        // need_sync = true;
                        // break;
                        std::cerr << "Error: aggregation flags on CPU to GPU modified not implemented yet" << std::endl;
                        throw std::runtime_error("Flags on CPU to GPU modified not implemented yet.");
                    }
                }
                for (int d = 0; d < device_queues.size(); d++)
                {
                    if (need_sync)
                        break;
                    if (d == device_index)
                        continue;
                    for (bool modified : flags_modified_devices[d])
                    {
                        if (modified)
                        {
                            // need_sync = true;
                            // break;
                            std::cerr << "Error: aggregation flags on other GPU modified not implemented yet" << std::endl;
                            throw std::runtime_error("Flags on other GPU modified not implemented yet.");
                        }
                    }
                }
            }
            else
            {
                for (const auto &flags_modified_gpu : flags_modified_devices)
                {
                    for (bool modified : flags_modified_gpu)
                    {
                        if (modified)
                        {
                            need_sync = true;
                            break;
                        }
                    }
                    if (need_sync)
                        break;
                }
            }

            if (need_sync)
            {
                for (int d = 0; d < device_queues.size(); d++)
                    compress_and_sync(
                        cpu_allocator,
                        device_allocators[d],
                        d
                    );
            }
            uint64_t *final_result = (on_device ?
                device_allocators[device_index].alloc_zero<uint64_t>(1) :
                cpu_allocator.alloc_zero<uint64_t>(1)
                );

            agg_bundles.reserve(input_segments.size());

            for (int i = 0; i < input_segments.size(); i++)
            {
                const Segment &input_segment = input_segments[i];
                KernelBundle bundle(on_device, device_index);
                bundle.add_kernel(
                    KernelData(
                        KernelType::AggregateOperationKernel,
                        input_segment.aggregate_operator(
                            (on_device ? flags_devices[device_index] : flags_host) + i * SEGMENT_SIZE,
                            on_device,
                            device_index,
                            final_result
                        )
                    )
                );
                agg_bundles.push_back(bundle);
            }

            pending_kernels.push_back(agg_bundles);

            auto dependencies = execute_pending_kernels();
            pending_kernels_dependencies_cpu = dependencies.first;
            for (int d = 0; d < device_queues.size(); d++)
            {
                pending_kernels_dependencies_devices[d] = dependencies.second[d];
                bool *new_flags = device_allocators[d].alloc<bool>(1, true);
                pending_kernels_dependencies_devices[d].push_back(
                    device_queues[d].fill<bool>(new_flags, true, 1)
                );
                flags_devices[d] = new_flags;
                flags_modified_devices[d] = { false };
            }

            bool *new_cpu_flags = cpu_allocator.alloc<bool>(1, true);
            new_cpu_flags[0] = true;
            flags_host = new_cpu_flags;
            flags_modified_host = { false };

            nrows = 1;

            Column &result_column = materialized_columns.emplace_back(
                final_result,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                nrows
            );

            current_columns.clear();
            current_columns.push_back(&result_column);
        }
        else
        {
            const Column *agg_column = current_columns[agg.operands[0]];
            bool on_device = agg_column->is_all_on_same_device(),
                need_sync = false;
            int device_index = agg_column->get_segments()[0].get_device_index();
            for (int i = 0; i < group.size() && on_device; i++)
            {
                on_device = current_columns[group[i]]->is_all_on_same_device() &&
                    current_columns[group[i]]->get_segments()[0].get_device_index() == device_index;
            }

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying group-by aggregate on "
                << (on_device ? "GPU" : "CPU") << std::endl;
            #endif

            if (on_device)
            {
                for (bool modified : flags_modified_host)
                {
                    if (modified)
                    {
                        // need_sync = true;
                        // break;
                        std::cerr << "Error: aggregation flags on CPU to GPU modified not implemented yet" << std::endl;
                        throw std::runtime_error("Flags on CPU to GPU modified not implemented yet.");
                    }
                }
                for (int d = 0; d < device_queues.size(); d++)
                {
                    if (need_sync)
                        break;
                    if (d == device_index)
                        continue;
                    for (bool modified : flags_modified_devices[d])
                    {
                        if (modified)
                        {
                            // need_sync = true;
                            // break;
                            std::cerr << "Error: aggregation flags on other GPU modified not implemented yet" << std::endl;
                            throw std::runtime_error("Flags on other GPU modified not implemented yet.");
                        }
                    }
                }
            }
            else
            {
                for (const auto &flags_modified_gpu : flags_modified_devices)
                {
                    for (bool modified : flags_modified_gpu)
                    {
                        if (modified)
                        {
                            need_sync = true;
                            break;
                        }
                    }
                    if (need_sync)
                        break;
                }
            }

            need_sync = need_sync || agg_column->needs_copy_on(on_device, device_index);
            for (int i = 0; i < group.size() && !need_sync; i++)
                need_sync = current_columns[group[i]]->needs_copy_on(on_device, device_index);


            if (need_sync)
            {
                std::vector<Column *> columns_to_sync;
                columns_to_sync.reserve(group.size() + 1);
                columns_to_sync.push_back(const_cast<Column *>(agg_column));
                for (int i = 0; i < group.size(); i++)
                    columns_to_sync.push_back(current_columns[group[i]]);
                for (int d = 0; d < device_queues.size(); d++)
                    compress_and_sync(
                        cpu_allocator,
                        device_allocators[d],
                        d,
                        columns_to_sync
                    );
            }

            memory_manager &allocator = on_device ? device_allocators[device_index] : cpu_allocator;

            uint64_t prod_ranges = 1;
            int *min = allocator.alloc<int>(group.size(), !on_device),
                *max = allocator.alloc<int>(group.size(), !on_device);

            for (int i = 0; i < group.size(); i++)
            {
                auto min_max = current_columns[group[i]]->get_min_max();
                min[i] = min_max.first;
                max[i] = min_max.second;
                prod_ranges *= max[i] - min[i] + 1;
            }

            uint64_t *aggregate_result = allocator.alloc_zero<uint64_t>(prod_ranges);
            unsigned *temp_flags = allocator.alloc_zero<unsigned>(prod_ranges);
            int **results = allocator.alloc<int *>(group.size(), !on_device);

            for (int i = 0; i < group.size(); i++)
                results[i] = allocator.alloc<int>(prod_ranges, true);

            const std::vector<Segment> &agg_segments = agg_column->get_segments();

            agg_bundles.reserve(agg_segments.size());

            for (int i = 0; i < agg_segments.size(); i++)
            {
                const int **contents = allocator.alloc<const int *>(group.size(), !on_device);
                for (int j = 0; j < group.size(); j++)
                {
                    const Segment &segment = current_columns[group[j]]->get_segments()[i];
                    contents[j] = segment.get_data(on_device, device_index);
                }
                const Segment &agg_segment = agg_segments[i];

                KernelBundle bundle(on_device, device_index);
                bundle.add_kernel(
                    KernelData(
                        KernelType::GroupByAggregateKernel,
                        agg_segment.group_by_aggregate_operator(
                            contents,
                            max,
                            min,
                            (on_device ? flags_devices[device_index] : flags_host) + i * SEGMENT_SIZE,
                            aggregate_result,
                            group.size(),
                            results,
                            temp_flags,
                            on_device,
                            device_index,
                            prod_ranges
                        )
                    )
                );
                agg_bundles.push_back(bundle);
            }

            pending_kernels.push_back(agg_bundles);

            // std::cout << "Executing aggregate kernels" << std::endl;
            auto dependencies = execute_pending_kernels();

            bool *new_cpu_flags = device_allocators[device_index].alloc<bool>(prod_ranges, false);
            flags_host = new_cpu_flags;

            nrows = prod_ranges;

            uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);

            if (on_device)
            {
                bool *new_gpu_flags = device_allocators[device_index].alloc<bool>(prod_ranges, true);
                auto e1 = device_queues[device_index].submit(
                    [&](sycl::handler &cgh)
                    {
                        if (!dependencies.second[0].empty())
                            cgh.depends_on(dependencies.second[0]);
                        cgh.parallel_for(
                            prod_ranges,
                            [=](sycl::id<1> idx)
                            {
                                new_gpu_flags[idx[0]] = temp_flags[idx[0]] != 0;
                            }
                        );
                    }
                );

                flags_devices[device_index] = new_gpu_flags;
                pending_kernels_dependencies_devices[device_index].push_back(
                    device_queues[device_index].memcpy(
                        new_cpu_flags,
                        new_gpu_flags,
                        sizeof(bool) * prod_ranges,
                        e1
                    )
                );
                for (int d = 0; d < device_queues.size(); d++)
                {
                    // if (d != device_index)
                    // {
                    //     bool *new_flags = device_allocators[d].alloc<bool>(prod_ranges, true);
                    //     pending_kernels_dependencies_devices[d].push_back(
                    //         device_queues[d].memcpy(
                    //             new_flags,
                    //             new_gpu_flags,
                    //             sizeof(bool) * prod_ranges,
                    //             e1
                    //         )
                    //     );
                    //     flags_devices[d] = new_flags;
                    // }
                    flags_modified_devices[d].resize(segment_num, false);
                    std::fill(
                        flags_modified_devices[d].begin(),
                        flags_modified_devices[d].end(),
                        false
                    );
                }
            }
            else
            {
                pending_kernels_dependencies_cpu.push_back(
                    cpu_queue.submit(
                        [&](sycl::handler &cgh)
                        {
                            if (!dependencies.first.empty())
                                cgh.depends_on(dependencies.first);
                            cgh.parallel_for(
                                prod_ranges,
                                [=](sycl::id<1> idx)
                                {
                                    new_cpu_flags[idx[0]] = temp_flags[idx[0]] != 0;
                                }
                            );
                        }
                    )
                );

                // gpu update skipped since after aggregation on cpu, nothing is run on gpu
            }

            current_columns.clear();

            for (int i = 0; i < group.size(); i++)
            {
                Column &new_col = materialized_columns.emplace_back(
                    results[i],
                    on_device,
                    device_index,
                    cpu_queue,
                    device_queues,
                    cpu_allocator,
                    device_allocators,
                    prod_ranges
                );
                current_columns.push_back(&new_col);
            }

            Column &agg_col = materialized_columns.emplace_back(
                aggregate_result,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                prod_ranges
            );
            current_columns.push_back(&agg_col);

            flags_modified_host.resize(segment_num, false);
            std::fill(
                flags_modified_host.begin(),
                flags_modified_host.end(),
                false
            );
        }
    }

    void apply_join(
        TransientTable &right_table,
        const RelNode &rel,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators)
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
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying semi-join" << std::endl;
            #endif

            bool *ht_cpu = nullptr;
            std::vector<bool *> ht_devices(device_queues.size(), nullptr);
            int build_min_value, build_max_value;

            for (const Segment &seg : current_columns[left_column]->get_segments())
            {
                int device_index = seg.get_device_index();
                if (seg.is_on_device() && ht_devices[device_index] == nullptr)
                {
                    auto ht_data = right_table.build_keys_hash_table(
                        right_column,
                        cpu_allocator,
                        device_allocators[device_index],
                        true,
                        device_index
                    );
                    ht_devices[device_index] = std::get<0>(ht_data);
                    build_min_value = std::get<1>(ht_data);
                    build_max_value = std::get<2>(ht_data);
                }
                else if (!seg.is_on_device() && ht_cpu == nullptr)
                {
                    auto ht_data = right_table.build_keys_hash_table(
                        right_column,
                        cpu_allocator,
                        device_allocators[0],
                        false,
                        -1
                    );
                    ht_cpu = std::get<0>(ht_data);
                    build_min_value = std::get<1>(ht_data);
                    build_max_value = std::get<2>(ht_data);
                }
            }

            auto ht_dependencies = right_table.execute_pending_kernels();

            pending_kernels_dependencies_cpu.insert(
                pending_kernels_dependencies_cpu.end(),
                ht_dependencies.first.begin(),
                ht_dependencies.first.end()
            );
            for (int d = 0; d < device_queues.size(); d++)
            {
                pending_kernels_dependencies_devices[d].insert(
                    pending_kernels_dependencies_devices[d].end(),
                    ht_dependencies.second[d].begin(),
                    ht_dependencies.second[d].end()
                );
            }

            pending_kernels.push_back(
                current_columns[left_column]->semi_join(
                    flags_host,
                    flags_devices,
                    build_min_value,
                    build_max_value,
                    ht_cpu,
                    ht_devices,
                    flags_modified_host,
                    flags_modified_devices
                )
            );

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);
        }
        else
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying full join" << std::endl;
            #endif

            auto probe_col_locations = current_columns[left_column]->get_positions();

            bool can_probe_on_device = false, can_build_on_device = false;
            for (bool on_device : probe_col_locations.second)
                can_probe_on_device = can_probe_on_device || on_device;

            auto col_devices = right_table.current_columns[right_column]->get_full_col_on_device(),
                group_by_col_devices = right_table.group_by_column->get_full_col_on_device();

            for (int d = 0; d < device_queues.size(); d++)
            {
                col_devices[d] = col_devices[d] && group_by_col_devices[d];
                can_build_on_device = can_build_on_device || col_devices[d];
            }

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Join hash table will be built on "
                << (can_build_on_device && can_probe_on_device ? "GPU" : "CPU")
                << (can_build_on_device && can_probe_on_device && probe_col_locations.first ? " and CPU" : "")
                << std::endl;
            #endif

            Column &new_column = materialized_columns.emplace_back(
                nrows,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators[0],
                true
            );

            auto min_max_gb = right_table.group_by_column->get_min_max();
            uint64_t group_by_col_index = current_columns.size() + right_table.group_by_column_index;

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);

            current_columns[group_by_col_index] = &materialized_columns[materialized_columns.size() - 1];

            int build_min_value, build_max_value, *ht_host = nullptr;
            std::vector<int *> ht_devices(device_queues.size(), nullptr);

            for (int d = 0; d < device_queues.size(); d++)
            {
                if (probe_col_locations.second[d] && col_devices[d])
                {
                    auto ht_data = right_table.build_key_vals_hash_table(
                        right_column,
                        true,
                        d,
                        cpu_allocator,
                        device_allocators[d]
                    );
                    ht_devices[d] = std::get<0>(ht_data);
                    build_min_value = std::get<1>(ht_data);
                    build_max_value = std::get<2>(ht_data);
                }
            }

            // Here we don't check if all probe segments have compatible device locations with build
            // Assumed all probe segments on a device have the full build columns on device too
            if (probe_col_locations.first)
            {
                auto ht_data = right_table.build_key_vals_hash_table(
                    right_column,
                    false,
                    -1,
                    cpu_allocator,
                    device_allocators[0]
                );
                ht_host = std::get<0>(ht_data);
                build_min_value = std::get<1>(ht_data);
                build_max_value = std::get<2>(ht_data);
            }

            auto ht_dependencies = right_table.execute_pending_kernels();

            pending_kernels_dependencies_cpu.insert(
                pending_kernels_dependencies_cpu.end(),
                ht_dependencies.first.begin(),
                ht_dependencies.first.end()
            );
            for (int d = 0; d < device_queues.size(); d++)
            {
                pending_kernels_dependencies_devices[d].insert(
                    pending_kernels_dependencies_devices[d].end(),
                    ht_dependencies.second[d].begin(),
                    ht_dependencies.second[d].end()
                );
            }

            auto join_ops = current_columns[left_column]->full_join_operation(
                flags_host,
                flags_devices,
                build_min_value,
                build_max_value,
                min_max_gb.first,
                min_max_gb.second,
                ht_host,
                ht_devices,
                new_column,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                flags_modified_host,
                flags_modified_devices
            );

            pending_kernels.push_back(join_ops);
        }
    }
};