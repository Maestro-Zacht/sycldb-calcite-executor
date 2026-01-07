#pragma once

#include <sycl/sycl.hpp>

#include "../common.hpp"

#include "models.hpp"
#include "execution.hpp"
#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/common.hpp"

class TransientTable
{
private:
    bool *flags_host, *flags_gpu;
    std::vector<bool> flags_modified_host, flags_modified_gpu;
    sycl::queue &gpu_queue, &cpu_queue;
    #if USE_FUSION
    sycl::ext::codeplay::experimental::fusion_wrapper fw_gpu, fw_cpu;
    #endif
    std::vector<Column *> current_columns;
    std::vector<Column> materialized_columns;
    uint64_t nrows;
    Column *group_by_column;
    uint64_t group_by_column_index;
    std::vector<std::vector<KernelBundle>> pending_kernels;
    std::vector<sycl::event> pending_kernels_dependencies_gpu, pending_kernels_dependencies_cpu;
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
        // std::cout << "Creating transient table with " << nrows << " rows." << std::endl;
        flags_gpu = gpu_allocator.alloc<bool>(nrows, true);
        auto e1 = gpu_queue.fill<bool>(flags_gpu, true, nrows);

        flags_host = cpu_allocator.alloc<bool>(nrows, true);
        auto e2 = cpu_queue.fill<bool>(flags_host, true, nrows);

        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);

        flags_modified_gpu.resize(segment_num, false);
        flags_modified_host.resize(segment_num, false);

        const std::vector<Column> &base_columns = base_table->get_columns();

        current_columns.reserve(base_columns.size() + 100);
        for (const Column &col : base_columns)
            current_columns.push_back(const_cast<Column *>(&col));

        materialized_columns.reserve(50);

        e1.wait();
        e2.wait();
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

    std::pair<std::vector<sycl::event>, std::vector<sycl::event>> execute_pending_kernels(
        #if USE_FUSION
        bool fuse = true
        #endif
    )
    {
        // std::cout << "start execute" << std::endl;
        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
        std::vector<sycl::event> events_gpu, events_cpu;
        bool executed_gpu = false,
            executed_cpu = false;

        if (pending_kernels.size() == 0)
        {
            events_gpu = pending_kernels_dependencies_gpu;
            events_cpu = pending_kernels_dependencies_cpu;
            pending_kernels_dependencies_gpu.clear();
            pending_kernels_dependencies_cpu.clear();

            // std::cout << "No pending kernels to execute." << std::endl;

            return { events_gpu, events_cpu };
        }

        for (const auto &phases : pending_kernels)
        {
            if (phases.size() != segment_num)
            {
                std::cerr << "Pending kernels segment number mismatch: expected " << segment_num << ", got " << phases.size() << std::endl;
                throw std::runtime_error("Pending kernels segment number mismatch.");
            }
        }

        events_gpu.reserve(
            segment_num
            #if USE_FUSION 
            * 2
            #endif
        );
        events_cpu.reserve(
            segment_num
            #if USE_FUSION 
            * 2
            #endif
        );

        for (uint64_t segment_index = 0; segment_index < segment_num; segment_index++)
        {
            // std::cout << " Executing segment " << segment_index + 1 << "/" << segment_num << std::endl;
            // std::vector<sycl::event> deps_gpu = pending_kernels_dependencies_gpu,
            //     deps_cpu = pending_kernels_dependencies_cpu,
            //     tmp;
            std::vector<sycl::event> deps_gpu, deps_cpu, tmp;

            if (pending_kernels_dependencies_gpu.size() == segment_num)
                deps_gpu.push_back(pending_kernels_dependencies_gpu[segment_index]);
            else
                deps_gpu = pending_kernels_dependencies_gpu;

            if (pending_kernels_dependencies_cpu.size() == segment_num)
                deps_cpu.push_back(pending_kernels_dependencies_cpu[segment_index]);
            else
                deps_cpu = pending_kernels_dependencies_cpu;


            for (int i = 0; i < 2; i++)
            {
                bool kernel_present = false;

                #if USE_FUSION
                if (i > 0 && fuse)
                {
                    fw_gpu.start_fusion();
                }
                #endif

                for (const auto &phases : pending_kernels)
                {
                    const KernelBundle &bundle = phases[segment_index];
                    bool on_device = bundle.is_on_device();

                    if ((i == 0) != on_device)
                    {
                        tmp = bundle.execute(
                            gpu_queue,
                            cpu_queue,
                            deps_gpu,
                            deps_cpu
                        );

                        if (on_device)
                            deps_gpu = std::move(tmp);
                        else
                            deps_cpu = std::move(tmp);

                        kernel_present = true;
                    }
                }

                #if USE_FUSION
                if (i == 0 && kernel_present)
                {
                    events_cpu.insert(
                        events_cpu.end(),
                        deps_cpu.begin(),
                        deps_cpu.end()
                    );
                    executed_cpu = true;
                }
                else if (i > 0)
                {
                    if (!kernel_present)
                    {
                        fw_gpu.cancel_fusion();
                    }
                    else if (fuse)
                    {
                        events_gpu.push_back(
                            fw_gpu.complete_fusion(sycl::ext::codeplay::experimental::property::no_barriers {})
                        );
                        executed_gpu = true;
                    }
                    else
                    {
                        events_gpu.insert(
                            events_gpu.end(),
                            deps_gpu.begin(),
                            deps_gpu.end()
                        );
                        executed_gpu = true;
                    }
                }
                #else
                if (kernel_present)
                {
                    if (i > 0)
                    {
                        events_gpu.insert(
                            events_gpu.end(),
                            deps_gpu.begin(),
                            deps_gpu.end()
                        );
                        executed_gpu = true;
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

        if (!executed_gpu)
            events_gpu = pending_kernels_dependencies_gpu;
        if (!executed_cpu)
            events_cpu = pending_kernels_dependencies_cpu;

        pending_kernels.clear();
        pending_kernels_dependencies_gpu.clear();
        pending_kernels_dependencies_cpu.clear();

        // std::cout << "end execute" << std::endl;

        return { events_gpu, events_cpu };
    }

    void sync_flags(int column, memory_manager &gpu_allocator, memory_manager &cpu_allocator)
    {
        bool need_flag_sync = false;
        std::vector<KernelBundle> flag_sync_kernels;

        const std::vector<Segment> &segments = current_columns[column]->get_segments();

        flag_sync_kernels.reserve(segments.size());

        for (uint64_t i = 0; i < segments.size(); i++)
        {
            const Segment &seg = segments[i];
            bool seg_on_device = seg.is_on_device();
            KernelBundle bundle(seg_on_device);
            if (seg_on_device)
            {
                if (flags_modified_host[i])
                {
                    need_flag_sync = true;
                    if (flags_modified_gpu[i])
                    {
                        bool *temp_flags = gpu_allocator.alloc<bool>(seg.get_nrows(), true);
                        bundle.add_kernel(
                            KernelData(
                                KernelType::SyncFlagsKernel,
                                new SyncFlagsKernel(
                                    flags_host + i * SEGMENT_SIZE,
                                    flags_gpu + i * SEGMENT_SIZE,
                                    temp_flags,
                                    seg.get_nrows()
                                )
                            )
                        );
                    }
                    else
                    {
                        bundle.add_kernel(
                            KernelData(
                                KernelType::CopyKernel,
                                new CopyKernel(
                                    flags_host + i * SEGMENT_SIZE,
                                    flags_gpu + i * SEGMENT_SIZE,
                                    seg.get_nrows(),
                                    sizeof(bool)
                                )
                            )
                        );
                    }
                    flags_modified_host[i] = false;
                }
                else
                {
                    bundle.add_kernel(
                        KernelData(
                            KernelType::EmptyKernel,
                            new EmptyKernel(1)
                        )
                    );
                }
            }
            else
            {
                if (flags_modified_gpu[i])
                {
                    need_flag_sync = true;
                    if (flags_modified_host[i])
                    {
                        bool *temp_flags = cpu_allocator.alloc<bool>(seg.get_nrows(), true);
                        bundle.add_kernel(
                            KernelData(
                                KernelType::SyncFlagsKernel,
                                new SyncFlagsKernel(
                                    flags_gpu + i * SEGMENT_SIZE,
                                    flags_host + i * SEGMENT_SIZE,
                                    temp_flags,
                                    seg.get_nrows()
                                )
                            )
                        );
                    }
                    else
                    {
                        bundle.add_kernel(
                            KernelData(
                                KernelType::CopyKernel,
                                new CopyKernel(
                                    flags_gpu + i * SEGMENT_SIZE,
                                    flags_host + i * SEGMENT_SIZE,
                                    seg.get_nrows(),
                                    sizeof(bool)
                                )
                            )
                        );
                    }
                    flags_modified_gpu[i] = false;
                }
                else
                {
                    bundle.add_kernel(
                        KernelData(
                            KernelType::EmptyKernel,
                            new EmptyKernel(1)
                        )
                    );
                }
            }

            flag_sync_kernels.push_back(bundle);
        }

        if (need_flag_sync)
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Synchronizing flags" << std::endl;
            #endif
            pending_kernels.push_back(flag_sync_kernels);
        }
        #if not PERFORMANCE_MEASUREMENT_ACTIVE
        else
        {
            std::cout << "No flag synchronization needed" << std::endl;
        }
        #endif
    }

    void update_flags(bool to_device, memory_manager &gpu_allocator, memory_manager &cpu_allocator)
    {
        uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
        bool need_flag_sync = false;
        std::vector<KernelBundle> flag_sync_kernels;

        flag_sync_kernels.reserve(segment_num);

        for (uint64_t i = 0; i < segment_num; i++)
        {
            uint64_t seg_size = (i == segment_num - 1) ? (nrows - i * SEGMENT_SIZE) : SEGMENT_SIZE;
            KernelBundle bundle(to_device);
            if (to_device)
            {
                if (flags_modified_host[i])
                {
                    need_flag_sync = true;
                    if (flags_modified_gpu[i])
                    {
                        bool *temp_flags = gpu_allocator.alloc<bool>(seg_size, true);
                        bundle.add_kernel(
                            KernelData(
                                KernelType::SyncFlagsKernel,
                                new SyncFlagsKernel(
                                    flags_host + i * SEGMENT_SIZE,
                                    flags_gpu + i * SEGMENT_SIZE,
                                    temp_flags,
                                    seg_size
                                )
                            )
                        );
                    }
                    else
                    {
                        bundle.add_kernel(
                            KernelData(
                                KernelType::CopyKernel,
                                new CopyKernel(
                                    flags_host + i * SEGMENT_SIZE,
                                    flags_gpu + i * SEGMENT_SIZE,
                                    seg_size,
                                    sizeof(bool)
                                )
                            )
                        );
                    }
                    flags_modified_host[i] = false;
                }
                else
                {
                    bundle.add_kernel(
                        KernelData(
                            KernelType::EmptyKernel,
                            new EmptyKernel(1)
                        )
                    );
                }
            }
            else
            {
                if (flags_modified_gpu[i])
                {
                    need_flag_sync = true;
                    if (flags_modified_host[i])
                    {
                        bool *temp_flags = cpu_allocator.alloc<bool>(seg_size, true);
                        bundle.add_kernel(
                            KernelData(
                                KernelType::SyncFlagsKernel,
                                new SyncFlagsKernel(
                                    flags_gpu + i * SEGMENT_SIZE,
                                    flags_host + i * SEGMENT_SIZE,
                                    temp_flags,
                                    seg_size
                                )
                            )
                        );
                    }
                    else
                    {
                        bundle.add_kernel(
                            KernelData(
                                KernelType::CopyKernel,
                                new CopyKernel(
                                    flags_gpu + i * SEGMENT_SIZE,
                                    flags_host + i * SEGMENT_SIZE,
                                    seg_size,
                                    sizeof(bool)
                                )
                            )
                        );
                    }
                    flags_modified_gpu[i] = false;
                }
                else
                {
                    bundle.add_kernel(
                        KernelData(
                            KernelType::EmptyKernel,
                            new EmptyKernel(1)
                        )
                    );
                }
            }

            flag_sync_kernels.push_back(bundle);
        }

        if (need_flag_sync)
        {
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Synchronizing flags to " << (to_device ? "GPU" : "CPU") << std::endl;
            #endif
            pending_kernels.push_back(flag_sync_kernels);
        }
        #if not PERFORMANCE_MEASUREMENT_ACTIVE
        else
        {
            std::cout << "No flag synchronization needed" << std::endl;
        }
        #endif
    }

    void assert_flags_to_cpu()
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

    uint64_t count_flags_true(bool on_device, memory_manager &gpu_allocator, memory_manager &cpu_allocator, bool do_flags_update = true)
    {
        auto dependencies = execute_pending_kernels();

        if (do_flags_update)
        {
            pending_kernels_dependencies_gpu = dependencies.first;
            pending_kernels_dependencies_cpu = dependencies.second;

            update_flags(on_device, gpu_allocator, cpu_allocator);

            dependencies = execute_pending_kernels(
                #if USE_FUSION
                false
                #endif
            );
        }

        return count_true_flags(
            on_device ? flags_gpu : flags_host,
            nrows,
            on_device ? gpu_queue : cpu_queue,
            on_device ? dependencies.first : dependencies.second
        );
    }

    std::tuple<int *, sycl::event, int *, sycl::event, uint64_t> build_row_ids(int segment_n, int segment_size, memory_manager &gpu_allocator, memory_manager &cpu_allocator)
    {
        int *row_id_index = gpu_allocator.alloc<int>(1, true);
        bool *flags = flags_gpu + segment_n * SEGMENT_SIZE;
        uint64_t n_rows_new = count_true_flags(
            flags,
            segment_size,
            gpu_queue,
            gpu_allocator
        );
        int *row_ids_gpu = gpu_allocator.alloc<int>(n_rows_new, true);
        sycl::event e_row_id_gpu = gpu_queue.submit(
            [&](sycl::handler &cgh)
            {
                cgh.parallel_for(
                    segment_size,
                    [=](sycl::id<1> idx)
                    {
                        auto id = idx[0];
                        if (flags[id])
                        {
                            sycl::atomic_ref<
                                int,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space
                            > row_idx_atomic(*row_id_index);
                            int pos = row_idx_atomic.fetch_add(1);
                            row_ids_gpu[pos] = id;
                        }
                    }
                );
            }
        );
        int *row_ids_host = gpu_allocator.alloc<int>(n_rows_new, false);
        sycl::event e_row_id_host = gpu_queue.memcpy(
            row_ids_host,
            row_ids_gpu,
            n_rows_new * sizeof(int),
            e_row_id_gpu
        );

        return { row_ids_gpu, e_row_id_gpu, row_ids_host, e_row_id_host, n_rows_new };
    }

    // This is a sync point
    void compress_and_sync(memory_manager &gpu_allocator, memory_manager &cpu_allocator, const std::vector<Column *> &columns_to_sync)
    {
        int num_segments = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);

        gpu_queue.wait_and_throw();
        cpu_queue.wait_and_throw();

        for (int i = 0; i < num_segments; i++)
        {
            int *row_ids_host = nullptr,
                *row_ids_gpu = nullptr;
            sycl::event e_row_id_host,
                e_row_id_gpu;
            uint64_t n_rows_new = 0;

            uint64_t segment_size = (i == num_segments - 1) ? (nrows - i * SEGMENT_SIZE) : SEGMENT_SIZE;

            if (flags_modified_gpu[i])
            {
                auto row_id_res = build_row_ids(i, segment_size, gpu_allocator, cpu_allocator);
                row_ids_gpu = std::get<0>(row_id_res);
                e_row_id_gpu = std::get<1>(row_id_res);
                row_ids_host = std::get<2>(row_id_res);
                e_row_id_host = std::get<3>(row_id_res);
                n_rows_new = std::get<4>(row_id_res);
            }

            for (const Column *col : columns_to_sync)
            {
                Segment &seg = const_cast<Segment &>(col->get_segments()[i]);
                if (seg.needs_copy_on(false))
                {
                    if (row_ids_gpu == nullptr)
                    {
                        auto row_id_res = build_row_ids(i, segment_size, gpu_allocator, cpu_allocator);
                        row_ids_gpu = std::get<0>(row_id_res);
                        e_row_id_gpu = std::get<1>(row_id_res);
                        row_ids_host = std::get<2>(row_id_res);
                        e_row_id_host = std::get<3>(row_id_res);
                        n_rows_new = std::get<4>(row_id_res);
                    }

                    seg.compress_sync(
                        row_ids_gpu,
                        e_row_id_gpu,
                        row_ids_host,
                        e_row_id_host,
                        n_rows_new,
                        gpu_allocator,
                        cpu_allocator
                    );
                }
            }

            if (flags_modified_gpu[i])
            {
                if (flags_modified_host[i])
                {
                    std::cerr << "Both flags modified on GPU and CPU during compress_and_sync." << std::endl;
                    throw std::runtime_error("Both flags modified on GPU and CPU during compress_and_sync.");
                }
                bool *flags = flags_host + i * SEGMENT_SIZE;
                auto e_memset = cpu_queue.memset(
                    flags,
                    0,
                    segment_size * sizeof(bool)
                );
                std::cout << "segment size: " << segment_size << " - row ids: [";
                e_row_id_host.wait();
                e_memset.wait();
                for (uint64_t k = 0; k < n_rows_new; k++)
                {
                    std::cout << row_ids_host[k] << (k == n_rows_new - 1 ? "" : ", ");
                    flags[row_ids_host[k]] = true;
                }
                std::cout << "]" << std::endl;
                // cpu_queue.submit(
                //     [&](sycl::handler &cgh)
                //     {
                //         cgh.depends_on(e_row_id_host);
                //         cgh.depends_on(e_memset);
                //         cgh.parallel_for(
                //             n_rows_new,
                //             [=](sycl::id<1> idx)
                //             {
                //                 auto i = idx[0];
                //                 int row_id = row_ids_host[i];
                //                 flags[row_id] = true;
                //             }
                //         );
                //     }
                // );
                flags_modified_gpu[i] = false;
            }
        }

        gpu_queue.wait_and_throw();
        cpu_queue.wait_and_throw();
    }

    std::tuple<bool *, int, int, bool> build_keys_hash_table(int column, memory_manager &gpu_allocator, memory_manager &cpu_allocator)
    {
        bool on_device = current_columns[column]->is_all_on_device(),
            need_sync = false;

        if (on_device)
        {
            for (bool modified : flags_modified_host)
            {
                if (modified)
                {
                    need_sync = true;
                    break;
                }
            }
        }
        else
        {
            for (bool modified : flags_modified_gpu)
            {
                if (modified)
                {
                    need_sync = true;
                    break;
                }
            }
        }

        if (need_sync)
        {
            auto dependencies = execute_pending_kernels();
            pending_kernels_dependencies_gpu = dependencies.first;
            pending_kernels_dependencies_cpu = dependencies.second;

            sync_flags(column, gpu_allocator, cpu_allocator);

            dependencies = execute_pending_kernels();
            pending_kernels_dependencies_gpu = dependencies.first;
            pending_kernels_dependencies_cpu = dependencies.second;
        }

        auto ht_res = current_columns[column]->build_keys_hash_table(
            (on_device ? flags_gpu : flags_host),
            (on_device ? gpu_allocator : cpu_allocator),
            on_device
        );

        std::vector<KernelBundle> ht_kernels = std::get<3>(ht_res);
        pending_kernels.push_back(ht_kernels);

        return std::make_tuple(
            std::get<0>(ht_res),
            std::get<1>(ht_res),
            std::get<2>(ht_res),
            on_device
        );
    }

    std::tuple<int *, int, int> build_key_vals_hash_table(int column, bool on_device, memory_manager &gpu_allocator, memory_manager &cpu_allocator)
    {
        bool need_sync = false;
        if (on_device)
        {
            for (bool modified : flags_modified_host)
            {
                if (modified)
                {
                    need_sync = true;
                    break;
                }
            }
        }
        else
        {
            for (bool modified : flags_modified_gpu)
            {
                if (modified)
                {
                    need_sync = true;
                    break;
                }
            }
        }

        if (need_sync)
        {
            auto dependencies = execute_pending_kernels();
            pending_kernels_dependencies_gpu = dependencies.first;
            pending_kernels_dependencies_cpu = dependencies.second;

            update_flags(on_device, gpu_allocator, cpu_allocator);

            dependencies = execute_pending_kernels(
                #if USE_FUSION
                false
                #endif
            );
            #if USE_FUSION
            gpu_queue.wait_and_throw();
            cpu_queue.wait_and_throw();
            #else
            pending_kernels_dependencies_gpu = dependencies.first;
            pending_kernels_dependencies_cpu = dependencies.second;
            #endif
        }

        auto ht_res = current_columns[column]->build_key_vals_hash_table(
            group_by_column,
            (on_device ? flags_gpu : flags_host),
            (on_device ? gpu_allocator : cpu_allocator),
            on_device
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
            const std::vector<Segment> &segments = current_columns[expr.operands[0].input]->get_segments();

            ops.reserve(segments.size());

            for (size_t segment_number = 0; segment_number < segments.size(); segment_number++)
            {
                const Segment &segment = segments[segment_number];
                KernelBundle bundle = segment.search_operator(
                    expr,
                    parent_op,
                    gpu_allocator,
                    cpu_allocator,
                    flags_gpu + segment_number * SEGMENT_SIZE,
                    flags_host + segment_number * SEGMENT_SIZE
                );

                if (bundle.is_on_device())
                    flags_modified_gpu[segment_number] = true;
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
                bool on_device =
                    segment.is_on_device() &&
                    (literal || (cols[1]->get_segments()[segment_number].is_on_device()));
                KernelBundle bundle(on_device);

                bundle.add_kernel(
                    KernelData(
                        literal ? KernelType::SelectionKernelLiteral : KernelType::SelectionKernelColumns,
                        literal ?
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                literal_value,
                                (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE
                            )
                            ) :
                        static_cast<KernelDefinition *>(
                            segment.filter_operator(
                                expr.op,
                                parent_op,
                                cols[1]->get_segments()[segment_number],
                                (on_device ? flags_gpu : flags_host) + segment_number * SEGMENT_SIZE
                            )
                            )
                    )
                );

                if (on_device)
                    flags_modified_gpu[segment_number] = true;
                else
                    flags_modified_host[segment_number] = true;

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
                    cpu_allocator
                );

                // TODO better way

                std::vector<KernelBundle> fill_bundles_cpu = new_col.fill_with_literal(literal_value, false, cpu_allocator);
                pending_kernels.push_back(fill_bundles_cpu);

                std::vector<KernelBundle> fill_bundles_gpu = new_col.fill_with_literal(literal_value, true, gpu_allocator);
                pending_kernels.push_back(fill_bundles_gpu);

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
                    cpu_allocator
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
                        bool on_device = segment_a.is_on_device() && segment_b.is_on_device();
                        KernelBundle bundle(on_device);

                        if (on_device)
                            segment_result.build_on_device(gpu_allocator);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelColumns,
                                segment_result.perform_operator(
                                    segment_a,
                                    segment_b,
                                    on_device,
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
                        bool on_device = segment.is_on_device();
                        KernelBundle bundle(on_device);

                        if (on_device)
                            segment_result.build_on_device(gpu_allocator);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralFirst,
                                segment_result.perform_operator(
                                    (int)expr.operands[0].literal.value,
                                    segment,
                                    on_device,
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
                        bool on_device = segment.is_on_device();
                        KernelBundle bundle(on_device);

                        if (on_device)
                            segment_result.build_on_device(gpu_allocator);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::PerformOperationKernelLiteralSecond,
                                segment_result.perform_operator(
                                    segment,
                                    (int)expr.operands[1].literal.value,
                                    on_device,
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

    void apply_aggregate(
        const AggType &agg,
        const std::vector<long> &group,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator)
    {
        std::vector<KernelBundle> agg_bundles;

        if (group.size() == 0)
        {
            const std::vector<Segment> &input_segments = current_columns[agg.operands[0]]->get_segments();

            bool on_device = current_columns[agg.operands[0]]->is_all_on_device(),
                need_sync = false;

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
                        need_sync = true;
                        break;
                    }
                }
            }
            else
            {
                for (bool modified : flags_modified_gpu)
                {
                    if (modified)
                    {
                        need_sync = true;
                        break;
                    }
                }
            }

            if (need_sync)
            {
                auto dependencies = execute_pending_kernels();
                pending_kernels_dependencies_gpu = dependencies.first;
                pending_kernels_dependencies_cpu = dependencies.second;

                // std::cout << "Updating flags before aggregate.\nDeps gpu: "
                //     << pending_kernels_dependencies_gpu.size()
                //     << ", cpu: "
                //     << pending_kernels_dependencies_cpu.size()
                //     << std::endl;

                update_flags(on_device, gpu_allocator, cpu_allocator);

                dependencies = execute_pending_kernels(
                    #if USE_FUSION
                    false
                    #endif
                );
                #if USE_FUSION
                gpu_queue.wait_and_throw();
                cpu_queue.wait_and_throw();
                #else
                pending_kernels_dependencies_gpu = dependencies.first;
                pending_kernels_dependencies_cpu = dependencies.second;
                #endif
            }
            uint64_t *final_result = (on_device ?
                gpu_allocator.alloc<uint64_t>(1, true) :
                cpu_allocator.alloc<uint64_t>(1, true)
                );

            agg_bundles.reserve(input_segments.size());

            for (int i = 0; i < input_segments.size(); i++)
            {
                const Segment &input_segment = input_segments[i];
                KernelBundle bundle(on_device);
                bundle.add_kernel(
                    KernelData(
                        KernelType::AggregateOperationKernel,
                        input_segment.aggregate_operator(
                            (on_device ? flags_gpu : flags_host) + i * SEGMENT_SIZE,
                            on_device,
                            final_result
                        )
                    )
                );
                agg_bundles.push_back(bundle);
            }

            pending_kernels.push_back(agg_bundles);

            auto dependencies = execute_pending_kernels();
            pending_kernels_dependencies_gpu = dependencies.first;
            pending_kernels_dependencies_cpu = dependencies.second;

            bool *new_gpu_flags = gpu_allocator.alloc<bool>(1, true),
                *new_cpu_flags = cpu_allocator.alloc<bool>(1, true);

            pending_kernels_dependencies_gpu.push_back(gpu_queue.fill<bool>(new_gpu_flags, true, 1));
            new_cpu_flags[0] = true;

            flags_gpu = new_gpu_flags;
            flags_host = new_cpu_flags;

            flags_modified_gpu = { false };
            flags_modified_host = { false };

            nrows = 1;

            Column &result_column = materialized_columns.emplace_back(
                final_result,
                on_device,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator,
                nrows
            );

            current_columns.clear();
            current_columns.push_back(&result_column);
        }
        else
        {
            const Column *agg_column = current_columns[agg.operands[0]];
            bool on_device = agg_column->is_all_on_device(),
                need_flag_sync = false, need_data_sync = false;
            for (int i = 0; i < group.size() && on_device; i++)
                on_device = current_columns[group[i]]->is_all_on_device();

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
                        need_flag_sync = true;
                        break;
                    }
                }
            }
            else
            {
                for (bool modified : flags_modified_gpu)
                {
                    if (modified)
                    {
                        need_flag_sync = true;
                        break;
                    }
                }
            }

            need_data_sync = agg_column->needs_copy_on(on_device);
            for (int i = 0; i < group.size() && !need_data_sync; i++)
                need_data_sync = current_columns[group[i]]->needs_copy_on(on_device);


            if (need_data_sync || need_flag_sync)
            {
                // std::cout << "Executing flag/data sync kernels before aggregate.\nDeps gpu: "
                //     << pending_kernels_dependencies_gpu.size()
                //     << ", cpu: "
                //     << pending_kernels_dependencies_cpu.size()
                //     << std::endl;
                execute_pending_kernels();

                std::vector<Column *> columns_to_sync;
                if (need_data_sync)
                {
                    columns_to_sync.reserve(group.size() + 1);
                    columns_to_sync.push_back(const_cast<Column *>(agg_column));
                    for (int i = 0; i < group.size(); i++)
                        columns_to_sync.push_back(current_columns[group[i]]);
                }
                compress_and_sync(
                    gpu_allocator,
                    cpu_allocator,
                    columns_to_sync
                );
            }

            memory_manager &allocator = on_device ? gpu_allocator : cpu_allocator;

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

            uint64_t *aggregate_result = allocator.alloc<uint64_t>(prod_ranges, true);
            unsigned *temp_flags = allocator.alloc<unsigned>(prod_ranges, true);
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
                    contents[j] = segment.get_data(on_device);
                }
                const Segment &agg_segment = agg_segments[i];

                KernelBundle bundle(on_device);
                bundle.add_kernel(
                    KernelData(
                        KernelType::GroupByAggregateKernel,
                        agg_segment.group_by_aggregate_operator(
                            contents,
                            max,
                            min,
                            (on_device ? flags_gpu : flags_host) + i * SEGMENT_SIZE,
                            aggregate_result,
                            group.size(),
                            results,
                            temp_flags,
                            on_device,
                            prod_ranges
                        )
                    )
                );
                agg_bundles.push_back(bundle);
            }

            pending_kernels.push_back(agg_bundles);

            // std::cout << "Executing aggregate kernels" << std::endl;
            auto dependencies = execute_pending_kernels();

            bool *new_cpu_flags = cpu_allocator.alloc<bool>(prod_ranges, true),
                *new_gpu_flags = gpu_allocator.alloc<bool>(prod_ranges, true);

            if (on_device)
            {
                auto e1 = gpu_queue.submit(
                    [&](sycl::handler &cgh)
                    {
                        if (!dependencies.first.empty())
                            cgh.depends_on(dependencies.first);
                        cgh.parallel_for(
                            prod_ranges,
                            [=](sycl::id<1> idx)
                            {
                                new_gpu_flags[idx[0]] = temp_flags[idx[0]] != 0;
                            }
                        );
                    }
                );

                pending_kernels_dependencies_gpu.push_back(
                    gpu_queue.memcpy(
                        new_cpu_flags,
                        new_gpu_flags,
                        sizeof(bool) * prod_ranges,
                        e1
                    )
                );
            }
            else
            {
                pending_kernels_dependencies_cpu.push_back(
                    cpu_queue.submit(
                        [&](sycl::handler &cgh)
                        {
                            if (!dependencies.second.empty())
                                cgh.depends_on(dependencies.second);
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

                // e = gpu_queue.memcpy(
                //     new_gpu_flags,
                //     new_cpu_flags,
                //     sizeof(bool) * prod_ranges,
                //     e1
                // );
            }

            current_columns.clear();

            for (int i = 0; i < group.size(); i++)
            {
                Column &new_col = materialized_columns.emplace_back(
                    results[i],
                    on_device,
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
                on_device,
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

            uint64_t segment_num = nrows / SEGMENT_SIZE + (nrows % SEGMENT_SIZE > 0);
            flags_modified_gpu = std::vector<bool>(segment_num, false);
            flags_modified_host = std::vector<bool>(segment_num, false);
        }
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
            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Applying semi-join" << std::endl;
            #endif

            auto ht_data = right_table.build_keys_hash_table(
                right_column,
                gpu_allocator,
                cpu_allocator
            );
            auto ht_dependencies = right_table.execute_pending_kernels();

            bool *ht = std::get<0>(ht_data),
                *ht_other = nullptr;
            int build_min_value = std::get<1>(ht_data),
                build_max_value = std::get<2>(ht_data);

            bool ht_on_device = std::get<3>(ht_data);

            pending_kernels_dependencies_gpu.insert(
                pending_kernels_dependencies_gpu.end(),
                ht_dependencies.first.begin(),
                ht_dependencies.first.end()
            );
            pending_kernels_dependencies_cpu.insert(
                pending_kernels_dependencies_cpu.end(),
                ht_dependencies.second.begin(),
                ht_dependencies.second.end()
            );

            for (const Segment &seg : current_columns[left_column]->get_segments())
            {
                if (ht_on_device && !seg.is_on_device())
                {
                    memory_manager &allocator = ht_on_device ? cpu_allocator : gpu_allocator;

                    int ht_len = build_max_value - build_min_value + 1;

                    ht_other = allocator.alloc<bool>(ht_len, true);

                    pending_kernels_dependencies_cpu.push_back(
                        gpu_queue.memcpy(
                            ht_other,
                            ht,
                            ht_len * sizeof(bool),
                            ht_dependencies.first
                        )
                    );

                    break;
                }
            }

            pending_kernels.push_back(
                current_columns[left_column]->semi_join(
                    flags_gpu,
                    flags_host,
                    build_min_value,
                    build_max_value,
                    ht_on_device ? ht : ht_other,
                    ht_on_device ? ht_other : ht,
                    flags_modified_gpu,
                    flags_modified_host
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

            bool can_build_on_device = right_table.current_columns[right_column]->is_all_on_device() &&
                right_table.group_by_column->is_all_on_device();

            #if not PERFORMANCE_MEASUREMENT_ACTIVE
            std::cout << "Join hash table will be built on "
                << (can_build_on_device && probe_col_locations.first ? "GPU" : "CPU")
                << (can_build_on_device && probe_col_locations.first && probe_col_locations.second ? " and CPU" : "")
                << std::endl;
            #endif

            Column &new_column = materialized_columns.emplace_back(
                nrows,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator,
                true
            );

            auto min_max_gb = right_table.group_by_column->get_min_max();

            uint64_t group_by_col_index = current_columns.size() + right_table.group_by_column_index;

            for (int i = 0; i < right_table.current_columns.size(); i++)
                current_columns.push_back(nullptr);

            current_columns[group_by_col_index] = &materialized_columns[materialized_columns.size() - 1];

            int *ht = nullptr, build_min_value, build_max_value;

            if (can_build_on_device && probe_col_locations.first)
            {
                auto ht_data = right_table.build_key_vals_hash_table(
                    right_column,
                    true,
                    gpu_allocator,
                    cpu_allocator
                );

                auto ht_dependencies = right_table.execute_pending_kernels();

                pending_kernels_dependencies_gpu.insert(
                    pending_kernels_dependencies_gpu.end(),
                    ht_dependencies.first.begin(),
                    ht_dependencies.first.end()
                );
                pending_kernels_dependencies_cpu.insert(
                    pending_kernels_dependencies_cpu.end(),
                    ht_dependencies.second.begin(),
                    ht_dependencies.second.end()
                );

                ht = std::get<0>(ht_data);
                build_min_value = std::get<1>(ht_data);
                build_max_value = std::get<2>(ht_data);

                auto join_ops = current_columns[left_column]->full_join_operation(
                    flags_gpu,
                    build_min_value,
                    build_max_value,
                    min_max_gb.first,
                    min_max_gb.second,
                    ht,
                    new_column,
                    gpu_allocator,
                    gpu_queue,
                    cpu_queue,
                    true,
                    false,
                    flags_modified_gpu
                );

                pending_kernels.push_back(join_ops);
            }

            if (probe_col_locations.second || !(can_build_on_device && probe_col_locations.first))
            {
                int *ht_host;
                if (ht == nullptr)
                {
                    auto ht_data = right_table.build_key_vals_hash_table(
                        right_column,
                        false,
                        gpu_allocator,
                        cpu_allocator
                    );

                    auto ht_dependencies = right_table.execute_pending_kernels();

                    pending_kernels_dependencies_gpu.insert(
                        pending_kernels_dependencies_gpu.end(),
                        ht_dependencies.first.begin(),
                        ht_dependencies.first.end()
                    );
                    pending_kernels_dependencies_cpu.insert(
                        pending_kernels_dependencies_cpu.end(),
                        ht_dependencies.second.begin(),
                        ht_dependencies.second.end()
                    );

                    ht_host = std::get<0>(ht_data);
                    build_min_value = std::get<1>(ht_data);
                    build_max_value = std::get<2>(ht_data);
                }
                else
                {
                    ht_host = cpu_allocator.alloc<int>(
                        (build_max_value - build_min_value + 1) * 2,
                        true
                    );
                    pending_kernels_dependencies_cpu.push_back(
                        gpu_queue.memcpy(
                            ht_host,
                            ht,
                            (build_max_value - build_min_value + 1) * 2 * sizeof(int),
                            pending_kernels_dependencies_gpu
                        )
                    );
                }

                auto join_ops = current_columns[left_column]->full_join_operation(
                    flags_host,
                    build_min_value,
                    build_max_value,
                    min_max_gb.first,
                    min_max_gb.second,
                    ht_host,
                    new_column,
                    gpu_allocator,
                    gpu_queue,
                    cpu_queue,
                    false,
                    !(can_build_on_device && probe_col_locations.first),
                    flags_modified_host
                );

                pending_kernels.push_back(join_ops);
            }
        }
    }
};