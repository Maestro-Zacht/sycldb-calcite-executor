#pragma once

#include <sycl/sycl.hpp>

#include <fstream>

#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"
#include "../kernels/selection.hpp"
#include "../kernels/projection.hpp"
#include "../kernels/types.hpp"
#include "../kernels/join.hpp"
#include "../common.hpp"

#include "execution.hpp"


class Segment
{
private:
    int *data_host;
    std::vector<int *> device_ptrs;
    int min, max;
    uint64_t nrows;
    sycl::queue &cpu_queue;
    std::vector<sycl::queue> &device_queues;
    std::vector<bool> on_device_vec;
    bool on_device, is_aggregate_result, is_materialized, dirty_cache;
public:
    Segment(const int *init_data, sycl::queue &cpu_queue, std::vector<sycl::queue> &device_queues, uint64_t count = SEGMENT_SIZE)
        :
        device_ptrs(device_queues.size(), nullptr),
        nrows(count),
        cpu_queue(cpu_queue),
        device_queues(device_queues),
        on_device_vec(device_queues.size(), false),
        on_device(false),
        is_aggregate_result(false),
        is_materialized(false),
        dirty_cache(false)
    {
        if (count > SEGMENT_SIZE)
        {
            std::cerr << "Segment allocation failed: requested size " << count << " exceeds SEGMENT_SIZE " << SEGMENT_SIZE << std::endl;
            throw std::bad_alloc();
        }
        data_host = sycl::malloc_host<int>(count, cpu_queue);

        sycl::event e;
        if (init_data != nullptr)
            e = cpu_queue.memcpy(data_host, init_data, count * sizeof(int));
        else
        {
            std::cerr << "Error: Segment not initialized" << std::endl;
            throw std::runtime_error("Segment not initialized");
        }

        int *min_val = sycl::malloc_host<int>(1, cpu_queue);
        int *max_val = sycl::malloc_host<int>(1, cpu_queue);
        int *data = data_host;
        min_val[0] = init_data[0];
        max_val[0] = init_data[0];

        cpu_queue.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(e);
                cgh.parallel_for(
                    sycl::range<1>(count - 1),
                    sycl::reduction(max_val, sycl::maximum<int>()),
                    sycl::reduction(min_val, sycl::minimum<int>()),
                    [=](sycl::id<1> idx, auto &maxr, auto &minr)
                    {
                        auto j = idx[0] + 1;
                        int val = data[j];
                        maxr.combine(val);
                        minr.combine(val);
                    }
                );
            }
        ).wait();

        min = *min_val;
        max = *max_val;

        sycl::free(min_val, cpu_queue);
        sycl::free(max_val, cpu_queue);
    }

    Segment(
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        memory_manager &cpu_allocator,
        memory_manager &device_allocator,
        bool use_alloc_host = false,
        uint64_t count = SEGMENT_SIZE
    )
        :
        device_ptrs(device_queues.size(), nullptr),
        nrows(count),
        cpu_queue(cpu_queue),
        device_queues(device_queues),
        on_device_vec(device_queues.size(), false),
        on_device(false),
        is_aggregate_result(false),
        is_materialized(true),
        dirty_cache(false)
    {
        if (count > SEGMENT_SIZE)
        {
            std::cerr << "Segment allocation failed: requested size " << count << " exceeds SEGMENT_SIZE " << SEGMENT_SIZE << std::endl;
            throw std::bad_alloc();
        }

        if (use_alloc_host)
            data_host = device_allocator.alloc<int>(count, false);
        else
            data_host = cpu_allocator.alloc<int>(count, true);
    }

    Segment(
        uint64_t *init_data,
        bool on_device,
        int device_index,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators,
        uint64_t count = SEGMENT_SIZE
    )
        :
        device_ptrs(device_queues.size(), nullptr),
        min(0),
        max(0),
        nrows(count),
        cpu_queue(cpu_queue),
        device_queues(device_queues),
        on_device_vec(device_queues.size(), false),
        on_device(on_device),
        is_aggregate_result(true),
        is_materialized(true),
        dirty_cache(true)
    {
        if (count > SEGMENT_SIZE)
        {
            std::cerr << "Segment allocation failed: requested size " << count << " exceeds SEGMENT_SIZE " << SEGMENT_SIZE << std::endl;
            throw std::bad_alloc();
        }
        if (on_device && (device_index < 0 || device_index >= device_queues.size()))
        {
            std::cerr << "Segment allocation failed: invalid device index " << device_index << std::endl;
            throw std::bad_alloc();
        }

        if (on_device)
        {
            on_device_vec[device_index] = true;
            device_ptrs[device_index] = reinterpret_cast<int *>(init_data);
            data_host = reinterpret_cast<int *>(device_allocators[device_index].alloc<uint64_t>(count, false));
        }
        else
        {
            data_host = reinterpret_cast<int *>(init_data);
        }
    }

    Segment(
        int *init_data,
        bool on_device,
        int device_index,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators,
        uint64_t count = SEGMENT_SIZE
    )
        :
        device_ptrs(device_queues.size(), nullptr),
        min(0),
        max(0),
        nrows(count),
        cpu_queue(cpu_queue),
        device_queues(device_queues),
        on_device_vec(device_queues.size(), false),
        on_device(on_device),
        is_aggregate_result(false),
        is_materialized(true),
        dirty_cache(true)
    {
        if (count > SEGMENT_SIZE)
        {
            std::cerr << "Segment allocation failed: requested size " << count << " exceeds SEGMENT_SIZE " << SEGMENT_SIZE << std::endl;
            throw std::bad_alloc();
        }
        if (on_device && (device_index < 0 || device_index >= device_queues.size()))
        {
            std::cerr << "Segment allocation failed: invalid device index " << device_index << std::endl;
            throw std::bad_alloc();
        }

        if (on_device)
        {
            on_device_vec[device_index] = true;
            device_ptrs[device_index] = init_data;
            data_host = device_allocators[device_index].alloc<int>(count, false);
        }
        else
        {
            data_host = init_data;
        }
    }

    ~Segment()
    {
        if (!is_materialized)
        {
            if (data_host != nullptr)
            {
                // std::cout << "Freeing data_host " << data_host << std::endl;
                sycl::free(data_host, cpu_queue);
            }
            for (size_t i = 0; i < device_ptrs.size(); i++)
            {
                if (device_ptrs[i] != nullptr)
                {
                    // std::cout << "Freeing data_device " << device_ptrs[i] << " on device " << i << std::endl;
                    sycl::free(device_ptrs[i], device_queues[i]);
                }
            }
            // std::cout << "Segment free completed" << std::endl;
        }
    }

    bool is_on_device() const { return on_device; }
    bool is_on_device(int device_index) const
    {
        return on_device
            && device_index >= 0 && device_index < on_device_vec.size()
            && on_device_vec[device_index];
    }

    const std::vector<bool> &get_on_device_vec() const { return on_device_vec; }
    int get_min() const { return min; }
    int get_max() const { return max; }
    uint64_t get_nrows() const { return nrows; }

    int get_device_index() const
    {
        for (size_t i = 0; i < on_device_vec.size(); i++)
        {
            if (on_device_vec[i])
                return i;
        }
        return -1;
    }

    FillKernel *fill_with_literal(int literal, bool fill_on_device, int device_index)
    {
        if (!is_materialized)
        {
            std::cerr << "Cannot fill non-materialized segment" << std::endl;
            throw std::runtime_error("Cannot fill non-materialized segment");
        }
        if (is_aggregate_result)
        {
            std::cerr << "Cannot fill aggregate result segment with int literal" << std::endl;
            throw std::runtime_error("Cannot fill aggregate result segment with int literal");
        }
        if (fill_on_device && !on_device)
        {
            std::cerr << "Cannot fill on device a segment that is on host only" << std::endl;
            throw std::runtime_error("Cannot fill on device a segment that is on host only");
        }
        if (fill_on_device && (device_index < 0 || device_index >= device_queues.size()))
        {
            std::cerr << "Invalid device index " << device_index << " for fill_with_literal" << std::endl;
            throw std::runtime_error("Invalid device index for fill_with_literal");
        }

        min = literal;
        max = literal;

        dirty_cache = fill_on_device;

        return new FillKernel(
            fill_on_device ? device_ptrs[device_index] : data_host,
            literal,
            nrows
        );
    }

    void build_on_device(memory_manager &device_allocator, int device_index)
    {
        if (!is_materialized)
        {
            std::cerr << "Segment build_on_device: cannot build non-materialized segment on device" << std::endl;
            throw std::runtime_error("Segment build_on_device: cannot build non-materialized segment on device");
        }
        if (on_device && on_device_vec[device_index])
            return;

        on_device = true;
        on_device_vec[device_index] = true;
        dirty_cache = true;
        device_ptrs[device_index] = device_allocator.alloc<int>(nrows, true);
    }

    void set_min(int value)
    {
        dirty_cache = true;
        min = value;
    }

    void set_max(int value)
    {
        dirty_cache = true;
        max = value;
    }

    const int *get_data(bool device, int device_index) const
    {
        if (device && !on_device)
        {
            std::cerr << "Segment data requested on device but it is on host" << std::endl;
            throw std::runtime_error("Segment data requested on device but it is on host");
        }
        if (is_aggregate_result)
        {
            std::cerr << "Segment data requested but it is an aggregate result" << std::endl;
            throw std::runtime_error("Segment data requested but it is an aggregate result");
        }
        if (device && (device_index < 0 || device_index >= device_queues.size()))
        {
            std::cerr << "Invalid device index " << device_index << " for get_data" << std::endl;
            throw std::runtime_error("Invalid device index for get_data");
        }
        return device ? device_ptrs[device_index] : data_host;
    }

    int *get_data(bool device, int device_index)
    {
        dirty_cache = true;
        return const_cast<int *>(static_cast<const Segment &>(*this).get_data(device, device_index));
    }

    const uint64_t *get_aggregate_data(bool device, int device_index) const
    {
        if (device && !on_device)
        {
            std::cerr << "Segment data requested on device but it is on host" << std::endl;
            throw std::runtime_error("Segment data requested on device but it is on host");
        }
        if (!is_aggregate_result)
        {
            std::cerr << "Segment aggregate data requested but it is not an aggregate result" << std::endl;
            throw std::runtime_error("Segment aggregate data requested but it is not an aggregate result");
        }
        if (device && (device_index < 0 || device_index >= device_queues.size()))
        {
            std::cerr << "Invalid device index " << device_index << " for get_aggregate_data" << std::endl;
            throw std::runtime_error("Invalid device index for get_aggregate_data");
        }
        return reinterpret_cast<uint64_t *>(device ? device_ptrs[device_index] : data_host);
    }

    uint64_t *get_aggregate_data(bool device, int device_index)
    {
        dirty_cache = true;
        return const_cast<uint64_t *>(static_cast<const Segment &>(*this).get_aggregate_data(device, device_index));
    }

    const int &operator[](uint64_t index) const
    {
        if (index >= nrows)
        {
            std::cerr << "Segment index out of range: " << index << " >= " << nrows << std::endl;
            throw std::out_of_range("Segment index out of range");
        }
        if (is_aggregate_result)
        {
            std::cerr << "Segment operator[] called on aggregate result segment" << std::endl;
            throw std::runtime_error("wrong operator[]");
        }

        const_cast<Segment &>(*this).copy_on_host().wait();

        return data_host[index];
    }

    const uint64_t &get_aggregate_value(uint64_t index) const
    {
        if (index >= nrows)
            throw std::out_of_range("Segment index out of range");
        if (!is_aggregate_result)
            throw std::runtime_error("wrong get_aggregate_value");

        const_cast<Segment &>(*this).copy_on_host().wait();

        return reinterpret_cast<uint64_t *>(data_host)[index];
    }

    uint64_t get_data_size(bool gpu_only, int device_index) const
    {
        if (gpu_only &&
            (!on_device || device_index < 0 || device_index >= device_queues.size() || !on_device_vec[device_index]))
            return 0;
        return nrows * (is_aggregate_result ? sizeof(uint64_t) : sizeof(int));
    }

    sycl::event move_to_device(int device_index)
    {
        if (is_materialized || is_aggregate_result)
        {
            std::cerr << "Segment move_to_device: cannot move materialized or aggregate result segment to device" << std::endl;
            throw std::runtime_error("Segment move_to_device: cannot move materialized or aggregate result segment to device");
        }
        if (device_index < 0 || device_index >= device_queues.size())
        {
            std::cerr << "Segment move_to_device: invalid device index " << device_index << std::endl;
            throw std::runtime_error("Segment move_to_device: invalid device index");
        }
        if (on_device && on_device_vec[device_index])
            return sycl::event();

        if (device_ptrs[device_index] == nullptr)
            device_ptrs[device_index] = sycl::malloc_device<int>(nrows, device_queues[device_index]);

        on_device = true;
        on_device_vec[device_index] = true;
        return device_queues[device_index].memcpy(device_ptrs[device_index], data_host, nrows * sizeof(int));
    }

    sycl::event copy_on_host()
    {
        sycl::event e;

        if (on_device && dirty_cache && is_materialized)
        {
            for (int i = 0; i < device_queues.size(); i++)
            {
                if (on_device_vec[i])
                {
                    if (is_aggregate_result)
                        e = device_queues[i].memcpy(
                            reinterpret_cast<uint64_t *>(data_host),
                            reinterpret_cast<uint64_t *>(device_ptrs[i]),
                            nrows * sizeof(uint64_t)
                        );
                    else
                        e = device_queues[i].memcpy(data_host, device_ptrs[i], nrows * sizeof(int));
                    break;
                }
            }
            dirty_cache = false;
        }
        else
            e = sycl::event();

        return e;
    }

    bool needs_copy_on(bool device, int device_index) const
    {
        return (device != on_device || (device && !on_device_vec[device_index]))
            && is_materialized && dirty_cache;
    }

    KernelBundle search_operator(
        const ExprType &expr,
        std::string parent_op,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators,
        bool *cpu_flags,
        std::vector<bool *> &device_flags) const
    {
        int device_index = -1;
        if (on_device)
        {
            device_index = get_device_index();
            if (device_index == -1)
            {
                std::cerr << "Segment search_operator: no device found for on_device segment" << std::endl;
                throw std::runtime_error("Segment search_operator: no device found for on_device segment");
            }
        }
        memory_manager &allocator = on_device ? device_allocators[device_index] : cpu_allocator;
        int *data = on_device ? device_ptrs[device_index] : data_host;
        bool *flags = on_device ? device_flags[device_index] : cpu_flags;

        bool *local_flags = allocator.alloc<bool>(nrows, true);

        KernelBundle operations(on_device, device_index);

        if (expr.operands[1].literal.rangeSet.size() == 1) // range
        {
            int lower = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                upper = std::stoi(expr.operands[1].literal.rangeSet[0][2]);

            operations.add_kernel(
                KernelData(
                    KernelType::SelectionKernelLiteral,
                    new SelectionKernelLiteral(
                        local_flags,
                        data,
                        ">=",
                        lower,
                        "NONE",
                        nrows
                    )
                )
            );
            operations.add_kernel(
                KernelData(
                    KernelType::SelectionKernelLiteral,
                    new SelectionKernelLiteral(
                        local_flags,
                        data,
                        "<=",
                        upper,
                        "AND",
                        nrows
                    )
                )
            );

            // TODO: min and max here
            // table_data.columns[col_index].min_value = lower;
            // table_data.columns[col_index].max_value = upper;
        }
        else // or between two values
        {
            int first = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                second = std::stoi(expr.operands[1].literal.rangeSet[1][1]);

            operations.add_kernel(
                KernelData(
                    KernelType::SelectionKernelLiteral,
                    new SelectionKernelLiteral(
                        local_flags,
                        data,
                        "==",
                        first,
                        "NONE",
                        nrows
                    )
                )
            );
            operations.add_kernel(
                KernelData(
                    KernelType::SelectionKernelLiteral,
                    new SelectionKernelLiteral(
                        local_flags,
                        data,
                        "==",
                        second,
                        "OR",
                        nrows
                    )
                )
            );
        }

        logical_op logic = get_logical_op(parent_op);

        operations.add_kernel(
            KernelData(
                KernelType::LogicalKernel,
                new LogicalKernel(
                    logic,
                    flags,
                    local_flags,
                    nrows
                )
            )
        );

        return operations;
    }

    SelectionKernelLiteral *filter_operator(
        std::string op,
        std::string parent_op,
        int literal_value,
        bool *flags,
        int device_index) const
    {
        return new SelectionKernelLiteral(
            flags,
            on_device ? device_ptrs[device_index] : data_host,
            op,
            literal_value,
            parent_op,
            nrows
        );
    }

    SelectionKernelColumns *filter_operator(
        std::string op,
        std::string parent_op,
        const Segment &other_segment,
        bool *flags,
        int device_index) const
    {
        return new SelectionKernelColumns(
            flags,
            on_device ? device_ptrs[device_index] : data_host,
            op,
            other_segment.get_data(on_device, device_index),
            parent_op,
            nrows
        );
    }

    PerformOperationKernelColumns *perform_operator(
        const Segment &first_operand,
        const Segment &second_operand,
        bool perform_on_device,
        int device_index,
        const bool *flags,
        const std::string &op)
    {
        if (perform_on_device && !on_device)
        {
            std::cerr << "Perform operation: Mismatched segment locations between columns" << std::endl;
            throw std::runtime_error("Perform operation: Mismatched segment locations between columns");
        }

        min = std::min(first_operand.min, second_operand.min);
        max = std::max(first_operand.max, second_operand.max);

        dirty_cache = true;

        return new PerformOperationKernelColumns(
            perform_on_device ? device_ptrs[device_index] : data_host,
            first_operand.get_data(perform_on_device, device_index),
            second_operand.get_data(perform_on_device, device_index),
            flags,
            op,
            nrows
        );
    }

    PerformOperationKernelLiteralSecond *perform_operator(
        const Segment &first_operand,
        int second_operand,
        bool perform_on_device,
        int device_index,
        const bool *flags,
        const std::string &op)
    {
        if (perform_on_device && !on_device)
        {
            std::cerr << "Perform operation: Mismatched segment locations between columns" << std::endl;
            throw std::runtime_error("Perform operation: Mismatched segment locations between columns");
        }

        min = first_operand.min;
        max = first_operand.max;

        dirty_cache = true;

        return new PerformOperationKernelLiteralSecond(
            perform_on_device ? device_ptrs[device_index] : data_host,
            first_operand.get_data(perform_on_device, device_index),
            second_operand,
            flags,
            op,
            nrows
        );
    }

    PerformOperationKernelLiteralFirst *perform_operator(
        int first_operand,
        const Segment &second_operand,
        bool perform_on_device,
        int device_index,
        const bool *flags,
        const std::string &op)
    {
        if (perform_on_device && !on_device)
        {
            std::cerr << "Perform operation: Mismatched segment locations between columns" << std::endl;
            throw std::runtime_error("Perform operation: Mismatched segment locations between columns");
        }

        min = second_operand.min;
        max = second_operand.max;

        dirty_cache = true;

        return new PerformOperationKernelLiteralFirst(
            perform_on_device ? device_ptrs[device_index] : data_host,
            first_operand,
            second_operand.get_data(perform_on_device, device_index),
            flags,
            op,
            nrows
        );
    }

    AggregateOperationKernel *aggregate_operator(
        const bool *flags,
        bool aggregate_on_device,
        int device_index,
        uint64_t *agg_res) const
    {
        return new AggregateOperationKernel(
            aggregate_on_device ? device_ptrs[device_index] : data_host,
            flags,
            nrows,
            agg_res
        );
    }

    BuildKeysHTKernel *build_keys_hash_table(
        bool *ht,
        const bool *flags,
        int ht_len,
        int ht_min_value,
        int device_index
    ) const
    {
        return new BuildKeysHTKernel(
            ht,
            on_device ? device_ptrs[device_index] : data_host,
            flags,
            ht_len,
            ht_min_value,
            nrows
        );
    }

    FilterJoinKernel *semi_join_operator(
        bool *probe_flags,
        int build_min_value,
        int build_max_value,
        const bool *build_ht,
        int device_index) const
    {
        return new FilterJoinKernel(
            on_device ? device_ptrs[device_index] : data_host,
            probe_flags,
            build_ht,
            build_min_value,
            build_max_value,
            nrows
        );
    }

    BuildKeyValsHTKernel *build_key_vals_hash_ht(
        int *ht,
        const bool *flags,
        int ht_len,
        int ht_min_value,
        bool build_on_device,
        int device_index,
        const Segment &value_segment) const
    {
        if (build_on_device &&
            (!on_device || !on_device_vec[device_index]
                || !value_segment.on_device || !value_segment.on_device_vec[device_index]))
        {
            std::cerr << "Build key-vals hash table: Mismatched segment locations between columns" << std::endl;
            throw std::runtime_error("Build key-vals hash table: Mismatched segment locations between columns");
        }

        return new BuildKeyValsHTKernel(
            ht,
            build_on_device ? device_ptrs[device_index] : data_host,
            build_on_device ? value_segment.device_ptrs[device_index] : value_segment.data_host,
            flags,
            ht_len,
            ht_min_value,
            nrows
        );
    }

    FullJoinKernel *full_join_operator(
        Segment &result_segment,
        bool *probe_flags,
        const int *ht,
        int ht_min_value,
        int ht_max_value,
        bool ht_on_device,
        int device_index
    ) const
    {
        if (ht_on_device &&
            (!on_device || !on_device_vec[device_index]
                || !result_segment.on_device || !result_segment.on_device_vec[device_index]))
        {
            std::cerr << "Full join: Mismatched segment locations between columns" << std::endl;
            throw std::runtime_error("Full join: Mismatched segment locations between columns");
        }

        return new FullJoinKernel(
            ht_on_device ? device_ptrs[device_index] : data_host,
            ht_on_device ? result_segment.device_ptrs[device_index] : result_segment.data_host,
            probe_flags,
            ht,
            ht_min_value,
            ht_max_value,
            nrows
        );
    }

    GroupByAggregateKernel *group_by_aggregate_operator(
        const int **contents,
        const int *max,
        const int *min,
        const bool *flags,
        uint64_t *agg_res,
        int group_size,
        int **results,
        unsigned *result_flags,
        bool aggregate_on_device,
        int device_index,
        uint64_t prod_ranges
    ) const
    {
        return new GroupByAggregateKernel(
            contents,
            aggregate_on_device ? device_ptrs[device_index] : data_host,
            max,
            min,
            flags,
            group_size,
            nrows,
            results,
            agg_res,
            result_flags,
            prod_ranges
        );
    }


    // assumption: sync point. needs to be improved
    sycl::event compress_sync(
        int *row_ids_device,
        int *row_ids_host,
        sycl::event &e_row_ids_host,
        uint64_t nrows_selected,
        memory_manager &device_allocator,
        int device_index)
    {
        if (is_aggregate_result)
        {
            std::cerr << "Compress operator: cannot compress aggregate result segment" << std::endl;
            throw std::runtime_error("Compress operator: cannot compress aggregate result segment");
        }
        if (!on_device || !on_device_vec[device_index])
        {
            std::cerr << "Compress operator: segment not on device" << std::endl;
            throw std::runtime_error("Compress operator: segment not on device");
        }

        int *data_device_compressed = device_allocator.alloc<int>(nrows_selected, true);
        int *data_host_compressed = device_allocator.alloc<int>(nrows_selected, false);
        int *device_ptr = device_ptrs[device_index];
        int *host_ptr = data_host;

        auto e1 = device_queues[device_index].submit(
            [&](sycl::handler &cgh)
            {
                cgh.parallel_for(
                    nrows_selected,
                    [=](sycl::id<1> idx)
                    {
                        auto i = idx[0];
                        int row_id = row_ids_device[i];
                        data_device_compressed[i] = device_ptr[row_id];
                    }
                );
            }
        );

        auto e2 = device_queues[device_index].memcpy(
            data_host_compressed,
            data_device_compressed,
            nrows_selected * sizeof(int),
            e1
        );

        auto e3 = cpu_queue.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(e2);
                cgh.depends_on(e_row_ids_host);
                cgh.parallel_for(
                    nrows_selected,
                    [=](sycl::id<1> idx)
                    {
                        auto i = idx[0];
                        int row_id = row_ids_host[i];
                        host_ptr[row_id] = data_host_compressed[i];
                    }
                );
            }
        );

        dirty_cache = false;

        return e3;
    }
};


class Column
{
private:
    std::vector<Segment> segments;
    bool is_aggregate_result;
public:
    Column() : is_aggregate_result(false)
    {
        std::cerr << "Warning: Empty column created" << std::endl;
    }

    Column(const int *init_data, uint64_t nrows, sycl::queue &cpu_queue, std::vector<sycl::queue> &device_queues)
        : is_aggregate_result(false)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(init_data + i * SEGMENT_SIZE, cpu_queue, device_queues);

        if (remainder > 0)
            segments.emplace_back(init_data + full_segments * SEGMENT_SIZE, cpu_queue, device_queues, remainder);
    }

    Column(
        uint64_t nrows,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        memory_manager &cpu_allocator,
        memory_manager &device_allocator,
        bool use_alloc_host = false)
        : is_aggregate_result(false)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE,
            remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(cpu_queue, device_queues, cpu_allocator, device_allocator, use_alloc_host);

        if (remainder > 0)
            segments.emplace_back(cpu_queue, device_queues, cpu_allocator, device_allocator, use_alloc_host, remainder);
    }

    Column(
        uint64_t *init_data,
        bool on_device,
        int device_index,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators,
        uint64_t nrows)
        : is_aggregate_result(true)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(
                init_data + i * SEGMENT_SIZE,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators
            );

        if (remainder > 0)
            segments.emplace_back(
                init_data + full_segments * SEGMENT_SIZE,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                remainder
            );
    }

    Column(
        int *init_data,
        bool on_device,
        int device_index,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators,
        uint64_t nrows)
        : is_aggregate_result(false)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(
                init_data + i * SEGMENT_SIZE,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators
            );

        if (remainder > 0)
            segments.emplace_back(
                init_data + full_segments * SEGMENT_SIZE,
                on_device,
                device_index,
                cpu_queue,
                device_queues,
                cpu_allocator,
                device_allocators,
                remainder
            );
    }

    const std::vector<Segment> &get_segments() const { return segments; }
    std::vector<Segment> &get_segments() { return segments; }
    bool get_is_aggregate_result() const { return is_aggregate_result; }

    bool is_all_on_same_device() const
    {
        int device_index = segments[0].get_device_index();
        for (const auto &seg : segments)
        {
            if (!seg.is_on_device() || seg.get_device_index() != device_index)
                return false;
        }
        return true;
    }

    std::vector<bool> get_full_col_on_device() const
    {
        std::vector<bool> on_device_vec(segments[0].get_on_device_vec());
        for (const auto &seg : segments)
        {
            const std::vector<bool> &seg_on_device_vec = seg.get_on_device_vec();
            for (int i = 0; i < seg_on_device_vec.size(); i++)
                on_device_vec[i] = on_device_vec[i] && seg_on_device_vec[i];
        }
        return on_device_vec;
    }

    std::pair<bool, std::vector<bool>> get_positions() const
    {
        bool has_host = false;
        std::vector<bool> has_device;
        for (const auto &seg : segments)
        {
            const std::vector<bool> &on_device_vec = seg.get_on_device_vec();
            has_device.resize(on_device_vec.size(), false);
            if (seg.is_on_device())
            {
                for (int i = 0; i < on_device_vec.size(); i++)
                    has_device[i] = has_device[i] || on_device_vec[i];
            }
            else
                has_host = true;
        }
        return { has_host, has_device };
    }

    std::pair<int, int> get_min_max() const
    {
        if (is_aggregate_result)
            throw std::runtime_error("can't get min/max of aggregate result column");

        int overall_min = segments[0].get_min();
        int overall_max = segments[0].get_max();

        for (int i = 1; i < segments.size(); i++)
        {
            overall_min = std::min(overall_min, segments[i].get_min());
            overall_max = std::max(overall_max, segments[i].get_max());
        }

        return { overall_min, overall_max };
    }

    const int &operator[](uint64_t index) const
    {
        if (is_aggregate_result)
        {
            std::cerr << "wrong operator[]" << std::endl;
            throw std::runtime_error("wrong operator[]");
        }
        uint64_t segment_index = index / SEGMENT_SIZE;
        uint64_t offset = index % SEGMENT_SIZE;
        return segments[segment_index][offset];
    }

    const uint64_t &get_aggregate_value(uint64_t index) const
    {
        if (!is_aggregate_result)
        {
            std::cerr << "wrong get_aggregate_value" << std::endl;
            throw std::runtime_error("wrong get_aggregate_value");
        }
        uint64_t segment_index = index / SEGMENT_SIZE;
        uint64_t offset = index % SEGMENT_SIZE;
        return segments[segment_index].get_aggregate_value(offset);
    }

    std::vector<KernelBundle> fill_with_literal(int literal, bool fill_on_device, int device_index, memory_manager &device_allocator)
    {
        std::vector<KernelBundle> operations;
        operations.reserve(segments.size());

        for (auto &seg : segments)
        {
            KernelBundle bundle(fill_on_device, device_index);

            if (fill_on_device)
                seg.build_on_device(device_allocator, device_index);

            bundle.add_kernel(
                KernelData(
                    KernelType::FillKernel,
                    seg.fill_with_literal(literal, fill_on_device, device_index)
                )
            );
            operations.push_back(std::move(bundle));
        }

        return operations;
    }

    void move_all_to_device(int device_index)
    {
        for (auto &seg : segments)
            seg.move_to_device(device_index);
    }

    void move_to_device(int device_index, const std::vector<bool> &segments_choices = {})
    {
        for (int i = 0; i < segments.size(); i++)
        {
            if (segments_choices.size() <= i || segments_choices[i])
                segments[i].move_to_device(device_index);
        }
    }

    uint64_t get_data_size(bool gpu_only, int device_index) const
    {
        uint64_t total_size = 0;
        for (const auto &seg : segments)
            total_size += seg.get_data_size(gpu_only, device_index);
        return total_size;
    }

    bool needs_copy_on(bool device, int device_index) const
    {
        for (const auto &seg : segments)
            if (seg.needs_copy_on(device, device_index))
                return true;
        return false;
    }

    std::tuple<bool *, int, int, std::vector<KernelBundle>> build_keys_hash_table(bool *flags, memory_manager &allocator, bool on_device, int device_index) const
    {
        std::vector<KernelBundle> ops;
        ops.reserve(segments.size());

        auto min_max = get_min_max();
        int min_value = min_max.first;
        int max_value = min_max.second;

        int ht_len = max_value - min_value + 1;

        bool *ht = allocator.alloc_zero<bool>(ht_len);

        for (int i = 0; i < segments.size(); i++)
        {
            const Segment &seg = segments[i];
            if (on_device && !seg.is_on_device(device_index))
            {
                std::cerr << "Build keys hash table: segment " << i << " is not on device " << device_index << std::endl;
                throw std::runtime_error("Build keys hash table: segment is not on device");
            }
            KernelBundle bundle(on_device, device_index);
            bundle.add_kernel(
                KernelData(
                    KernelType::BuildKeysHTKernel,
                    seg.build_keys_hash_table(
                        ht,
                        flags + i * SEGMENT_SIZE,
                        ht_len,
                        min_value,
                        device_index
                    )
                )
            );
            ops.push_back(bundle);
        }

        return { ht, min_value, max_value, ops };
    }

    std::vector<KernelBundle> semi_join(
        bool *probe_flags_cpu,
        std::vector<bool *> &probe_flags_devices,
        int build_min_value,
        int build_max_value,
        const bool *ht_cpu,
        const std::vector<bool *> &ht_devices,
        std::vector<bool> &flags_modified_host,
        std::vector<std::vector<bool>> &flags_modified_devices
    ) const
    {
        std::vector<KernelBundle> ops;
        ops.reserve(segments.size());

        for (int i = 0; i < segments.size(); i++)
        {
            const Segment &seg = segments[i];
            int device_index = seg.get_device_index();
            bool on_device = seg.is_on_device() && ht_devices[device_index] != nullptr;
            KernelBundle bundle(on_device, device_index);

            bundle.add_kernel(
                KernelData(
                    KernelType::FilterJoinKernel,
                    seg.semi_join_operator(
                        (on_device ? probe_flags_devices[device_index] : probe_flags_cpu) + i * SEGMENT_SIZE,
                        build_min_value,
                        build_max_value,
                        on_device ? ht_devices[device_index] : ht_cpu,
                        device_index
                    )
                )
            );
            ops.push_back(bundle);

            if (on_device)
                flags_modified_devices[device_index][i] = true;
            else
                flags_modified_host[i] = true;
        }

        return ops;
    }

    std::tuple<int *, int, int, std::vector<KernelBundle>> build_key_vals_hash_table(
        const Column *vals_column,
        bool *flags,
        memory_manager &allocator,
        bool on_device,
        int device_index) const
    {
        std::vector<KernelBundle> ops;
        ops.reserve(segments.size());

        auto min_max = get_min_max();
        int min_value = min_max.first;
        int max_value = min_max.second;

        int ht_len = max_value - min_value + 1;

        int *ht = allocator.alloc_zero<int>(ht_len * 2);

        for (int i = 0; i < segments.size(); i++)
        {
            KernelBundle bundle(on_device, device_index);
            bundle.add_kernel(
                KernelData(
                    KernelType::BuildKeyValsHTKernel,
                    segments[i].build_key_vals_hash_ht(
                        ht,
                        flags + i * SEGMENT_SIZE,
                        ht_len,
                        min_value,
                        on_device,
                        device_index,
                        vals_column->segments[i]
                    )
                )
            );
            ops.push_back(bundle);
        }

        return { ht, min_value, max_value, ops };
    }

    std::vector<KernelBundle> full_join_operation(
        bool *probe_flags_host,
        std::vector<bool *> &probe_flags_devices,
        int build_min_value,
        int build_max_value,
        int group_by_column_min,
        int group_by_column_max,
        const int *build_ht_host,
        const std::vector<int *> &build_hts_devices,
        Column &new_column,
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        memory_manager &cpu_allocator,
        std::vector<memory_manager> &device_allocators,
        std::vector<bool> &flags_modified_host,
        std::vector<std::vector<bool>> &flags_modified_devices) const
    {
        std::vector<KernelBundle> ops;
        ops.reserve(segments.size());

        for (int i = 0; i < segments.size(); i++)
        {
            const Segment &seg = segments[i];
            Segment &new_seg = new_column.segments[i];
            bool on_device = seg.is_on_device(), built = false;
            const std::vector<bool> &on_device_vec = seg.get_on_device_vec();

            if (on_device)
            {
                for (int d = 0; d < on_device_vec.size(); d++)
                {
                    if (on_device_vec[d] && build_hts_devices[d] != nullptr)
                    {
                        KernelBundle bundle(on_device, d);

                        new_seg.build_on_device(device_allocators[d], d);

                        bundle.add_kernel(
                            KernelData(
                                KernelType::FullJoinKernel,
                                seg.full_join_operator(
                                    new_seg,
                                    probe_flags_devices[d] + i * SEGMENT_SIZE,
                                    build_hts_devices[d],
                                    build_min_value,
                                    build_max_value,
                                    on_device,
                                    d
                                )
                            )
                        );
                        new_seg.set_min(group_by_column_min);
                        new_seg.set_max(group_by_column_max);
                        flags_modified_devices[d][i] = true;
                        ops.push_back(bundle);
                        built = true;
                        break;
                    }
                }
            }

            if (!built)
            {
                KernelBundle bundle(false, -1);

                // new_seg.build_on_device(device_allocators[d], d);

                bundle.add_kernel(
                    KernelData(
                        KernelType::FullJoinKernel,
                        seg.full_join_operator(
                            new_seg,
                            probe_flags_host + i * SEGMENT_SIZE,
                            build_ht_host,
                            build_min_value,
                            build_max_value,
                            false,
                            -1
                        )
                    )
                );
                new_seg.set_min(group_by_column_min);
                new_seg.set_max(group_by_column_max);
                flags_modified_host[i] = true;
                ops.push_back(bundle);
            }
        }

        return ops;
    }
};


class Table
{
private:
    std::string table_name;
    std::vector<Column> columns;
    uint64_t nrows;
public:
    Table(const std::string table_name, sycl::queue &cpu_queue, std::vector<sycl::queue> &device_queues)
        : table_name(table_name)
    {
        int col_number = table_column_numbers[table_name], *content;
        columns.reserve(col_number);

        const std::set<int> &columns_needed = table_column_indices[table_name];

        for (int i = 0; i < col_number; i++)
        {
            std::string col_name = table_name + std::to_string(i);
            std::transform(col_name.begin(), col_name.end(), col_name.begin(), ::toupper);

            std::string filename = DATA_DIR + col_name;
            std::ifstream colData(filename.c_str(), std::ios::in | std::ios::binary);

            colData.seekg(0, std::ios::end);
            std::streampos fileSize = colData.tellg();
            uint64_t num_entries = static_cast<uint64_t>(fileSize / sizeof(int));

            if (i == 0)
            {
                nrows = num_entries;
                content = sycl::malloc_host<int>(nrows, cpu_queue);
            }

            if (num_entries != nrows)
            {
                // throw std::runtime_error("Column length mismatch in " + filename + ": expected " + std::to_string(nrows) + ", got " + std::to_string(num_entries));
                std::cerr << "Warning: Column length mismatch in " << filename << ": expected " << nrows << ", got " << num_entries << std::endl;
                columns.emplace_back();
            }
            else if (columns_needed.find(i) == columns_needed.end())
            {
                // Skip loading this column
                std::cout << "Skipping loading column " << col_name << std::endl;
                columns.emplace_back();
            }
            else
            {
                colData.seekg(0, std::ios::beg);
                colData.read((char *)content, num_entries * sizeof(int));
                columns.emplace_back(content, num_entries, cpu_queue, device_queues);
            }

            colData.close();
        }
        sycl::free(content, cpu_queue);
    }

    uint64_t get_nrows() const { return nrows; }
    const std::vector<Column> &get_columns() const { return columns; }
    const std::string &get_name() const { return table_name; }

    uint64_t get_data_size(bool gpu_only, int device_index) const
    {
        uint64_t total_size = 0;
        for (const auto &col : columns)
            total_size += col.get_data_size(gpu_only, device_index);
        return total_size;
    }

    void move_all_to_device(int device_index)
    {
        for (auto &col : columns)
            col.move_all_to_device(device_index);
    }

    void move_column_to_device(int col_index, int device_index, const std::vector<bool> &segments = {})
    {
        if (segments.empty())
            columns[col_index].move_all_to_device(device_index);
        else
            columns[col_index].move_to_device(device_index, segments);
    }

    uint64_t num_segments() const
    {
        return columns[4].get_segments().size();
    }
};