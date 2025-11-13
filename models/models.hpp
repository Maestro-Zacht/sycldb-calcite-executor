#pragma once

#include <sycl/sycl.hpp>

#include <fstream>

#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"
#include "../kernels/selection.hpp"
#include "../kernels/projection.hpp"
#include "../kernels/types.hpp"
#include "../kernels/join.hpp"
#include "../operations/load.hpp"

#include "execution.hpp"

#define SEGMENT_SIZE (((uint64_t)1) << 22)


class Segment
{
private:
    int *data_host, *data_device;
    int min, max;
    uint64_t nrows;
    sycl::queue gpu_queue, cpu_queue;
    bool on_device, is_aggregate_result, is_materialized, dirty_cache;
public:
    Segment(const int *init_data, sycl::queue &gpu_queue, sycl::queue &cpu_queue, uint64_t count = SEGMENT_SIZE)
        :
        data_device(nullptr),
        nrows(count),
        gpu_queue(gpu_queue),
        cpu_queue(cpu_queue),
        on_device(false),
        is_aggregate_result(false),
        is_materialized(false),
        dirty_cache(false)
    {
        if (count > SEGMENT_SIZE)
            throw std::bad_alloc();

        data_host = sycl::malloc_host<int>(count, cpu_queue);

        sycl::event e;
        if (init_data != nullptr)
            e = cpu_queue.memcpy(data_host, init_data, count * sizeof(int));
        else
        {
            std::cerr << "Warning: Segment not initialized" << std::endl;
            return;
        }

        int *min_val = sycl::malloc_host<int>(1, gpu_queue);
        int *max_val = sycl::malloc_host<int>(1, gpu_queue);
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

        sycl::free(min_val, gpu_queue);
        sycl::free(max_val, gpu_queue);
    }

    Segment(
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue,
        memory_manager &cpu_allocator,
        memory_manager &gpu_allocator,
        bool on_device,
        bool is_aggregate_result,
        uint64_t count = SEGMENT_SIZE
    )
        :
        nrows(count),
        gpu_queue(gpu_queue),
        cpu_queue(cpu_queue),
        on_device(on_device),
        is_aggregate_result(is_aggregate_result),
        is_materialized(true),
        dirty_cache(false)
    {
        if (count > SEGMENT_SIZE)
            throw std::bad_alloc();

        if (is_aggregate_result)
        {
            data_host = reinterpret_cast<int *>(cpu_allocator.alloc<uint64_t>(count));
            data_device = reinterpret_cast<int *>(gpu_allocator.alloc<uint64_t>(count));
        }
        else
        {
            data_host = cpu_allocator.alloc<int>(count);
            data_device = gpu_allocator.alloc<int>(count);
        }

        min = 0;
        max = 0;
    }

    Segment(
        uint64_t *init_data,
        bool on_device,
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        uint64_t count = SEGMENT_SIZE
    )
        :
        nrows(count),
        gpu_queue(gpu_queue),
        cpu_queue(cpu_queue),
        on_device(on_device),
        is_aggregate_result(true),
        is_materialized(true),
        dirty_cache(true)
    {
        if (count > SEGMENT_SIZE)
            throw std::bad_alloc();

        if (on_device)
        {
            data_device = reinterpret_cast<int *>(init_data);
            data_host = reinterpret_cast<int *>(cpu_allocator.alloc<uint64_t>(count));
        }
        else
        {
            data_host = reinterpret_cast<int *>(init_data);
            data_device = reinterpret_cast<int *>(gpu_allocator.alloc<uint64_t>(count));
        }

        min = 0;
        max = 0;
    }

    Segment(
        int *init_data,
        bool on_device,
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        uint64_t count = SEGMENT_SIZE
    )
        :
        nrows(count),
        gpu_queue(gpu_queue),
        cpu_queue(cpu_queue),
        on_device(on_device),
        is_aggregate_result(false),
        is_materialized(true),
        dirty_cache(true)
    {
        if (count > SEGMENT_SIZE)
            throw std::bad_alloc();

        if (on_device)
        {
            data_device = init_data;
            data_host = cpu_allocator.alloc<int>(count);
        }
        else
        {
            data_host = init_data;
            data_device = gpu_allocator.alloc<int>(count);
        }

        min = 0;
        max = 0;
    }

    ~Segment()
    {
        if (!is_materialized)
        {
            if (data_host != nullptr)
                sycl::free(data_host, cpu_queue);
            if (data_device != nullptr)
                sycl::free(data_device, gpu_queue);
        }
    }

    FillKernel *fill_with_literal(int literal)
    {
        if (!is_materialized)
            throw std::runtime_error("Cannot fill non-materialized segment");
        if (is_aggregate_result)
            throw std::runtime_error("Cannot fill aggregate result segment with int literal");

        min = literal;
        max = literal;

        dirty_cache = true;

        return new FillKernel(
            on_device ? data_device : data_host,
            literal,
            nrows
        );
    }

    bool is_on_device() const { return on_device; }
    int get_min() const { return min; }
    int get_max() const { return max; }
    uint64_t get_nrows() const { return nrows; }

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

    const int *get_data(bool device) const
    {
        if (device && !on_device)
            throw std::runtime_error("Segment data requested on device but it is on host");
        if (is_aggregate_result)
            throw std::runtime_error("Segment data requested but it is an aggregate result");
        return device ? data_device : data_host;
    }

    int *get_data(bool device)
    {
        dirty_cache = true;
        return const_cast<int *>(static_cast<const Segment &>(*this).get_data(device));
    }

    const uint64_t *get_aggregate_data(bool device) const
    {
        if (device && !on_device)
            throw std::runtime_error("Segment data requested on device but it is on host");
        if (!is_aggregate_result)
            throw std::runtime_error("Segment aggregate data requested but it is not an aggregate result");
        return reinterpret_cast<uint64_t *>(device ? data_device : data_host);
    }

    uint64_t *get_aggregate_data(bool device)
    {
        dirty_cache = true;
        return const_cast<uint64_t *>(static_cast<const Segment &>(*this).get_aggregate_data(device));
    }

    const int &operator[](uint64_t index) const
    {
        if (index >= nrows)
            throw std::out_of_range("Segment index out of range");
        if (is_aggregate_result)
            throw std::runtime_error("wrong operator[]");

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

    uint64_t get_data_size(bool gpu_only = false) const
    {
        if (gpu_only && !on_device)
            return 0;
        return nrows * (is_aggregate_result ? sizeof(uint64_t) : sizeof(int));
    }

    sycl::event move_to_device()
    {
        if (on_device)
            return sycl::event();

        if (data_device == nullptr)
            data_device = sycl::malloc_device<int>(nrows, gpu_queue);

        on_device = true;
        return gpu_queue.memcpy(data_device, data_host, nrows * sizeof(int));
    }

    sycl::event copy_on_host()
    {
        sycl::event e;

        if (on_device && dirty_cache)
        {
            if (is_aggregate_result)
                e = gpu_queue.memcpy(
                    reinterpret_cast<uint64_t *>(data_host),
                    reinterpret_cast<uint64_t *>(data_device),
                    nrows * sizeof(uint64_t)
                );
            else
                e = gpu_queue.memcpy(data_host, data_device, nrows * sizeof(int));

            dirty_cache = false;
        }
        else
            e = sycl::event();

        return e;
    }

    KernelBundle search_operator(
        const ExprType &expr,
        std::string parent_op,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        bool *gpu_flags,
        bool *cpu_flags) const
    {
        memory_manager &allocator = on_device ? gpu_allocator : cpu_allocator;
        int *data = on_device ? data_device : data_host;
        bool *flags = on_device ? gpu_flags : cpu_flags;

        bool *local_flags = allocator.alloc<bool>(nrows);

        KernelBundle operations;

        if (expr.operands[1].literal.rangeSet.size() == 1) // range
        {
            int lower = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                upper = std::stoi(expr.operands[1].literal.rangeSet[0][2]);

            operations.add_kernel(
                KernelData(
                    KernelType::SelectionKernelLiteral,
                    selection_def(
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
                    selection_def(
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
                    selection_def(
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
                    selection_def(
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
        bool *gpu_flags,
        bool *cpu_flags) const
    {
        return selection_def(
            on_device ? gpu_flags : cpu_flags,
            on_device ? data_device : data_host,
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
        bool *gpu_flags,
        bool *cpu_flags) const
    {
        return selection_def(
            on_device ? gpu_flags : cpu_flags,
            on_device ? data_device : data_host,
            op,
            other_segment.get_data(on_device),
            parent_op,
            nrows
        );
    }

    PerformOperationKernelColumns *perform_operator(
        const Segment &first_operand,
        const Segment &second_operand,
        const bool *flags,
        const std::string &op)
    {
        min = std::min(first_operand.min, second_operand.min);
        max = std::max(first_operand.max, second_operand.max);

        dirty_cache = true;

        return new PerformOperationKernelColumns(
            on_device ? data_device : data_host,
            first_operand.get_data(on_device),
            second_operand.get_data(on_device),
            flags,
            op,
            nrows
        );
    }

    PerformOperationKernelLiteralSecond *perform_operator(
        const Segment &first_operand,
        int second_operand,
        const bool *flags,
        const std::string &op)
    {
        min = first_operand.min;
        max = first_operand.max;

        dirty_cache = true;

        return new PerformOperationKernelLiteralSecond(
            on_device ? data_device : data_host,
            first_operand.get_data(on_device),
            second_operand,
            flags,
            op,
            nrows
        );
    }

    PerformOperationKernelLiteralFirst *perform_operator(
        int first_operand,
        const Segment &second_operand,
        const bool *flags,
        const std::string &op)
    {
        min = second_operand.min;
        max = second_operand.max;

        dirty_cache = true;

        return new PerformOperationKernelLiteralFirst(
            on_device ? data_device : data_host,
            first_operand,
            second_operand.get_data(on_device),
            flags,
            op,
            nrows
        );
    }

    AggregateOperationKernel *aggregate_operator(
        const bool *flags,
        uint64_t *agg_res) const
    {
        return new AggregateOperationKernel(
            on_device ? data_device : data_host,
            flags,
            nrows,
            agg_res
        );
    }

    BuildKeysHTKernel *build_keys_hash_table(
        bool *ht,
        const bool *flags,
        int ht_len,
        int ht_min_value
    ) const
    {
        return new BuildKeysHTKernel(
            ht,
            on_device ? data_device : data_host,
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
        const bool *build_ht) const
    {
        return new FilterJoinKernel(
            on_device ? data_device : data_host,
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
        const Segment &value_segment) const
    {
        return new BuildKeyValsHTKernel(
            ht,
            on_device ? data_device : data_host,
            value_segment.on_device ? value_segment.data_device : value_segment.data_host,
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
        int ht_max_value
    ) const
    {
        return new FullJoinKernel(
            on_device ? data_device : data_host,
            result_segment.on_device ? result_segment.data_device : result_segment.data_host,
            probe_flags,
            ht,
            ht_min_value,
            ht_max_value,
            nrows
        );
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

    Column(const int *init_data, uint64_t nrows, sycl::queue &gpu_queue, sycl::queue &cpu_queue)
        : is_aggregate_result(false)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0 ? 1 : 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(init_data + i * SEGMENT_SIZE, gpu_queue, cpu_queue);

        if (remainder > 0)
            segments.emplace_back(init_data + full_segments * SEGMENT_SIZE, gpu_queue, cpu_queue, remainder);
    }

    Column(
        uint64_t nrows,
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        bool on_device,
        bool is_aggregate_result)
        : is_aggregate_result(is_aggregate_result)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0 ? 1 : 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(gpu_queue, cpu_queue, cpu_allocator, gpu_allocator, on_device, is_aggregate_result);

        if (remainder > 0)
            segments.emplace_back(gpu_queue, cpu_queue, cpu_allocator, gpu_allocator, on_device, is_aggregate_result, remainder);
    }

    Column(
        uint64_t *init_data,
        bool on_device,
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        uint64_t nrows)
        : is_aggregate_result(true)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0 ? 1 : 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(
                init_data + i * SEGMENT_SIZE,
                on_device,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator
            );

        if (remainder > 0)
            segments.emplace_back(
                init_data + full_segments * SEGMENT_SIZE,
                on_device,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator,
                remainder
            );
    }

    Column(
        int *init_data,
        bool on_device,
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        uint64_t nrows)
        : is_aggregate_result(false)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0 ? 1 : 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(
                init_data + i * SEGMENT_SIZE,
                on_device,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator
            );

        if (remainder > 0)
            segments.emplace_back(
                init_data + full_segments * SEGMENT_SIZE,
                on_device,
                gpu_queue,
                cpu_queue,
                gpu_allocator,
                cpu_allocator,
                remainder
            );
    }

    const std::vector<Segment> &get_segments() const { return segments; }
    std::vector<Segment> &get_segments() { return segments; }
    bool get_is_aggregate_result() const { return is_aggregate_result; }

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
            throw std::runtime_error("wrong operator[]");
        uint64_t segment_index = index / SEGMENT_SIZE;
        uint64_t offset = index % SEGMENT_SIZE;
        return segments[segment_index][offset];
    }

    const uint64_t &get_aggregate_value(uint64_t index) const
    {
        if (!is_aggregate_result)
            throw std::runtime_error("wrong get_aggregate_value");
        uint64_t segment_index = index / SEGMENT_SIZE;
        uint64_t offset = index % SEGMENT_SIZE;
        return segments[segment_index].get_aggregate_value(offset);
    }

    std::vector<KernelBundle> fill_with_literal(int literal)
    {
        std::vector<KernelBundle> operations;
        operations.reserve(segments.size());

        for (auto &seg : segments)
        {
            KernelBundle bundle;

            bundle.add_kernel(
                KernelData(
                    KernelType::FillKernel,
                    seg.fill_with_literal(literal)
                )
            );
            operations.push_back(std::move(bundle));
        }

        return operations;
    }

    void move_all_to_device()
    {
        for (auto &seg : segments)
            seg.move_to_device();
    }

    uint64_t get_data_size(bool gpu_only = false) const
    {
        uint64_t total_size = 0;
        for (const auto &seg : segments)
            total_size += seg.get_data_size(gpu_only);
        return total_size;
    }

    std::tuple<bool *, int, int, std::vector<KernelBundle>> build_keys_hash_table(bool *flags, memory_manager &gpu_allocator, memory_manager &cpu_allocator) const
    {
        std::vector<KernelBundle> ops;
        ops.reserve(segments.size());

        auto min_max = get_min_max();
        int min_value = min_max.first;
        int max_value = min_max.second;

        int ht_len = max_value - min_value + 1;

        bool *ht = gpu_allocator.alloc<bool>(ht_len);

        for (int i = 0; i < segments.size(); i++)
        {
            KernelBundle bundle;
            bundle.add_kernel(
                KernelData(
                    KernelType::BuildKeysHTKernel,
                    segments[i].build_keys_hash_table(
                        ht,
                        flags + i * SEGMENT_SIZE,
                        ht_len,
                        min_value
                    )
                )
            );
            ops.push_back(bundle);
        }

        return { ht, min_value, max_value, ops };
    }

    std::vector<KernelBundle> semi_join(
        bool *probe_flags,
        int build_min_value,
        int build_max_value,
        const bool *build_ht)
    {
        std::vector<KernelBundle> ops;
        ops.reserve(segments.size());

        for (int i = 0; i < segments.size(); i++)
        {
            KernelBundle bundle;
            bundle.add_kernel(
                KernelData(
                    KernelType::FilterJoinKernel,
                    segments[i].semi_join_operator(
                        probe_flags + i * SEGMENT_SIZE,
                        build_min_value,
                        build_max_value,
                        build_ht
                    )
                )
            );
            ops.push_back(bundle);
        }

        return ops;
    }

    std::tuple<int *, int, int, std::vector<KernelBundle>> build_key_vals_hash_table(
        const Column *vals_column,
        bool *flags,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator) const
    {
        std::vector<KernelBundle> ops;
        ops.reserve(segments.size());

        auto min_max = get_min_max();
        int min_value = min_max.first;
        int max_value = min_max.second;

        int ht_len = max_value - min_value + 1;

        int *ht = gpu_allocator.alloc<int>(ht_len * 2);

        for (int i = 0; i < segments.size(); i++)
        {
            KernelBundle bundle;
            bundle.add_kernel(
                KernelData(
                    KernelType::BuildKeyValsHTKernel,
                    segments[i].build_key_vals_hash_ht(
                        ht,
                        flags + i * SEGMENT_SIZE,
                        ht_len,
                        min_value,
                        vals_column->segments[i]
                    )
                )
            );
            ops.push_back(bundle);
        }

        return { ht, min_value, max_value, ops };
    }

    std::pair<std::vector<KernelBundle>, Column> full_join_operation(
        bool *probe_flags,
        int build_min_value,
        int build_max_value,
        int group_by_column_min,
        int group_by_column_max,
        const int *build_ht,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        sycl::queue &gpu_queue,
        sycl::queue &cpu_queue) const
    {
        std::vector<KernelBundle> ops;
        uint64_t num_rows = (segments.size() - 1) * SEGMENT_SIZE + segments.back().get_nrows();
        Column new_column(
            num_rows,
            gpu_queue,
            cpu_queue,
            gpu_allocator,
            cpu_allocator,
            true,
            false
        );

        ops.reserve(segments.size());

        for (int i = 0; i < segments.size(); i++)
        {
            const Segment &seg = segments[i];
            Segment &new_seg = new_column.segments[i];
            KernelBundle bundle;

            bundle.add_kernel(
                KernelData(
                    KernelType::FullJoinKernel,
                    seg.full_join_operator(
                        new_seg,
                        probe_flags + i * SEGMENT_SIZE,
                        build_ht,
                        build_min_value,
                        build_max_value
                    )
                )
            );
            ops.push_back(bundle);

            new_seg.set_min(group_by_column_min);
            new_seg.set_max(group_by_column_max);
        }

        return { ops, new_column };
    }
};


class Table
{
private:
    std::string table_name;
    std::vector<Column> columns;
    uint64_t nrows;
public:
    Table(const std::string table_name, sycl::queue &gpu_queue, sycl::queue &cpu_queue)
        : table_name(table_name)
    {
        int col_number = table_column_numbers[table_name], *content;
        columns.reserve(col_number);

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
            else
            {
                colData.seekg(0, std::ios::beg);
                colData.read((char *)content, num_entries * sizeof(int));
                columns.emplace_back(content, num_entries, gpu_queue, cpu_queue);
            }

            colData.close();
        }
        sycl::free(content, cpu_queue);
    }

    uint64_t get_nrows() const { return nrows; }
    const std::vector<Column> &get_columns() const { return columns; }
    const std::string &get_name() const { return table_name; }

    uint64_t get_data_size(bool gpu_only = false) const
    {
        uint64_t total_size = 0;
        for (const auto &col : columns)
            total_size += col.get_data_size(gpu_only);
        return total_size;
    }

    void move_all_to_device()
    {
        for (auto &col : columns)
            col.move_all_to_device();
    }
};