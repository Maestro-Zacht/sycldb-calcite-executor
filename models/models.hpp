#pragma once

#include <sycl/sycl.hpp>

#include <fstream>

#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"
#include "../kernels/selection.hpp"
#include "../kernels/types.hpp"
#include "../operations/load.hpp"

#define SEGMENT_SIZE (((uint64_t)1) << 20)


class Segment
{
private:
    int *data_host, *data_device;
    int min, max;
    uint64_t nrows;
    sycl::queue gpu_queue, cpu_queue;
    bool on_device, is_materialized;
public:
    Segment(const int *init_data, sycl::queue &gpu_queue, sycl::queue &cpu_queue, uint64_t count = SEGMENT_SIZE)
        :
        data_device(nullptr),
        nrows(count),
        gpu_queue(gpu_queue),
        cpu_queue(cpu_queue),
        on_device(false),
        is_materialized(false)
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
        uint64_t count = SEGMENT_SIZE
    )
        :
        nrows(count),
        gpu_queue(gpu_queue),
        cpu_queue(cpu_queue),
        on_device(on_device),
        is_materialized(true)
    {
        if (count > SEGMENT_SIZE)
            throw std::bad_alloc();

        if (on_device)
        {
            data_host = nullptr;
            data_device = gpu_allocator.alloc<int>(count);
        }
        else
        {
            data_device = nullptr;
            data_host = cpu_allocator.alloc<int>(count);
        }

        min = INT_MAX;
        max = INT_MIN;
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

    sycl::event fill_with_literal(int literal)
    {
        if (!is_materialized)
            throw std::runtime_error("Cannot fill non-materialized segment");

        min = literal;
        max = literal;

        if (on_device)
            return gpu_queue.fill<int>(data_device, literal, nrows);
        else
            return cpu_queue.fill<int>(data_host, literal, nrows);
    }

    bool is_on_device() const { return on_device; }

    int *get_data(bool device) const
    {
        if (device && !on_device)
            throw std::runtime_error("Segment data requested on device but it is on host");
        return device ? data_device : data_host;
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

    sycl::event search_operator(
        const ExprType &expr,
        std::string parent_op,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        bool *gpu_flags,
        bool *cpu_flags,
        const std::vector<sycl::event> &dependencies) const
    {
        memory_manager &allocator = on_device ? gpu_allocator : cpu_allocator;
        sycl::queue &queue = const_cast<sycl::queue &>(on_device ? gpu_queue : cpu_queue);
        int *data = on_device ? data_device : data_host;
        bool *flags = on_device ? gpu_flags : cpu_flags;

        bool *local_flags = allocator.alloc<bool>(nrows);

        sycl::event last_event;

        if (expr.operands[1].literal.rangeSet.size() == 1) // range
        {
            int lower = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                upper = std::stoi(expr.operands[1].literal.rangeSet[0][2]);

            last_event = selection(local_flags, data, ">=", lower, "NONE", nrows, queue, dependencies);
            last_event = selection(local_flags, data, "<=", upper, "AND", nrows, queue, { last_event });

            // TODO: min and max here
            // table_data.columns[col_index].min_value = lower;
            // table_data.columns[col_index].max_value = upper;
        }
        else // or between two values
        {
            int first = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                second = std::stoi(expr.operands[1].literal.rangeSet[1][1]);

            last_event = selection(local_flags, data, "==", first, "NONE", nrows, queue, dependencies);
            last_event = selection(local_flags, data, "==", second, "OR", nrows, queue, { last_event });
        }

        logical_op logic = get_logical_op(parent_op);

        return queue.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(last_event);
                cgh.parallel_for(
                    nrows,
                    [=](sycl::id<1> idx)
                    {
                        flags[idx[0]] = logical(logic, flags[idx[0]], local_flags[idx[0]]);
                    }
                );
            }
        );
    }

    sycl::event filter_operator(
        std::string op,
        std::string parent_op,
        int literal_value,
        bool *gpu_flags,
        bool *cpu_flags,
        const std::vector<sycl::event> &dependencies) const
    {
        return selection(
            on_device ? gpu_flags : cpu_flags,
            on_device ? data_device : data_host,
            op,
            literal_value,
            parent_op,
            nrows,
            const_cast<sycl::queue &>(on_device ? gpu_queue : cpu_queue),
            dependencies
        );
    }

    sycl::event filter_operator(
        std::string op,
        std::string parent_op,
        const Segment &other_segment,
        bool *gpu_flags,
        bool *cpu_flags,
        const std::vector<sycl::event> &dependencies) const
    {
        return selection(
            on_device ? gpu_flags : cpu_flags,
            on_device ? data_device : data_host,
            op,
            other_segment.get_data(on_device),
            parent_op,
            nrows,
            const_cast<sycl::queue &>(on_device ? gpu_queue : cpu_queue),
            dependencies
        );
    }

    sycl::event perform_operator(
        const Segment &first_operand,
        const Segment &second_operand,
        const bool *flags,
        const std::string &op,
        const std::vector<sycl::event> &dependencies)
    {
        sycl::event e = perform_operation(
            on_device ? data_device : data_host,
            first_operand.get_data(on_device),
            second_operand.get_data(on_device),
            flags,
            nrows,
            op,
            const_cast<sycl::queue &>(on_device ? gpu_queue : cpu_queue),
            dependencies
        );

        min = std::min(first_operand.min, second_operand.min);
        max = std::max(first_operand.max, second_operand.max);

        return e;
    }

    sycl::event perform_operator(
        const Segment &first_operand,
        int second_operand,
        const bool *flags,
        const std::string &op,
        const std::vector<sycl::event> &dependencies)
    {
        sycl::event e = perform_operation(
            on_device ? data_device : data_host,
            first_operand.get_data(on_device),
            second_operand,
            flags,
            nrows,
            op,
            const_cast<sycl::queue &>(on_device ? gpu_queue : cpu_queue),
            dependencies
        );

        min = first_operand.min;
        max = first_operand.max;

        return e;
    }

    sycl::event perform_operator(
        int first_operand,
        const Segment &second_operand,
        const bool *flags,
        const std::string &op,
        const std::vector<sycl::event> &dependencies)
    {
        sycl::event e = perform_operation(
            on_device ? data_device : data_host,
            first_operand,
            second_operand.get_data(on_device),
            flags,
            nrows,
            op,
            const_cast<sycl::queue &>(on_device ? gpu_queue : cpu_queue),
            dependencies
        );

        min = second_operand.min;
        max = second_operand.max;

        return e;
    }
};


class Column
{
private:
    std::vector<Segment> segments;
public:
    Column(const int *init_data, uint64_t nrows, sycl::queue &gpu_queue, sycl::queue &cpu_queue)
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
        bool on_device)
    {
        uint64_t full_segments = nrows / SEGMENT_SIZE;
        uint64_t remainder = nrows % SEGMENT_SIZE;

        segments.reserve(full_segments + (remainder > 0 ? 1 : 0));

        for (uint64_t i = 0; i < full_segments; i++)
            segments.emplace_back(gpu_queue, cpu_queue, cpu_allocator, gpu_allocator, on_device);

        if (remainder > 0)
            segments.emplace_back(gpu_queue, cpu_queue, cpu_allocator, gpu_allocator, on_device, remainder);
    }

    const std::vector<Segment> &get_segments() const { return segments; }
    std::vector<Segment> &get_segments() { return segments; }

    std::vector<sycl::event> fill_with_literal(int literal)
    {
        std::vector<sycl::event> events;
        events.reserve(segments.size());

        for (auto &seg : segments)
            events.push_back(seg.fill_with_literal(literal));

        return events;
    }

    void move_all_to_device()
    {
        for (auto &seg : segments)
            seg.move_to_device();
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
        int col_number = table_column_numbers[table_name];
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

            colData.seekg(0, std::ios::beg);

            int *content = new int[num_entries];
            colData.read((char *)content, num_entries * sizeof(int));
            colData.close();

            columns.emplace_back(content, num_entries, gpu_queue, cpu_queue);

            delete[] content;

            if (i == 0)
                nrows = num_entries;
            else if (num_entries != nrows)
            {
                // throw std::runtime_error("Column length mismatch in " + filename + ": expected " + std::to_string(nrows) + ", got " + std::to_string(num_entries));
                std::cerr << "Warning: Column length mismatch in " << filename << ": expected " << nrows << ", got " << num_entries << std::endl;
            }
        }
    }

    uint64_t get_nrows() const { return nrows; }
    const std::vector<Column> &get_columns() const { return columns; }
    const std::string &get_name() const { return table_name; }

    void move_all_to_device()
    {
        for (auto &col : columns)
            col.move_all_to_device();
    }
};