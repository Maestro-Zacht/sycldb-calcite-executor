#pragma once

#include <sycl/sycl.hpp>

#include <fstream>

#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"
#include "../kernels/selection.hpp"
#include "../kernels/types.hpp"
#include "../operations/load.hpp"

#define SEGMENT_SIZE (((uint64_t)1) << 20)

class Table;

class Segment
{
private:
    int *data_host, *data_device;
    uint64_t nrows;
    sycl::queue gpu_queue, cpu_queue;
    bool on_device;
public:
    Segment(const int *init_data, uint64_t count, sycl::queue &gpu_queue, sycl::queue &cpu_queue)
        : data_device(nullptr), nrows(count), gpu_queue(gpu_queue), cpu_queue(cpu_queue), on_device(false)
    {
        if (count > SEGMENT_SIZE)
            throw std::bad_alloc();

        data_host = sycl::malloc_host<int>(count, gpu_queue);

        if (init_data != nullptr)
            gpu_queue.memcpy(data_host, init_data, count * sizeof(int)).wait();
        else
            std::cerr << "Warning: Segment not initialized" << std::endl;
    }

    Segment(const int *init_data, sycl::queue &gpu_queue, sycl::queue &cpu_queue)
        : Segment(init_data, SEGMENT_SIZE, gpu_queue, cpu_queue)
    {}

    ~Segment()
    {
        if (data_host != nullptr)
            sycl::free(data_host, gpu_queue);
        if (data_device != nullptr)
            sycl::free(data_device, gpu_queue);
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

            auto e1 = selection(local_flags, data, ">=", lower, "NONE", nrows, queue, dependencies);
            last_event = selection(local_flags, data, "<=", upper, "AND", nrows, queue, { e1 });

            // TODO: min and max here
            // table_data.columns[col_index].min_value = lower;
            // table_data.columns[col_index].max_value = upper;
        }
        else // or between two values
        {
            int first = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                second = std::stoi(expr.operands[1].literal.rangeSet[1][1]);

            auto e1 = selection(local_flags, data, "==", first, "NONE", nrows, queue, dependencies);
            last_event = selection(local_flags, data, "==", second, "OR", nrows, queue, { e1 });
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
            segments.emplace_back(init_data + full_segments * SEGMENT_SIZE, full_segments, gpu_queue, cpu_queue);
    }

    const std::vector<Segment> &get_segments() const { return segments; }

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

    void move_all_to_device()
    {
        for (auto &col : columns)
            col.move_all_to_device();
    }
};