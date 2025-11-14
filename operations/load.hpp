#pragma once

#include <set>
#include <fstream>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"
#include "memory_manager.hpp"

#define DATA_DIR "/tmp/data/s20_columnar/"

TableData<int> loadTable(
    std::string table_name,
    int col_number,
    const std::set<int> &columns,
    sycl::queue &queue,
    memory_manager &gpu_allocator,
    bool load_on_device = true)
{
    TableData<int> res;

    res.col_number = col_number;
    res.columns_size = columns.size();
    res.table_name = table_name;
    res.ht = nullptr;

    res.columns = sycl::malloc_shared<ColumnData<int>>(res.columns_size, queue);

    int i = 0;
    for (auto &col_idx : columns)
    {
        res.column_indices[col_idx] = i; // map the column index to the actual position

        auto table_name = res.table_name;
        std::transform(table_name.begin(), table_name.end(), table_name.begin(), ::toupper);
        std::string col_name = table_name + std::to_string(col_idx);
        std::string filename = DATA_DIR + col_name;
        // std::cout << "Loading column: " << filename << std::endl;

        std::ifstream colData(filename.c_str(), std::ios::in | std::ios::binary);

        colData.seekg(0, std::ios::end);
        std::streampos fileSize = colData.tellg();
        int num_entries = static_cast<int>(fileSize / sizeof(int));

        colData.seekg(0, std::ios::beg);

        int *content = sycl::malloc_host<int>(num_entries, queue);
        colData.read((char *)content, num_entries * sizeof(int));
        colData.close();

        if (load_on_device)
        {
            res.columns[i].content = gpu_allocator.alloc<int>(num_entries);
            queue.memcpy(content, res.columns[i].content, num_entries * sizeof(int)).wait();
            sycl::free(content, queue);
        }
        else
        {
            res.columns[i].content = content;
        }



        res.col_len = num_entries;
        res.columns[i].has_ownership = true;
        res.columns[i].is_aggregate_result = false;

        int *min_val = sycl::malloc_shared<int>(1, queue);
        int *max_val = sycl::malloc_shared<int>(1, queue);
        content = res.columns[i].content;

        auto e2 = queue.copy(content, min_val, 1);
        auto e3 = queue.copy(content, max_val, 1);

        queue.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(e2);
                cgh.depends_on(e3);

                cgh.parallel_for(
                    sycl::range<1>(res.col_len - 1),
                    sycl::reduction(max_val, sycl::maximum<int>()),
                    sycl::reduction(min_val, sycl::minimum<int>()),
                    [=](sycl::id<1> idx, auto &maxr, auto &minr)
                    {
                        auto j = idx[0] + 1;
                        int val = content[j];
                        maxr.combine(val);
                        minr.combine(val);
                    }
                );
            }
        ).wait();

        res.columns[i].min_value = *min_val;
        res.columns[i].max_value = *max_val;
        sycl::free(min_val, queue);
        sycl::free(max_val, queue);

        i++;
    }

    std::cout << "Loaded table: " << res.table_name
        << " with " << res.col_len << " rows and "
        << res.col_number << " columns ("
        << res.columns_size << " in memory)" << std::endl;

    res.flags = gpu_allocator.alloc<bool>(res.col_len);
    queue.fill(res.flags, true, res.col_len).wait();
    return res;
}

std::map<std::string, TableData<int>> preload_all_tables(sycl::queue &queue, memory_manager &gpu_allocator)
{
    std::map<std::string, TableData<int>> tables;

    for (const auto &table_entry : table_column_indices)
    {
        const std::string &table_name = table_entry.first;
        const std::set<int> &columns = table_entry.second;

        tables[table_name] = loadTable(table_name, table_column_numbers[table_name], columns, queue, gpu_allocator, false);
    }

    return tables;
}

TableData<int> copy_table(
    const TableData<int> &table_data,
    const std::set<int> &columns,
    memory_manager &gpu_allocator,
    sycl::queue &queue)
{
    TableData<int> res;

    res.col_number = table_data.col_number;
    res.columns_size = columns.size();
    res.table_name = table_data.table_name;
    res.ht = nullptr;
    res.col_len = table_data.col_len;

    res.columns = sycl::malloc_shared<ColumnData<int>>(res.columns_size, queue);

    int i = 0;
    for (auto &col_idx : columns)
    {
        res.column_indices[col_idx] = i; // map the column index to the actual position
        int orig_col_idx = table_data.column_indices.at(col_idx);

        int *content = gpu_allocator.alloc<int>(table_data.col_len);
        queue.copy(table_data.columns[orig_col_idx].content, content, table_data.col_len);
        res.columns[i].content = content;
        res.columns[i].has_ownership = true;
        res.columns[i].is_aggregate_result = false;

        res.columns[i].min_value = table_data.columns[orig_col_idx].min_value;
        res.columns[i].max_value = table_data.columns[orig_col_idx].max_value;

        i++;
    }

    res.flags = gpu_allocator.alloc<bool>(res.col_len);
    queue.copy(table_data.flags, res.flags, res.col_len);
    return res;
}