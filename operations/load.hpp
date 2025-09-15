#pragma once

#include <set>
#include <fstream>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"

#define DATA_DIR "/tmp/data/s20_columnar/"

TableData<int> loadTable(std::string table_name, int col_number, const std::set<int> &columns, sycl::queue &queue)
{
    TableData<int> res;

    res.col_number = col_number;
    res.columns_size = columns.size();
    res.table_name = table_name;

    res.columns = sycl::malloc_shared<ColumnData<int>>(res.columns_size, queue);

    int i = 0;
    for (auto &col_idx : columns)
    {
        res.column_indices[col_idx] = i; // map the column index to the actual position

        auto table_name = res.table_name;
        std::transform(table_name.begin(), table_name.end(), table_name.begin(), ::toupper);
        std::string col_name = table_name + std::to_string(col_idx);
        std::string filename = DATA_DIR + col_name;
        std::cout << "Loading column: " << filename << std::endl;

        std::ifstream colData(filename.c_str(), std::ios::in | std::ios::binary);

        colData.seekg(0, std::ios::end);
        std::streampos fileSize = colData.tellg();
        int num_entries = static_cast<int>(fileSize / sizeof(int));

        colData.seekg(0, std::ios::beg);
        int *h_col = sycl::malloc_host<int>(num_entries, queue);
        colData.read((char *)h_col, num_entries * sizeof(int));

        res.columns[i].content = sycl::malloc_shared<int>(num_entries, queue);
        queue.memcpy(res.columns[i].content, h_col, num_entries * sizeof(int)).wait();
        queue.wait();

        res.col_len = num_entries;
        res.columns[i].has_ownership = true;
        res.columns[i].is_aggregate_result = false;
        res.columns[i].min_value = *std::min_element(h_col, h_col + res.col_len);
        res.columns[i].max_value = *std::max_element(h_col, h_col + res.col_len);
        sycl::free(h_col, queue);

        i++;
    }

    std::cout << "Loaded table: " << res.table_name << " with " << res.col_len << " rows and " << res.col_number << " columns (" << res.columns_size << " in memory)" << std::endl;

    bool *flags = sycl::malloc_host<bool>(res.col_len, queue);
    std::fill_n(flags, res.col_len, true);
    res.flags = sycl::malloc_shared<bool>(res.col_len, queue);
    queue.prefetch(res.flags, res.col_len * sizeof(bool));
    queue.memcpy(res.flags, flags, res.col_len * sizeof(bool)).wait();
    sycl::free(flags, queue);
    return res;
}