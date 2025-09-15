#pragma once

#include <sycl/sycl.hpp>

#include "types.hpp"

void sort_table(TableData<int> &table_data, const int *sort_columns, const bool *ascending, int num_sort_columns, sycl::queue &queue)
{
    // Create an array of indices to represent the original row order
    int *indices = sycl::malloc_host<int>(table_data.col_len, queue);
    for (int i = 0; i < table_data.col_len; i++)
        indices[i] = i;

    auto compare = [&](int a, int b)
    {
        for (size_t i = 0; i < num_sort_columns; i++)
        {
            bool asc = ascending[i];
            ColumnData<int> col = table_data.columns[table_data.column_indices.at(sort_columns[i])];

            if (col.is_aggregate_result)
            {
                uint64_t *content = (uint64_t *)col.content;

                if (content[a] != content[b])
                {
                    return asc != (content[a] > content[b]);
                }
            }
            else
            {
                int *content = col.content;

                if (content[a] != content[b])
                {
                    return asc != (content[a] > content[b]);
                }
            }
        }
        return false;
    };

    std::sort(indices, indices + table_data.col_len, compare);

    // Copy sorted data into new columns
    ColumnData<int> *sorted_columns = sycl::malloc_shared<ColumnData<int>>(table_data.col_number, queue);
    bool *sorted_flags = sycl::malloc_shared<bool>(table_data.col_len, queue);
    for (int i = 0; i < table_data.col_number; i++)
    {
        sorted_columns[i].is_aggregate_result = table_data.columns[i].is_aggregate_result;
        if (table_data.columns[i].is_aggregate_result)
            sorted_columns[i].content = sycl::malloc_shared<int>(sizeof(uint64_t) / sizeof(int) * table_data.col_len, queue);
        else
            sorted_columns[i].content = sycl::malloc_shared<int>(table_data.col_len, queue);

        sorted_columns[i].has_ownership = true;
        sorted_columns[i].min_value = table_data.columns[i].min_value;
        sorted_columns[i].max_value = table_data.columns[i].max_value;

        for (int j = 0; j < table_data.col_len; j++)
        {
            if (sorted_columns[i].is_aggregate_result)
                ((uint64_t *)sorted_columns[i].content)[j] = ((uint64_t *)table_data.columns[i].content)[indices[j]];
            else
                sorted_columns[i].content[j] = table_data.columns[i].content[indices[j]];
        }

        if (table_data.columns[i].has_ownership)
            sycl::free(table_data.columns[i].content, queue);
    }

    for (int i = 0; i < table_data.col_len; i++)
        sorted_flags[i] = table_data.flags[indices[i]];

    sycl::free(indices, queue);
    sycl::free(table_data.flags, queue);
    sycl::free(table_data.columns, queue);

    table_data.columns = sorted_columns;
    table_data.flags = sorted_flags;
}