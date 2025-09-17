#pragma once

#include <set>
#include <fstream>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"

#define DATA_DIR "/tmp/data/s20_columnar/"

TableData<int> loadTable(std::string table_name, int col_number, const std::set<int> &columns, std::vector<void *> &resources, sycl::queue &queue)
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
        // std::cout << "Loading column: " << filename << std::endl;

        std::ifstream colData(filename.c_str(), std::ios::in | std::ios::binary);

        colData.seekg(0, std::ios::end);
        std::streampos fileSize = colData.tellg();
        int num_entries = static_cast<int>(fileSize / sizeof(int));

        colData.seekg(0, std::ios::beg);

        int *content = sycl::malloc_host<int>(num_entries, queue);
        colData.read((char *)content, num_entries * sizeof(int));
        colData.close();

        res.columns[i].content = sycl::malloc_device<int>(num_entries, queue);
        auto e1 = queue.copy(content, res.columns[i].content, num_entries);
        resources.push_back(content);

        res.col_len = num_entries;
        res.columns[i].has_ownership = true;
        res.columns[i].is_aggregate_result = false;

        int *min_val = sycl::malloc_shared<int>(1, queue);
        int *max_val = sycl::malloc_shared<int>(1, queue);
        content = res.columns[i].content;

        auto e2 = queue.copy(content, min_val, 1, e1);
        auto e3 = queue.copy(content, max_val, 1, e1);

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
        );

        res.columns[i].min_value = *min_val;
        res.columns[i].max_value = *max_val;
        resources.push_back(min_val);
        resources.push_back(max_val);

        i++;
    }

    std::cout << "Loaded table: " << res.table_name
        << " with " << res.col_len << " rows and "
        << res.col_number << " columns ("
        << res.columns_size << " in memory)" << std::endl;

    res.flags = sycl::malloc_device<bool>(res.col_len, queue);
    queue.fill(res.flags, true, res.col_len);
    return res;
}