#pragma once

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"
#include "../kernels/sort.hpp"

#include "../gen-cpp/calciteserver_types.h"

void parse_sort(const RelNode &rel, TableData<int> &table_data, sycl::queue &queue)
{
    if (rel.collation.size() == 0)
        return;

    int *sort_columns = sycl::malloc_host<int>(rel.collation.size(), queue);
    bool *sort_orders = sycl::malloc_host<bool>(rel.collation.size(), queue);

    for (int i = 0; i < rel.collation.size(); i++)
    {
        sort_columns[i] = rel.collation[i].field;
        sort_orders[i] = rel.collation[i].direction == DirectionOption::ASCENDING;
    }

    sort_table(table_data, sort_columns, sort_orders, rel.collation.size(), queue);

    sycl::free(sort_columns, queue);
    sycl::free(sort_orders, queue);
}