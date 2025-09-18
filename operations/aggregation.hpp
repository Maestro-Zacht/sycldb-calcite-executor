#pragma once

#include <vector>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"
#include "../kernels/aggregation.hpp"

#include "../gen-cpp/calciteserver_types.h"

std::vector<sycl::event> parse_aggregate(
    TableData<int> &table_data,
    const AggType &agg,
    const std::vector<long> &group,
    std::vector<void *> &resources,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    std::vector<sycl::event> events;
    events.reserve(group.size() + 2);

    if (group.size() == 0)
    {
        uint64_t *result = sycl::malloc_device<uint64_t>(1, queue);
        events.push_back(aggregate_operation(result,
            table_data.columns[table_data.column_indices.at(agg.operands[0])].content,
            table_data.flags, table_data.col_len, agg.agg, queue, dependencies));

        // Free old columns and replace with the result column
        for (int i = 0; i < table_data.columns_size; i++)
            if (table_data.columns[i].has_ownership)
                resources.push_back(table_data.columns[i].content);
        resources.push_back(table_data.columns);
        resources.push_back(table_data.flags);
        table_data.column_indices.clear();

        table_data.columns = sycl::malloc_shared<ColumnData<int>>(1, queue);
        table_data.columns[0].content = (int *)result;
        table_data.columns[0].has_ownership = true;
        table_data.columns[0].is_aggregate_result = true;
        table_data.columns[0].min_value = 0; // TODO: set real min value
        table_data.columns[0].max_value = 0; // TODO: set real max value
        table_data.col_number = 1;
        table_data.columns_size = 1;
        table_data.col_len = 1;
        table_data.column_indices[0] = 0;

        table_data.flags = sycl::malloc_device<bool>(1, queue);
        events.push_back(queue.fill(table_data.flags, true, 1));
    }
    else
    {
        ColumnData<int> *group_columns = sycl::malloc_shared<ColumnData<int>>(group.size(), queue);
        for (int i = 0; i < group.size(); i++)
            group_columns[i] = table_data.columns[table_data.column_indices.at(group[i])];

        auto agg_res = group_by_aggregate(
            group_columns,
            table_data.columns[table_data.column_indices.at(agg.operands[0])].content,
            table_data.flags, group.size(), table_data.col_len, agg.agg,
            resources, queue, dependencies);

        resources.push_back(group_columns);

        // Free old columns and replace with the result columns
        for (int i = 0; i < table_data.columns_size; i++)
            if (table_data.columns[i].has_ownership)
                resources.push_back(table_data.columns[i].content);
        resources.push_back(table_data.columns);
        resources.push_back(table_data.flags);
        table_data.column_indices.clear();

        sycl::event agg_event = std::get<4>(agg_res);

        table_data.columns = sycl::malloc_shared<ColumnData<int>>(group.size() + 1, queue);
        for (int i = 0; i < group.size(); i++)
        {
            table_data.columns[i].content = sycl::malloc_device<int>(std::get<1>(agg_res), queue);
            events.push_back(queue.memcpy(table_data.columns[i].content,
                std::get<0>(agg_res) + i * std::get<1>(agg_res),
                std::get<1>(agg_res) * sizeof(int), agg_event));
            table_data.columns[i].has_ownership = true;
            table_data.columns[i].is_aggregate_result = false;
            table_data.columns[i].min_value = 0; // TODO: set real min value
            table_data.columns[i].max_value = 0; //  TODO: set real max value
            table_data.column_indices[i] = i;
        }

        table_data.columns[group.size()].content = (int *)std::get<3>(agg_res);
        table_data.columns[group.size()].has_ownership = true;
        table_data.columns[group.size()].is_aggregate_result = true;
        table_data.columns[group.size()].min_value = 0; // TODO: set real min value
        table_data.columns[group.size()].max_value = 0; // TODO: set real max value
        table_data.column_indices[group.size()] = group.size();

        table_data.col_number = group.size() + 1;
        table_data.columns_size = group.size() + 1;
        table_data.col_len = std::get<1>(agg_res);
        table_data.flags = std::get<2>(agg_res);

        resources.push_back(std::get<0>(agg_res));
    }

    return events;
}
