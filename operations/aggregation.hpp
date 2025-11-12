#pragma once

#include <vector>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"
#include "../kernels/aggregation.hpp"

#include "memory_manager.hpp"

#include "../gen-cpp/calciteserver_types.h"

std::vector<sycl::event> parse_aggregate(
    TableData<int> &table_data,
    const AggType &agg,
    const std::vector<long> &group,
    std::vector<void *> &resources,
    memory_manager &gpu_allocator,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    #if PRINT_AGGREGATE_DEBUG_INFO
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<sycl::event> events;
    events.reserve(group.size() + 2);

    if (group.size() == 0)
    {
        uint64_t *result = gpu_allocator.alloc<uint64_t>(1);
        auto agg_op = sycl::reduction(result, sycl::plus<>());
        events.push_back(aggregate_operation(
            table_data.columns[table_data.column_indices.at(agg.operands[0])].content,
            table_data.flags, table_data.col_len, agg_op, queue, dependencies));

        #if PRINT_AGGREGATE_DEBUG_INFO
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_time = end - start;
        std::cout << "Total aggregate time (no group by): " << total_time.count() << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        #endif

        // Free old columns and replace with the result column
        resources.push_back(table_data.columns);
        table_data.column_indices.clear();

        #if PRINT_AGGREGATE_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> free_time = end - start;
        std::cout << "Free old columns time: " << free_time.count() << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        #endif

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

        #if PRINT_AGGREGATE_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> alloc_time = end - start;
        std::cout << "Allocate new column time: " << alloc_time.count() << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        #endif

        table_data.flags = gpu_allocator.alloc<bool>(1);
        events.push_back(queue.fill(table_data.flags, true, 1));

        #if PRINT_AGGREGATE_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> flag_time = end - start;
        std::cout << "Allocate and set flags time: " << flag_time.count() << " ms" << std::endl;
        #endif
    }
    else
    {
        ColumnData<int> *group_columns = sycl::malloc_shared<ColumnData<int>>(group.size(), queue);
        for (int i = 0; i < group.size(); i++)
            group_columns[i] = table_data.columns[table_data.column_indices.at(group[i])];

        #if PRINT_AGGREGATE_DEBUG_INFO
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> copy_time = end - start;
        std::cout << "Prepare group columns time: " << copy_time.count() << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        #endif

        auto agg_res = group_by_aggregate(
            group_columns,
            table_data.columns[table_data.column_indices.at(agg.operands[0])].content,
            table_data.flags, group.size(), table_data.col_len, agg.agg,
            gpu_allocator, queue, dependencies);

        resources.push_back(group_columns);

        #if PRINT_AGGREGATE_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> agg_time = end - start;
        std::cout << "Total aggregate time (with group by): " << agg_time.count() << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        #endif

        // Free old columns and replace with the result columns
        resources.push_back(table_data.columns);
        table_data.column_indices.clear();

        #if PRINT_AGGREGATE_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> free_time = end - start;
        std::cout << "Free old columns time: " << free_time.count() << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        #endif

        sycl::event agg_event = std::get<4>(agg_res);

        int **results = std::get<0>(agg_res);

        table_data.columns = sycl::malloc_shared<ColumnData<int>>(group.size() + 1, queue);
        for (int i = 0; i < group.size(); i++)
        {
            table_data.columns[i].content = results[i];
            table_data.columns[i].has_ownership = true;
            table_data.columns[i].is_aggregate_result = false;
            table_data.columns[i].min_value = 0; // TODO: set real min value
            table_data.columns[i].max_value = 0; //  TODO: set real max value
            table_data.column_indices[i] = i;
        }

        resources.push_back(results);

        #if PRINT_AGGREGATE_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> alloc_time = end - start;
        std::cout << "Allocate and copy new columns time: " << alloc_time.count() << " ms" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        #endif

        table_data.columns[group.size()].content = (int *)std::get<3>(agg_res);
        table_data.columns[group.size()].has_ownership = true;
        table_data.columns[group.size()].is_aggregate_result = true;
        table_data.columns[group.size()].min_value = 0; // TODO: set real min value
        table_data.columns[group.size()].max_value = 0; // TODO: set real max value
        table_data.column_indices[group.size()] = group.size();

        table_data.col_number = group.size() + 1;
        table_data.columns_size = group.size() + 1;
        table_data.col_len = std::get<1>(agg_res);
        table_data.flags = std::get<2>(agg_res);;

        #if PRINT_AGGREGATE_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> flag_time = end - start;
        std::cout << "Allocate and set new flags time: " << flag_time.count() << " ms" << std::endl;
        #endif
    }

    return events;
}
