#pragma once

#include <iostream>
#include <map>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"
#include "../kernels/join.hpp"

#include "../gen-cpp/calciteserver_types.h"

std::vector<sycl::event> parse_join(
    const RelNode &rel,
    TableData<int> &left_table,
    TableData<int> &right_table,
    const std::map<std::string, int> &table_last_used,
    std::vector<void *> &resources,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    int left_column = rel.condition.operands[0].input,
        right_column = rel.condition.operands[1].input - left_table.col_number;

    sycl::event event;

    if (left_column < 0 ||
        left_column >= left_table.col_number ||
        right_column < 0 ||
        right_column >= right_table.col_number)
    {
        std::cerr << "Join operation: Invalid column indices in join condition." << std::endl;
        return {};
    }

    // filter joins if the right table is last accessed at this operation
    if (right_table.table_name != "" && table_last_used.at(right_table.table_name) == rel.id)
    {
        int max_value, min_value;

        if (right_table.ht != nullptr)
        {
            max_value = right_table.ht_max;
            min_value = right_table.ht_min;
        }
        else
        {
            max_value = right_table.columns[right_table.column_indices.at(right_column)].max_value;
            min_value = right_table.columns[right_table.column_indices.at(right_column)].min_value;
        }

        event = filter_join(
            right_table.columns[right_table.column_indices.at(right_column)].content,
            right_table.flags, right_table.col_len,
            max_value, min_value,
            left_table.columns[left_table.column_indices.at(left_column)].content,
            left_table.flags, left_table.col_len, (bool *)right_table.ht,
            resources, queue, dependencies);
    }
    else if (left_table.table_name == "lineorder")
    {
        event = full_join(left_table, right_table, left_column, right_column, resources, queue, dependencies);
    }
    else
    {
        std::cerr << "Join operation Unsupported" << std::endl;
    }
    left_table.col_number += right_table.col_number;

    return { event };
}
