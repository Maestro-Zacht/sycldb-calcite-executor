#pragma once

#include <iostream>
#include <map>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"
#include "../kernels/join.hpp"

#include "../gen-cpp/calciteserver_types.h"

void parse_join(const RelNode &rel, TableData<int> &left_table, TableData<int> &right_table, const std::map<std::string, int> &table_last_used, sycl::queue &queue)
{
    int left_column = rel.condition.operands[0].input,
        right_column = rel.condition.operands[1].input - left_table.col_number;

    if (left_column < 0 ||
        left_column >= left_table.col_number ||
        right_column < 0 ||
        right_column >= right_table.col_number)
    {
        std::cout << "Join operation: Invalid column indices in join condition." << std::endl;
        return;
    }

    // filter joins if the right table is last accessed at this operation
    if (right_table.table_name != "" && table_last_used.at(right_table.table_name) == rel.id)
    {
        filter_join(right_table.columns[right_table.column_indices.at(right_column)].content,
                    right_table.flags, right_table.col_len,
                    right_table.columns[right_table.column_indices.at(right_column)].max_value,
                    right_table.columns[right_table.column_indices.at(right_column)].min_value,
                    left_table.columns[left_table.column_indices.at(left_column)].content,
                    left_table.flags, left_table.col_len, queue);
    }
    else if (left_table.table_name == "lineorder")
    {
        full_join(left_table, right_table, left_column, right_column, queue);
    }
    else
    {
        std::cout << "Join operation Unsupported" << std::endl;
    }
    left_table.col_number += right_table.col_number;
}
