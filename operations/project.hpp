#pragma once

#include <vector>

#include <sycl/sycl.hpp>

#include "../kernels/types.hpp"
#include "../kernels/aggregation.hpp"

#include "memory_manager.hpp"

#include "../gen-cpp/calciteserver_types.h"

std::vector<sycl::event> parse_project(
    const std::vector<ExprType> &exprs,
    TableData<int> &table_data,
    std::vector<void *> &resources,
    memory_manager &gpu_allocator,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    ColumnData<int> *new_columns = sycl::malloc_shared<ColumnData<int>>(exprs.size(), queue);

    std::vector<sycl::event> events(dependencies);

    for (size_t i = 0; i < exprs.size(); i++)
    {
        switch (exprs[i].exprType)
        {
        case ExprOption::COLUMN:
            // Copy the column data from the original table and pass ownership
            new_columns[i] = table_data.columns[table_data.column_indices.at(exprs[i].input)];

            table_data.columns[table_data.column_indices.at(exprs[i].input)].has_ownership = false;
            if (exprs[i].input == table_data.group_by_column)
                table_data.group_by_column = i; // update group by column index
            break;
        case ExprOption::LITERAL:
            // create a new column with the literal value
            new_columns[i].content = gpu_allocator.alloc<int>(table_data.col_len, true);
            events = {
                queue.fill(new_columns[i].content, (int)exprs[i].literal.value, table_data.col_len, std::move(events))
            };
            new_columns[i].min_value = exprs[i].literal.value;
            new_columns[i].max_value = exprs[i].literal.value;
            new_columns[i].has_ownership = true;
            new_columns[i].is_aggregate_result = false;
            break;
        case ExprOption::EXPR:
            // Assumed only 2 operands which are either COLUMN or LITERAL
            if (exprs[i].operands.size() != 2)
            {
                std::cerr << "Project operation: Unsupported number of operands for EXPR" << std::endl;
                return {};
            }
            new_columns[i].content = gpu_allocator.alloc<int>(table_data.col_len, true);
            new_columns[i].has_ownership = true;
            new_columns[i].is_aggregate_result = false;

            // call the perform_operation overloaded function based on the operands types
            // and set min and max values of the column
            if (exprs[i].operands[0].exprType == ExprOption::COLUMN &&
                exprs[i].operands[1].exprType == ExprOption::COLUMN)
            {
                events = { perform_operation(new_columns[i].content,
                    table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].content,
                    table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].content,
                    table_data.flags, table_data.col_len, exprs[i].op, queue, std::move(events)
                ) };
                new_columns[i].min_value =
                    std::min(table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].min_value,
                        table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].min_value);
                new_columns[i].max_value =
                    std::max(table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].max_value,
                        table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].max_value);
            }
            else if (exprs[i].operands[0].exprType == ExprOption::LITERAL &&
                exprs[i].operands[1].exprType == ExprOption::COLUMN)
            {
                events = { perform_operation(new_columns[i].content,
                    (int)exprs[i].operands[0].literal.value,
                    table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].content,
                    table_data.flags, table_data.col_len, exprs[i].op, queue, std::move(events)) };
                new_columns[i].min_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].min_value;
                new_columns[i].max_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[1].input)].max_value;
            }
            else if (exprs[i].operands[0].exprType == ExprOption::COLUMN &&
                exprs[i].operands[1].exprType == ExprOption::LITERAL)
            {
                events = { perform_operation(new_columns[i].content,
                    table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].content,
                    (int)exprs[i].operands[1].literal.value,
                    table_data.flags, table_data.col_len, exprs[i].op, queue, std::move(events)) };
                new_columns[i].min_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].min_value;
                new_columns[i].max_value = table_data.columns[table_data.column_indices.at(exprs[i].operands[0].input)].max_value;
            }
            else
            {
                std::cerr << "Project operation: Unsupported parsing ExprType "
                    << exprs[i].operands[0].exprType << " and "
                    << exprs[i].operands[1].exprType
                    << " for EXPR" << std::endl;
                return {};
            }
            break;
        }
    }

    // Free old columns and replace with new ones
    resources.push_back(table_data.columns);

    table_data.columns = new_columns;
    table_data.col_number = exprs.size();
    table_data.columns_size = exprs.size();

    // update column indices (they are now just themselves)
    table_data.column_indices.clear();
    for (int i = 0; i < table_data.col_number; i++)
        table_data.column_indices[i] = i;

    return events;
}