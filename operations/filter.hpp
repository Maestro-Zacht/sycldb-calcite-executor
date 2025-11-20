#pragma once

#include <iostream>

#include <sycl/sycl.hpp>

#include "../gen-cpp/CalciteServer.h"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/types.hpp"
#include "../kernels/selection.hpp"
#include "memory_manager.hpp"

std::vector<sycl::event> parse_filter(
    const ExprType &expr,
    const TableData<int> table_data,
    std::string parent_op,
    memory_manager &gpu_allocator,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    // Recursive parsing of EXPR types. LITERAL and COLUMN are handled in parent EXPR type.
    if (expr.exprType != ExprOption::EXPR)
    {
        std::cerr << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
        return {};
    }

    std::vector<sycl::event> events;

    if (expr.op == "SEARCH")
    {
        int col_index = table_data.column_indices.at(expr.operands[0].input);
        bool *local_flags = gpu_allocator.alloc<bool>(table_data.col_len, true);
        sycl::event last_event;

        if (expr.operands[1].literal.rangeSet.size() == 1) // range
        {
            int lower = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                upper = std::stoi(expr.operands[1].literal.rangeSet[0][2]);

            auto e1 = selection(local_flags,
                table_data.columns[col_index].content,
                ">=", lower, "NONE", table_data.col_len, queue, dependencies);
            last_event = selection(local_flags,
                table_data.columns[col_index].content,
                "<=", upper, "AND", table_data.col_len, queue, { e1 });

            table_data.columns[col_index].min_value = lower;
            table_data.columns[col_index].max_value = upper;
        }
        else // or between two values
        {
            int first = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                second = std::stoi(expr.operands[1].literal.rangeSet[1][1]);

            auto e1 = selection(local_flags,
                table_data.columns[col_index].content,
                "==", first, "NONE", table_data.col_len, queue, dependencies);
            last_event = selection(local_flags,
                table_data.columns[col_index].content,
                "==", second, "OR", table_data.col_len, queue, { e1 });
        }
        bool *flags = table_data.flags;
        logical_op logic = get_logical_op(parent_op);
        events.push_back(
            queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(last_event);
                    cgh.parallel_for(
                        table_data.col_len,
                        [=](sycl::id<1> idx)
                        {
                            flags[idx[0]] = logical(logic, flags[idx[0]], local_flags[idx[0]]);
                        }
                    );
                }
            )
        );
    }
    else if (is_filter_logical(expr.op))
    {
        // Logical operation between other expressions. Pass parent op to the first then use the current op.
        // TODO: check if passing parent logic is correct in general
        bool parent_op_used = false;
        std::vector<sycl::event> child_deps(dependencies);
        for (const ExprType &operand : expr.operands)
        {
            child_deps = parse_filter(operand, table_data, parent_op_used ? expr.op : parent_op, gpu_allocator, queue, std::move(child_deps));
            parent_op_used = true;
        }
        events.insert(events.end(), child_deps.begin(), child_deps.end());
    }
    else
    {
        // Comparison between two operands
        int **cols = new int *[2];
        bool literal = false;
        if (expr.operands.size() != 2)
        {
            std::cerr << "Filter condition: Unsupported number of operands for EXPR" << std::endl;
            return {};
        }

        // Get the pointer to the two columns or make a new column with the literal value as first cell
        for (int i = 0; i < 2; i++)
        {
            switch (expr.operands[i].exprType)
            {
            case ExprOption::COLUMN:
                cols[i] = table_data.columns[table_data.column_indices.at(expr.operands[i].input)].content;
                break;
            case ExprOption::LITERAL:
                cols[i] = new int[1];
                literal = true;
                cols[i][0] = expr.operands[i].literal.value;
                break;
            default:
                std::cerr << "Filter condition: Unsupported parsing ExprType "
                    << expr.operands[i].exprType
                    << " for comparison operand"
                    << std::endl;
                return {};
            }
        }

        // Assumed literal is always the second operand.
        if (literal)
        {
            events.push_back(
                selection(table_data.flags, cols[0], expr.op, cols[1][0], parent_op, table_data.col_len, queue, dependencies)
            );
            delete[] cols[1];
        }
        else
            events.push_back(
                selection(table_data.flags, cols[0], expr.op, cols[1], parent_op, table_data.col_len, queue, dependencies)
            );

        delete[] cols;
    }
    return events;
}