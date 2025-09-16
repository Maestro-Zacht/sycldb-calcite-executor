#pragma once

#include <iostream>

#include <sycl/sycl.hpp>

#include "../gen-cpp/CalciteServer.h"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/types.hpp"
#include "../kernels/selection.hpp"

// logicals are AND, OR etc. while comparisons are ==, <= etc.
// So checking alpha characters is enough to determine if the operation is logical.
bool is_filter_logical(const std::string &op)
{
    for (int i = 0; i < op.length(); i++)
        if (!isalpha(op[i]))
            return false;
    return true;
}

void parse_filter(const ExprType &expr,
                  const TableData<int> table_data,
                  std::string parent_op, sycl::queue &queue)
{
    // Recursive parsing of EXRP types. LITERAL and COLUMN are handled in parent EXPR type.
    if (expr.exprType != ExprOption::EXPR)
    {
        std::cout << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
        return;
    }

    if (expr.op == "SEARCH")
    {
        int col_index = table_data.column_indices.at(expr.operands[0].input);
        bool *local_flags = sycl::malloc_device<bool>(table_data.col_len, queue);

        if (expr.operands[1].literal.rangeSet.size() == 1) // range
        {
            int lower = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                upper = std::stoi(expr.operands[1].literal.rangeSet[0][2]);

            selection(local_flags,
                      table_data.columns[col_index].content,
                      ">=", lower, "NONE", table_data.col_len, queue);
            selection(local_flags,
                      table_data.columns[col_index].content,
                      "<=", upper, "AND", table_data.col_len, queue);

            table_data.columns[col_index].min_value = lower;
            table_data.columns[col_index].max_value = upper;
        }
        else // or between two values
        {
            int first = std::stoi(expr.operands[1].literal.rangeSet[0][1]),
                second = std::stoi(expr.operands[1].literal.rangeSet[1][1]);

            selection(local_flags,
                      table_data.columns[col_index].content,
                      "==", first, "NONE", table_data.col_len, queue);
            selection(local_flags,
                      table_data.columns[col_index].content,
                      "==", second, "OR", table_data.col_len, queue);
        }
        bool *flags = table_data.flags;
        logical_op logic = get_logical_op(parent_op);
        queue.parallel_for(
                 table_data.col_len,
                 [=](sycl::id<1> idx)
                 {
                     flags[idx[0]] = logical(logic, flags[idx[0]], local_flags[idx[0]]);
                 })
            .wait();
        sycl::free(local_flags, queue);
    }
    else if (is_filter_logical(expr.op))
    {
        // Logical operation between other expressions. Pass parent op to the first then use the current op.
        // TODO: check if passing parent logic is correct in general
        bool parent_op_used = false;
        for (const ExprType &operand : expr.operands)
        {
            parse_filter(operand, table_data, parent_op_used ? expr.op : parent_op, queue);
            parent_op_used = true;
        }
    }
    else
    {
        // Comparison between two operands
        int **cols = new int *[2];
        bool literal = false;
        if (expr.operands.size() != 2)
        {
            std::cout << "Filter condition: Unsupported number of operands for EXPR" << std::endl;
            return;
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
                std::cout << "Filter condition: Unsupported parsing ExprType "
                          << expr.operands[i].exprType
                          << " for comparison operand"
                          << std::endl;
                break;
            }
        }

        // Assumed literal is always the second operand.
        if (literal)
        {
            selection(table_data.flags, cols[0], expr.op, cols[1][0], parent_op, table_data.col_len, queue);
            delete[] cols[1];
        }
        else
            selection(table_data.flags, cols[0], expr.op, cols[1], parent_op, table_data.col_len, queue);

        delete[] cols;
    }
}