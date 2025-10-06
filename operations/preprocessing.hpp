#pragma once

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <deque>
#include <string>

#include "../gen-cpp/CalciteServer.h"
#include "../gen-cpp/calciteserver_types.h"

#include "../kernels/types.hpp"


struct ExecutionInfo
{
    std::map<std::string, std::set<int>> loaded_columns;
    std::map<std::string, int> table_last_used, group_by_columns;
    std::map<std::string, std::tuple<int, int>> prepare_join;
    std::vector<int> dag_order;
};

void parse_expression_columns(const ExprType &expr, std::set<int> &columns)
{
    if (expr.exprType == ExprOption::COLUMN)
        columns.insert(expr.input);
    else if (expr.exprType == ExprOption::EXPR)
        for (const ExprType &operand : expr.operands)
            parse_expression_columns(operand, columns);
}

std::vector<int> dag_topological_sort(const PlanResult &result)
{
    int **dag = new int *[result.rels.size()];
    std::deque<int> S;
    for (int i = 0; i < result.rels.size(); i++)
    {
        dag[i] = new int[result.rels.size()];
        std::fill_n(dag[i], result.rels.size(), 0);
    }

    for (const RelNode &rel : result.rels)
    {
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
            S.push_front(rel.id); // push the table scan operation to the list of nodes without inbound edges
            break;
        case RelNodeType::FILTER:
        case RelNodeType::PROJECT:
        case RelNodeType::AGGREGATE:
        case RelNodeType::SORT:
            dag[rel.id - 1][rel.id] = 1; // connect to the previous operation
            break;
        case RelNodeType::JOIN:
            if (rel.inputs.size() != 2)
            {
                std::cerr << "Join operation: Invalid number of inputs." << std::endl;
                return std::vector<int>();
            }
            dag[rel.inputs[0]][rel.id] = 1; // connect left input to the join
            dag[rel.inputs[1]][rel.id] = 1; // connect right input to the join
            break;
        default:
            std::cerr << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            return std::vector<int>();
        }
    }

    // Topological sort of the DAG
    std::vector<int> sorted;
    sorted.reserve(result.rels.size());

    while (!S.empty())
    {
        int node = S.back();
        S.pop_back();

        if (S.size() != 0 &&
            result.rels[node].relOp == RelNodeType::TABLE_SCAN &&
            result.rels[node].tables[1] == "lineorder")
        {
            S.push_front(node);
            continue;
        }
        sorted.push_back(node);

        for (int i = 0; i < result.rels.size(); i++)
        {
            if (dag[node][i] == 1) // if there is an edge from node to i
            {
                dag[node][i] = 0; // remove the edge

                // check if there are any inbound edges to i
                bool has_inbound_edges = false;

                for (int j = 0; j < result.rels.size(); j++)
                {
                    if (dag[j][i] == 1)
                    {
                        has_inbound_edges = true;
                        break;
                    }
                }

                if (!has_inbound_edges)
                    S.push_front(i); // push to the list of nodes without inbound edges
            }
        }
    }

    for (int i = 0; i < result.rels.size(); i++)
        delete[] dag[i];
    delete[] dag;
    if (sorted.size() != result.rels.size())
    {
        std::cerr << "DAG topological sort failed: not all nodes were sorted." << std::endl;
        return std::vector<int>();
    }
    return sorted;
}

// initially parse the data structure in order to find all columns used and the last time each table was used
ExecutionInfo parse_execution_info(const PlanResult &result)
{
    ExecutionInfo info;

    // ops_info stores information about every operation.
    // index is the operation id, content is a vector of tuples
    // where the first element is the table name and the second is the column number.
    std::vector<std::vector<std::tuple<std::string, int>>> ops_info;
    ops_info.reserve(result.rels.size());

    for (const RelNode &rel : result.rels)
    {
        switch (rel.relOp)
        {
        case RelNodeType::TABLE_SCAN:
        {
            int num_columns = table_column_numbers[rel.tables[1]];
            std::vector<std::tuple<std::string, int>> table;
            table.reserve(num_columns);

            // all the columns of the table
            for (int i = 0; i < num_columns; i++)
                table.push_back(std::make_tuple(rel.tables[1], i));

            ops_info.push_back(table);

            // init info for the table in the result
            info.loaded_columns[rel.tables[1]] = std::set<int>();
            info.table_last_used[rel.tables[1]] = rel.id;
            break;
        }
        case RelNodeType::FILTER:
        {
            // a filter list of columns is the same as the previous operation
            std::vector<std::tuple<std::string, int>> op_info(ops_info.back());
            std::set<int> columns;

            parse_expression_columns(rel.condition, columns);

            // mark columns in the filter as used
            // mark the table as last used at current id
            for (int col : columns)
            {
                std::string table_name = std::get<0>(op_info[col]);
                info.loaded_columns[table_name].insert(std::get<1>(op_info[col]));
                info.table_last_used[table_name] = rel.id;
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::PROJECT:
        {
            std::vector<std::tuple<std::string, int>> op_info, last_op_info = ops_info.back();

            for (const ExprType &expr : rel.exprs)
            {
                std::set<int> columns;
                parse_expression_columns(expr, columns);

                // mark columns in the project as used (if any)
                // mark the table as last used at current id
                for (int col : columns)
                {
                    std::string table_name = std::get<0>(last_op_info[col]);
                    info.loaded_columns[table_name].insert(std::get<1>(last_op_info[col]));
                    info.table_last_used[table_name] = rel.id;
                }

                // if the expression contains at least one column, use the first one as a reference
                // TODO: improve this by considering all columns
                if (!columns.empty())
                    op_info.push_back(std::make_tuple(
                        std::get<0>(last_op_info[*columns.begin()]),
                        std::get<1>(last_op_info[*columns.begin()])));
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::AGGREGATE:
        {
            std::vector<std::tuple<std::string, int>> op_info, last_op_info = ops_info.back();

            // mark all columns in the group by as used (if any)
            // mark the table as last used at current id
            // save the info about the columns since they form the new table
            for (int agg_col : rel.group)
            {
                std::string table_name = std::get<0>(last_op_info[agg_col]);
                int col_index = std::get<1>(last_op_info[agg_col]);
                info.loaded_columns[table_name].insert(col_index);
                info.table_last_used[table_name] = rel.id;
                op_info.push_back(std::make_tuple(table_name, col_index));
                info.group_by_columns[table_name] = col_index;
            }

            for (const AggType &agg : rel.aggs)
            {
                // save columns and table for every aggregate operation
                for (int agg_col : agg.operands)
                {
                    std::string table_name = std::get<0>(last_op_info[agg_col]);
                    info.loaded_columns[table_name].insert(std::get<1>(last_op_info[agg_col]));
                    info.table_last_used[table_name] = rel.id;
                }

                // use the first column of the aggregate as a reference
                // TODO: improve this by considering all columns
                int agg_col = agg.operands[0];
                op_info.push_back(std::make_tuple(
                    std::get<0>(last_op_info[agg_col]),
                    std::get<1>(last_op_info[agg_col])));
            }
            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::JOIN:
        {
            int left_id = rel.inputs[0], right_id = rel.inputs[1];
            std::vector<std::tuple<std::string, int>> left_info = ops_info[left_id],
                right_info = ops_info[right_id],
                op_info;
            std::set<int> columns;

            parse_expression_columns(rel.condition, columns);

            op_info.reserve(left_info.size() + right_info.size());

            // mark columns in the join condition as used
            // mark the tables as last used at current id
            for (int col : columns)
            {
                std::string table_name = std::get<0>(
                    (col < left_info.size())
                    ? left_info[col]
                    : right_info[col - left_info.size()]);

                int col_index = std::get<1>(
                    (col < left_info.size())
                    ? left_info[col]
                    : right_info[col - left_info.size()]
                );

                if (table_name != "lineorder")
                {
                    int previous_op_id = info.table_last_used[table_name];
                    if (previous_op_id != rel.id)
                    {
                        int right_column = rel.condition.operands[1].input - left_info.size();
                        info.prepare_join[table_name] = std::make_tuple(rel.id, right_column);
                    }
                }

                info.loaded_columns[table_name].insert(col_index);
                info.table_last_used[table_name] = rel.id;
            }

            // insert left and right info into the operation info
            op_info.insert(op_info.begin(), left_info.begin(), left_info.end());
            op_info.insert(op_info.end(), right_info.begin(), right_info.end());

            ops_info.push_back(op_info);
            break;
        }
        case RelNodeType::SORT:
        {
            std::vector<std::tuple<std::string, int>> op_info(ops_info.back());
            std::set<int64_t> columns;

            for (const CollationType &col : rel.collation)
                columns.insert(col.field);

            for (int64_t col : columns)
            {
                if (col < op_info.size()) // some projects will add literals columns that are not in any original table
                {
                    std::string table_name = std::get<0>(op_info[col]);
                    int col_index = std::get<1>(op_info[col]);
                    info.loaded_columns[table_name].insert(col_index);
                    info.table_last_used[table_name] = rel.id;
                }
            }

            ops_info.push_back(op_info);
            break;
        }
        default:
            std::cout << "Unsupported RelNodeType: " << rel.relOp << std::endl;
            break;
        }
    }

    info.dag_order = dag_topological_sort(result);

    return info;
}