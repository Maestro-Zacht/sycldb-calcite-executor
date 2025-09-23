#pragma once

#include <map>

#define MAX_NTABLES 5

std::map<std::string, std::set<int>> table_column_indices(
    {
        {"lineorder", {2, 3, 4, 5, 8, 9, 11, 12, 13}},
        {"part", {0, 2, 3, 4}},
        {"supplier", {0, 3, 4, 5}},
        {"customer", {0, 3, 4, 5}},
        {"ddate", {0, 4, 5}}
    }
);

std::map<std::string, int> table_column_numbers(
    {
        {"lineorder", 17},
        {"part", 9},
        {"supplier", 7},
        {"customer", 8},
        {"ddate", 17}
    }
);

template <typename T>
struct ColumnData
{
    T *content;
    bool has_ownership;       // Indicates if this column owns the memory for its content
    bool is_aggregate_result; // Indicates if this column is the result of an aggregate operation
    T min_value;              // Minimum value in the column
    T max_value;              // Maximum value in the column
};

template <typename T>
struct TableData
{
    ColumnData<T> *columns;
    int columns_size;    // length of the columns array (number of columns loaded)
    int col_len;         // number of rows
    int col_number;      // total number of columns in the table
    bool *flags;         // selection flags
    int group_by_column; // column number used for grouping, -1 if not used
    void *ht;               // hash table for joins
    std::string table_name;
    std::map<int, int> column_indices; // Maps column numbers from calcite to its index in the columns array
};