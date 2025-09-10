#pragma once

template <typename T>
inline T HASH(T X, T Y, T Z)
{
    return ((X - Z) % Y);
}

template <typename T>
void build_keys_ht(T col[], bool flags[], int col_len, bool ht[], T ht_len, T ht_min_value, sycl::queue &queue)
{
    queue.parallel_for(
             col_len,
             [=](sycl::id<1> i)
             {
                 ht[HASH(col[i], ht_len, ht_min_value)] = flags[i];
             })
        .wait();
}

void build_key_vals_ht(int col[], int agg_col[], bool flags[], int col_len, int ht[], int ht_len, int ht_min_value, sycl::queue &queue)
{
    for (int i = 0; i < col_len; i++)
    {
        if (flags[i])
        {
            int hash = HASH(col[i], ht_len, ht_min_value);
            ht[hash << 1] = 1;
            ht[(hash << 1) + 1] = agg_col[i];
        }
        else
            ht[HASH(col[i], ht_len, ht_min_value) << 1] = 0;
    }
    // queue.parallel_for(
    //          col_len,
    //          [=](sycl::id<1> i)
    //          {
    //              if (flags[i])
    //              {
    //                  int hash = HASH(col[i], ht_len, ht_min_value);
    //                  ht[hash << 1] = 1;
    //                  ht[(hash << 1) + 1] = agg_col[i];
    //              }
    //              else
    //                  ht[HASH(col[i], ht_len, ht_min_value) << 1] = 0;
    //          })
    //     .wait();
}

template <typename T>
void filter_join(T build_col[],
                 bool build_flags[],
                 int build_col_len,
                 T build_max_value,
                 T build_min_value,
                 T probe_col[],
                 bool probe_col_flags[],
                 int probe_col_len, sycl::queue &queue)
{
    int ht_len = build_max_value - build_min_value + 1;
    // bool *ht = new bool[ht_len];
    // std::fill_n(ht, ht_len, false);
    bool *ht = sycl::malloc_shared<bool>(ht_len, queue);
    queue.fill(ht, false, ht_len).wait();

    build_keys_ht(build_col, build_flags, build_col_len, ht, ht_len, build_min_value, queue);

    for (int i = 0; i < probe_col_len; i++)
        if (probe_col_flags[i])
            probe_col_flags[i] = ht[HASH(probe_col[i], ht_len, build_min_value)];

    // delete[] ht;
    sycl::free(ht, queue);
}

// TODO: move to GPU
void full_join(TableData<int> &probe_table,
               TableData<int> &build_table,
               int probe_col_index,
               int build_col_index, sycl::queue &queue)
{
    int ht_len = build_table.columns[build_table.column_indices.at(build_col_index)].max_value -
                 build_table.columns[build_table.column_indices.at(build_col_index)].min_value + 1;
    int *ht = new int[ht_len * 2],
        build_column = build_table.column_indices.at(build_col_index),
        probe_column = probe_table.column_indices.at(probe_col_index);

    std::fill_n(ht, ht_len * 2, 0); // maybe it's not necessary to initialize

    build_key_vals_ht(
        build_table.columns[build_column].content,
        build_table.columns[build_table.column_indices.at(build_table.group_by_column)].content,
        build_table.flags, build_table.col_len, ht, ht_len,
        build_table.columns[build_column].min_value, queue);

    for (int i = 0; i < probe_table.col_len; i++)
    {
        if (probe_table.flags[i])
        {
            int hash = HASH(probe_table.columns[probe_column].content[i], ht_len,
                            build_table.columns[build_column].min_value);
            if (ht[hash << 1] == 1)
                probe_table.columns[probe_column].content[i] = ht[(hash << 1) + 1]; // replace the probe column value with the value to group by on
            else
                probe_table.flags[i] = false; // mark as not selected
        }
    }
    // the group by column index must refer to the old foreign key
    probe_table.column_indices.erase(probe_col_index);
    probe_table.column_indices[probe_table.col_number + build_table.group_by_column] = probe_column;

    // update min and max values of the probe column
    probe_table.columns[probe_column].min_value = build_table.columns[build_table.group_by_column].min_value;
    probe_table.columns[probe_column].max_value = build_table.columns[build_table.group_by_column].max_value;

    delete[] ht;
}