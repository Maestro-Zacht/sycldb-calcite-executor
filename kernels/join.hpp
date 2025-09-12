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
    queue.parallel_for(
             sycl::range<1>{(unsigned long)col_len},
             [=](sycl::id<1> idx)
             {
                 auto i = idx[0];
                 if (flags[i])
                 {
                     int hash = HASH(col[i], ht_len, ht_min_value);
                     ht[hash << 1] = 1;
                     ht[(hash << 1) + 1] = agg_col[i];
                 }
                 else
                     ht[HASH(col[i], ht_len, ht_min_value) << 1] = 0;
             })
        .wait();
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
    bool *ht = sycl::malloc_shared<bool>(ht_len, queue);
    queue.fill(ht, false, ht_len).wait();

    build_keys_ht(build_col, build_flags, build_col_len, ht, ht_len, build_min_value, queue);

    queue.parallel_for(
             sycl::range<1>{(unsigned long)probe_col_len},
             [=](sycl::id<1> idx)
             {
                 auto i = idx[0];
                 if (probe_col_flags[i] &&
                     probe_col[i] >= build_min_value &&
                     probe_col[i] <= build_max_value)
                     probe_col_flags[i] = ht[HASH(probe_col[i], ht_len, build_min_value)];
             })
        .wait();

    sycl::free(ht, queue);
}

void full_join(TableData<int> &probe_table,
               TableData<int> &build_table,
               int probe_col_index,
               int build_col_index, sycl::queue &queue)
{
    int build_column = build_table.column_indices.at(build_col_index),
        probe_column = probe_table.column_indices.at(probe_col_index),
        group_by_column = build_table.column_indices.at(build_table.group_by_column),
        ht_len = build_table.columns[build_column].max_value -
                 build_table.columns[build_column].min_value + 1,
        *ht = sycl::malloc_shared<int>(ht_len * 2, queue);

    queue.fill(ht, 0, ht_len * 2).wait();

    build_key_vals_ht(
        build_table.columns[build_column].content,
        build_table.columns[group_by_column].content,
        build_table.flags, build_table.col_len, ht, ht_len,
        build_table.columns[build_column].min_value, queue);

    bool *probe_flags = probe_table.flags;
    int *probe_content = probe_table.columns[probe_column].content,
        build_col_min = build_table.columns[build_column].min_value,
        build_col_max = build_table.columns[build_column].max_value;

    queue.parallel_for(
             sycl::range<1>{(unsigned long)probe_table.col_len},
             [=](sycl::id<1> idx)
             {
                 auto i = idx[0];
                 if (probe_flags[i])
                 {
                     int hash = HASH(probe_content[i], ht_len,
                                     build_col_min);
                     if (probe_content[i] >= build_col_min &&
                         probe_content[i] <= build_col_max &&
                         ht[hash << 1] == 1)
                         probe_content[i] = ht[(hash << 1) + 1]; // replace the probe column value with the value to group by on
                     else
                         probe_flags[i] = false; // mark as not selected
                 }
             })
        .wait();

    // the group by column index must refer to the old foreign key
    probe_table.column_indices.erase(probe_col_index);
    probe_table.column_indices[probe_table.col_number + build_table.group_by_column] = probe_column;

    // update min and max values of the probe column
    probe_table.columns[probe_column].min_value = build_table.columns[group_by_column].min_value;
    probe_table.columns[probe_column].max_value = build_table.columns[group_by_column].max_value;

    sycl::free(ht, queue);
}