#pragma once

#include <sycl/sycl.hpp>

#include "../operations/memory_manager.hpp"
#include "types.hpp"
#include "common.hpp"

#define PRINT_JOIN_DEBUG_INFO 0

template <typename T>
inline T HASH(T X, T Y, T Z)
{
    return ((X - Z) % Y);
}

class BuildKeysHTKernel : public KernelDefinition
{
private:
    bool *ht;
    const int *col;
    const bool *flags;
    int ht_len, ht_min_value;
public:
    BuildKeysHTKernel(bool *hash_table, const int *column, const bool *flags, int ht_length, int ht_min, int col_len)
        : KernelDefinition(col_len), ht(hash_table), col(column), flags(flags), ht_len(ht_length), ht_min_value(ht_min)
    {}

    void operator()(sycl::id<1> idx) const
    {
        ht[HASH(col[idx], ht_len, ht_min_value)] = flags[idx];
    }
};

sycl::event build_keys_ht(
    const int col[],
    const bool flags[],
    int col_len,
    bool ht[],
    int ht_len,
    int ht_min_value,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                col_len,
                [=](sycl::id<1> i)
                {
                    ht[HASH(col[i], ht_len, ht_min_value)] = flags[i];
                }
            );
        }
    );
}

class BuildKeyValsHTKernel : public KernelDefinition
{
private:
    int *ht;
    const int *col;
    const int *agg_col;
    const bool *flags;
    int ht_len, ht_min_value;
public:
    BuildKeyValsHTKernel(
        int *hash_table,
        const int *column,
        const int *agg_column,
        const bool *flgs,
        int ht_length,
        int ht_min,
        int col_len)
        : KernelDefinition(col_len), ht(hash_table), col(column), agg_col(agg_column), flags(flgs),
        ht_len(ht_length), ht_min_value(ht_min)
    {}

    void operator()(sycl::id<1> idx) const
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
    }
};

sycl::event build_key_vals_ht(
    int col[],
    int agg_col[],
    const bool flags[],
    int col_len,
    int ht[],
    int ht_len,
    int ht_min_value,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                col_len,
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
                }
            );
        }
    );
}

class FilterJoinKernel : public KernelDefinition
{
private:
    const int *probe_col;
    bool *probe_col_flags;
    const bool *build_ht;
    int build_min_value, build_max_value, ht_len;
public:
    FilterJoinKernel(
        const int *probe_column,
        bool *probe_column_flags,
        const bool *build_hash_table,
        int build_min,
        int build_max,
        int col_len)
        : KernelDefinition(col_len), probe_col(probe_column), probe_col_flags(probe_column_flags), build_ht(build_hash_table),
        build_min_value(build_min), build_max_value(build_max)
    {
        ht_len = build_max_value - build_min_value + 1;
    }

    void operator()(sycl::id<1> idx) const
    {
        auto i = idx[0];
        if (
            probe_col_flags[i] &&
            probe_col[i] >= build_min_value &&
            probe_col[i] <= build_max_value
            )
            probe_col_flags[i] = build_ht[HASH(probe_col[i], ht_len, build_min_value)];
        else
            probe_col_flags[i] = false;
    }
};

sycl::event filter_join(
    const int *probe_col,
    bool *probe_col_flags,
    int probe_col_len,
    const bool *build_ht,
    int build_min_value,
    int build_max_value,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    int ht_len = build_max_value - build_min_value + 1;

    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                probe_col_len,
                [=](sycl::id<1> idx)
                {
                    auto i = idx[0];
                    if (
                        probe_col_flags[i] &&
                        probe_col[i] >= build_min_value &&
                        probe_col[i] <= build_max_value
                        )
                        probe_col_flags[i] = build_ht[HASH(probe_col[i], ht_len, build_min_value)];
                    else
                        probe_col_flags[i] = false;
                }
            );
        }
    );
}

sycl::event filter_join(
    int build_col[],
    bool build_flags[],
    int build_col_len,
    int build_max_value,
    int build_min_value,
    int probe_col[],
    bool probe_col_flags[],
    int probe_col_len,
    bool *build_ht,
    memory_manager &gpu_allocator,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    std::vector<sycl::event> events(dependencies);
    int ht_len = build_max_value - build_min_value + 1;
    bool *ht;

    if (build_ht != nullptr)
    {
        ht = build_ht;
    }
    else
    {
        ht = gpu_allocator.alloc_zero<bool>(ht_len);

        auto e2 = build_keys_ht(build_col, build_flags, build_col_len, ht, ht_len, build_min_value, queue, events);
        events = { e2 };

        #if PRINT_JOIN_DEBUG_INFO
        std::cout << "JOIN ht built in FILTER JOIN op" << std::endl;
        #endif
    }

    return filter_join(
        probe_col,
        probe_col_flags,
        probe_col_len,
        ht,
        build_min_value,
        build_max_value,
        queue,
        events
    );
}

class FullJoinKernel : public KernelDefinition
{
private:
    const int *probe_col;
    int *probe_val_out;
    bool *probe_flags;
    const int *ht;
    int ht_len, ht_min_value, ht_max_value;
public:
    FullJoinKernel(
        const int *probe_column,
        int *probe_value_output,
        bool *probe_column_flags,
        const int *hash_table,
        int ht_min,
        int ht_max,
        int col_len)
        : KernelDefinition(col_len), probe_col(probe_column), probe_val_out(probe_value_output),
        probe_flags(probe_column_flags), ht(hash_table), ht_min_value(ht_min), ht_max_value(ht_max)
    {
        ht_len = ht_max - ht_min + 1;
    }

    void operator()(sycl::id<1> idx) const
    {
        auto i = idx[0];
        if (probe_flags[i])
        {
            int hash = HASH(probe_col[i], ht_len, ht_min_value) << 1;
            if (probe_col[i] >= ht_min_value &&
                probe_col[i] <= ht_max_value &&
                ht[hash] == 1)
            {
                probe_val_out[i] = ht[hash + 1]; // save the value to group by on
            }
            else
            {
                probe_flags[i] = false; // mark as not selected
            }
        }
    }
};

sycl::event full_join(
    const int *probe_col,
    int *probe_val_out,
    bool *probe_flags,
    int probe_col_len,
    const int *ht,
    int ht_min,
    int ht_max,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies
)
{
    int ht_len = ht_max - ht_min + 1;

    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                sycl::range<1>{(unsigned long)probe_col_len},
                [=](sycl::id<1> idx)
                {
                    auto i = idx[0];
                    if (probe_flags[i])
                    {
                        int hash = HASH(probe_col[i], ht_len, ht_min);
                        if (probe_col[i] >= ht_min &&
                            probe_col[i] <= ht_max &&
                            ht[hash << 1] == 1)
                        {
                            probe_val_out[i] = ht[(hash << 1) + 1]; // save the value to group by on
                        }
                        else
                        {
                            probe_flags[i] = false; // mark as not selected
                        }
                    }
                }
            );
        }
    );
}

sycl::event full_join(
    TableData<int> &probe_table,
    TableData<int> &build_table,
    int probe_col_index,
    int build_col_index,
    memory_manager &gpu_allocator,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    std::vector<sycl::event> events(dependencies);
    int build_column = build_table.column_indices.at(build_col_index),
        probe_column = probe_table.column_indices.at(probe_col_index),
        group_by_column = build_table.column_indices.at(build_table.group_by_column),
        build_col_min, build_col_max, ht_len, *ht;

    #if PRINT_JOIN_DEBUG_INFO
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    #endif

    if (build_table.ht != nullptr)
    {
        ht = (int *)build_table.ht;
        build_col_min = build_table.ht_min;
        build_col_max = build_table.ht_max;
        ht_len = build_col_max - build_col_min + 1;
    }
    else
    {
        build_col_min = build_table.columns[build_column].min_value;
        build_col_max = build_table.columns[build_column].max_value;
        ht_len = build_col_max - build_col_min + 1;

        ht = gpu_allocator.alloc_zero<int>(ht_len * 2);

        #if PRINT_JOIN_DEBUG_INFO
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff1 = end - start;

        start = std::chrono::high_resolution_clock::now();
        #endif

        auto e2 = build_key_vals_ht(
            build_table.columns[build_column].content,
            build_table.columns[group_by_column].content,
            build_table.flags, build_table.col_len, ht, ht_len,
            build_col_min, queue, events);

        #if PRINT_JOIN_DEBUG_INFO
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff2 = end - start;

        std::cout << "JOIN ht built in FULL JOIN op\n"
            << "diff1 (" << ht_len << "): " << diff1.count() << " ms\n"
            << "diff2: " << diff2.count() << " ms\n";
        #endif

        events = { e2 };
    }

    bool *probe_flags = probe_table.flags;
    int *probe_content = probe_table.columns[probe_column].content;

    #if PRINT_JOIN_DEBUG_INFO
    start = std::chrono::high_resolution_clock::now();
    #endif

    auto e3 = full_join(
        probe_content,
        probe_content,
        probe_flags,
        probe_table.col_len,
        ht,
        build_col_min,
        build_col_max,
        queue,
        events
    );

    #if PRINT_JOIN_DEBUG_INFO
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff3 = end - start;

    start = std::chrono::high_resolution_clock::now();
    #endif

    // the group by column index must refer to the old foreign key
    probe_table.column_indices.erase(probe_col_index);
    probe_table.column_indices[probe_table.col_number + build_table.group_by_column] = probe_column;

    // update min and max values of the probe column
    probe_table.columns[probe_column].min_value = build_table.columns[group_by_column].min_value;
    probe_table.columns[probe_column].max_value = build_table.columns[group_by_column].max_value;

    #if PRINT_JOIN_DEBUG_INFO
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff4 = end - start;

    std::cout << "diff3: " << diff3.count() << " ms\n"
        << "diff4: " << diff4.count() << " ms"
        << std::endl;
    #endif

    return e3;
}