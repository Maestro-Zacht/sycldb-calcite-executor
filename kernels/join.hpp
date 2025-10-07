#pragma once

#include <sycl/sycl.hpp>

#include "types.hpp"

#define PRINT_JOIN_DEBUG_INFO 0

template <typename T>
inline T HASH(T X, T Y, T Z)
{
    return ((X - Z) % Y);
}

template <typename T>
sycl::event build_keys_ht(
    T col[],
    bool flags[],
    int col_len,
    bool ht[],
    T ht_len,
    T ht_min_value,
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

sycl::event build_key_vals_ht(
    int col[],
    int agg_col[],
    bool flags[],
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

template <typename T>
sycl::event filter_join(
    T build_col[],
    bool build_flags[],
    int build_col_len,
    T build_max_value,
    T build_min_value,
    T probe_col[],
    bool probe_col_flags[],
    int probe_col_len,
    bool *build_ht,
    std::vector<void *> &resources,
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
        ht = sycl::malloc_device<bool>(ht_len, queue);
        auto e1 = queue.fill(ht, false, ht_len);
        events.push_back(e1);

        auto e2 = build_keys_ht(build_col, build_flags, build_col_len, ht, ht_len, build_min_value, queue, events);
        events = { e2 };

        #if PRINT_JOIN_DEBUG_INFO
        std::cout << "JOIN ht built in FILTER JOIN op" << std::endl;
        #endif
    }



    auto e3 = queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(events);
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
                        probe_col_flags[i] = ht[HASH(probe_col[i], ht_len, build_min_value)];
                }
            );
        }
    );

    if (build_ht == nullptr)
        resources.push_back(ht);

    return e3;
}

sycl::event full_join(
    TableData<int> &probe_table,
    TableData<int> &build_table,
    int probe_col_index,
    int build_col_index,
    std::vector<void *> &resources,
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

        ht = sycl::malloc_device<int>(ht_len * 2, queue);

        auto e1 = queue.fill(ht, 0, ht_len * 2);
        events.push_back(e1);

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

    auto e3 = queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(events);
            cgh.parallel_for(
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
                }
            );
        }
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

    if (build_table.ht == nullptr)
        resources.push_back(ht);

    #if PRINT_JOIN_DEBUG_INFO
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff4 = end - start;

    std::cout << "diff3: " << diff3.count() << " ms\n"
        << "diff4: " << diff4.count() << " ms"
        << std::endl;
    #endif

    return e3;
}