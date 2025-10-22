#pragma once

#include <sycl/sycl.hpp>

#include "types.hpp"
#include "../operations/memory_manager.hpp"

#define PRINT_AGGREGATE_DEBUG_INFO 0

enum class BinaryOp : uint8_t
{
    Multiply,
    Divide,
    Add,
    Subtract
};

template <typename T>
inline T element_operation(T a, T b, BinaryOp op)
{
    switch (op)
    {
    case BinaryOp::Multiply:
        return a * b;
    case BinaryOp::Divide:
        return a / b;
    case BinaryOp::Add:
        return a + b;
    case BinaryOp::Subtract:
        return a - b;
    default:
        return 0;
    }
}

BinaryOp get_op_from_string(const std::string &op)
{
    if (op == "*")
        return BinaryOp::Multiply;
    if (op == "/")
        return BinaryOp::Divide;
    if (op == "+")
        return BinaryOp::Add;
    if (op == "-")
        return BinaryOp::Subtract;
    throw std::invalid_argument("Unknown operation: " + op);
}

template <typename T>
sycl::event perform_operation(
    T result[],
    const T a[],
    const T b[],
    const bool flags[],
    int size,
    const std::string &op,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    BinaryOp op_enum = get_op_from_string(op);

    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                size,
                [=](sycl::id<1> i)
                {
                    if (flags[i])
                        result[i] = element_operation(a[i], b[i], op_enum);
                }
            );
        }
    );
}

template <typename T>
sycl::event perform_operation(
    T result[],
    T a,
    const T b[],
    const bool flags[],
    int size,
    const std::string &op,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    BinaryOp op_enum = get_op_from_string(op);

    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                size,
                [=](sycl::id<1> i)
                {
                    if (flags[i])
                        result[i] = element_operation(a, b[i], op_enum);
                }
            );
        }
    );
}

template <typename T>
sycl::event perform_operation(
    T result[],
    const T a[],
    T b,
    const bool flags[],
    int size,
    const std::string &op,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    BinaryOp op_enum = get_op_from_string(op);

    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                size,
                [=](sycl::id<1> i)
                {
                    if (flags[i])
                        result[i] = element_operation(a[i], b, op_enum);
                }
            );
        }
    );
}

template <typename T, typename U>
sycl::event aggregate_operation(
    U *result,
    const T a[],
    bool flags[],
    int size,
    const std::string &op,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    #if PRINT_AGGREGATE_DEBUG_INFO
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    auto e1 = queue.memset(result, 0, sizeof(U));

    #if PRINT_AGGREGATE_DEBUG_INFO
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memset_time = end - start;
    std::cout << "Memset time: " << memset_time.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    #endif

    auto e2 = queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.depends_on(e1);
            cgh.parallel_for(
                sycl::range<1>(size),
                sycl::reduction(result, sycl::plus<>()),
                [=](sycl::id<1> idx, auto &sum)
                {
                    if (flags[idx])
                        sum.combine(a[idx]);
                }
            );
        }
    );

    #if PRINT_AGGREGATE_DEBUG_INFO
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_time = end - start;
    std::cout << "Aggregate kernel time: " << kernel_time.count() << " ms" << std::endl;
    #endif

    return e2;
}

std::tuple<
    int **,
    unsigned long long,
    bool *,
    uint64_t *,
    sycl::event
> group_by_aggregate(
    ColumnData<int> *group_columns,
    int *agg_column,
    bool *flags,
    int col_num,
    int col_len,
    const std::string &agg_op,
    memory_manager &gpu_allocator,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    unsigned long long prod_ranges = 1;
    std::vector<sycl::event> events(dependencies);
    events.reserve(dependencies.size() + col_num + 2);

    #if PRINT_AGGREGATE_DEBUG_INFO
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    for (int i = 0; i < col_num; i++)
    {
        prod_ranges *= group_columns[i].max_value - group_columns[i].min_value + 1;
    }

    #if PRINT_AGGREGATE_DEBUG_INFO
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> range_time = end - start;
    std::cout << "Range calculation time: " << range_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    #endif

    int **results = sycl::malloc_shared<int *>(col_num, queue);
    for (int i = 0; i < col_num; i++)
    {
        results[i] = gpu_allocator.alloc<int>(prod_ranges);
    }

    uint64_t *agg_result = gpu_allocator.alloc<uint64_t>(prod_ranges);
    unsigned *res_flags = gpu_allocator.alloc<unsigned>(prod_ranges);

    #if PRINT_AGGREGATE_DEBUG_INFO
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> alloc_time = end - start;
    std::cout << "Allocation and memset time: " << alloc_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    #endif

    auto e4 = queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(events);
            cgh.parallel_for(
                col_len,
                [=](sycl::id<1> idx)
                {
                    auto i = idx[0];
                    if (flags[i])
                    {
                        int hash = 0, mult = 1;
                        for (int j = 0; j < col_num; j++)
                        {
                            hash += (group_columns[j].content[i] - group_columns[j].min_value) * mult;
                            mult *= group_columns[j].max_value - group_columns[j].min_value + 1;
                        }
                        hash %= prod_ranges;

                        sycl::atomic_ref<
                            unsigned,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space
                        > flag_obj(res_flags[hash]);
                        if (flag_obj.fetch_add(1) == 0)
                        {
                            for (int j = 0; j < col_num; j++)
                                results[j][hash] = group_columns[j].content[i];
                        }

                        // if (agg_op == "SUM")

                        sycl::atomic_ref<
                            uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space
                        > sum_obj(agg_result[hash]);
                        sum_obj.fetch_add(agg_column[i]);
                    }
                }
            );
        }
    );

    #if PRINT_AGGREGATE_DEBUG_INFO
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_time = end - start;
    std::cout << "Group by aggregation kernel time: " << kernel_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    #endif

    bool *final_flags = gpu_allocator.alloc<bool>(prod_ranges);
    auto e5 = queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(e4);
            cgh.parallel_for(
                prod_ranges,
                [=](sycl::id<1> idx)
                {
                    final_flags[idx] = res_flags[idx] != 0;
                }
            );
        }
    );

    #if PRINT_AGGREGATE_DEBUG_INFO
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> flag_time = end - start;
    std::cout << "Final flags kernel time: " << flag_time.count() << " ms" << std::endl;
    #endif

    return std::make_tuple(results, prod_ranges, final_flags, agg_result, e5);
}