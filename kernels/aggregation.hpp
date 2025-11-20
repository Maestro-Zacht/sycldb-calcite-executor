#pragma once

#include <sycl/sycl.hpp>

#include "types.hpp"
#include "../operations/memory_manager.hpp"
#include "common.hpp"

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

class PerformOperationKernelColumns : public KernelDefinition
{
private:
    int *result;
    const int *col1, *col2;
    const bool *flags;
    BinaryOp op_enum;
public:
    PerformOperationKernelColumns(int *result, const int *col1, const int *col2, const bool *flags, BinaryOp op, int col_len)
        : KernelDefinition(col_len), result(result), col1(col1), col2(col2), flags(flags), op_enum(op)
    {}

    PerformOperationKernelColumns(int *res, const int *a, const int *b, const bool *flgs, const std::string &op, int col_len)
        : KernelDefinition(col_len), result(res), col1(a), col2(b), flags(flgs), op_enum(get_op_from_string(op))
    {}

    void operator()(sycl::id<1> idx) const
    {
        if (flags[idx])
        {
            result[idx] = element_operation(col1[idx], col2[idx], op_enum);
        }
    }
};

sycl::event perform_operation(
    int result[],
    const int a[],
    const int b[],
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

class PerformOperationKernelLiteralFirst : public KernelDefinition
{
private:
    int *result;
    int literal;
    const int *col;
    const bool *flags;
    BinaryOp op_enum;
public:
    PerformOperationKernelLiteralFirst(int *res, int lit, const int *column, const bool *flgs, BinaryOp op, int col_len)
        : KernelDefinition(col_len), result(res), literal(lit), col(column), flags(flgs), op_enum(op)
    {}

    PerformOperationKernelLiteralFirst(int *res, int lit, const int *column, const bool *flgs, const std::string &op, int col_len)
        : KernelDefinition(col_len), result(res), literal(lit), col(column), flags(flgs), op_enum(get_op_from_string(op))
    {}

    void operator()(sycl::id<1> idx) const
    {
        if (flags[idx])
        {
            result[idx] = element_operation(literal, col[idx], op_enum);
        }
    }
};

sycl::event perform_operation(
    int result[],
    int a,
    const int b[],
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

class PerformOperationKernelLiteralSecond : public KernelDefinition
{
private:
    int *result;
    const int *col;
    int literal;
    const bool *flags;
    BinaryOp op_enum;
public:
    PerformOperationKernelLiteralSecond(int *res, const int *column, int lit, const bool *flgs, BinaryOp op, int col_len)
        : KernelDefinition(col_len), result(res), col(column), literal(lit), flags(flgs), op_enum(op)
    {}

    PerformOperationKernelLiteralSecond(int *res, const int *column, int lit, const bool *flgs, const std::string &op, int col_len)
        : KernelDefinition(col_len), result(res), col(column), literal(lit), flags(flgs), op_enum(get_op_from_string(op))
    {}

    void operator()(sycl::id<1> idx) const
    {
        if (flags[idx])
        {
            result[idx] = element_operation(col[idx], literal, op_enum);
        }
    }
};

sycl::event perform_operation(
    int result[],
    const int a[],
    int b,
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

class AggregateOperationKernel : public KernelDefinition
{
private:
    const int *data;
    const bool *flags;
    uint64_t *agg_res;
public:
    AggregateOperationKernel(const int *data, const bool *flags, int col_len, uint64_t *agg_res)
        : KernelDefinition(col_len), data(data), flags(flags), agg_res(agg_res)
    {}

    void operator()(sycl::id<1> idx) const
    {
        if (flags[idx])
        {
            sycl::atomic_ref<
                uint64_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space
            > sum_obj(*agg_res);
            sum_obj.fetch_add(data[idx]);
        }
    }
};

sycl::event aggregate_operation(
    const int a[],
    const bool flags[],
    int size,
    uint64_t *agg_res,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies)
{
    #if PRINT_AGGREGATE_DEBUG_INFO
    auto start = std::chrono::high_resolution_clock::now();
    #endif


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
            cgh.parallel_for(
                sycl::range<1>(size),
                // agg,
                [=](sycl::id<1> idx)
                {
                    if (flags[idx])
                    {
                        // sum.combine(a[idx]);
                        sycl::atomic_ref<
                            uint64_t,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space
                        > sum_obj(*agg_res);
                        sum_obj.fetch_add(a[idx]);
                    }
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

class GroupByAggregateKernel : public KernelDefinition
{
private:
    const int **contents;
    const int *agg_column;
    const int *max;
    const int *min;
    const bool *flags;
    int col_num;
    int **results;
    uint64_t prod_ranges;
    uint64_t *agg_result;
    unsigned *result_flags;
public:
    GroupByAggregateKernel(
        const int **contents,
        const int *agg_column,
        const int *max,
        const int *min,
        const bool *flags,
        int col_num,
        int col_len,
        int **results,
        uint64_t *agg_result,
        unsigned *result_flags,
        uint64_t prod_ranges)
        : KernelDefinition(col_len), contents(contents), agg_column(agg_column),
        max(max), min(min), flags(flags), col_num(col_num),
        results(results), prod_ranges(prod_ranges), agg_result(agg_result),
        result_flags(result_flags)
    {}

    void operator()(sycl::id<1> idx) const
    {
        auto i = idx[0];
        if (flags[i])
        {
            int hash = 0, mult = 1;
            for (int j = 0; j < col_num; j++)
            {
                hash += (contents[j][i] - min[j]) * mult;
                mult *= max[j] - min[j] + 1;
            }
            hash %= prod_ranges;

            sycl::atomic_ref<
                unsigned,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space
            > flag_obj(result_flags[hash]);
            if (flag_obj.exchange(1) == 0)
            {
                for (int j = 0; j < col_num; j++)
                    results[j][hash] = contents[j][i];
            }

            sycl::atomic_ref<
                uint64_t,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space
            > sum_obj(agg_result[hash]);
            sum_obj.fetch_add(agg_column[i]);
        }
    }
};

sycl::event group_by_aggregate(
    const int **contents,
    const int *agg_column,
    const int *max,
    const int *min,
    const bool *flags,
    int col_len,
    int col_num,
    int **results,
    uint64_t *agg_result,
    unsigned *result_flags,
    uint64_t prod_ranges,
    const std::string &agg_op,
    sycl::queue &gpu_queue,
    const std::vector<sycl::event> &dependencies)
{
    return gpu_queue.submit(
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
                        int hash = 0, mult = 1;
                        for (int j = 0; j < col_num; j++)
                        {
                            hash += (contents[j][i] - min[j]) * mult;
                            mult *= max[j] - min[j] + 1;
                        }
                        hash %= prod_ranges;

                        sycl::atomic_ref<
                            unsigned,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space
                        > flag_obj(result_flags[hash]);
                        unsigned expected = 0;
                        bool first = flag_obj.compare_exchange_strong(expected, 1);
                        if (first)
                        {
                            for (int j = 0; j < col_num; j++)
                            {
                                results[j][hash] = contents[j][i];
                            }
                        }

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
        results[i] = gpu_allocator.alloc<int>(prod_ranges, true);
    }

    uint64_t *agg_result = gpu_allocator.alloc<uint64_t>(prod_ranges, true);
    unsigned *res_flags = gpu_allocator.alloc<unsigned>(prod_ranges, true);

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
                        if (flag_obj.exchange(1) == 0)
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

    bool *final_flags = gpu_allocator.alloc<bool>(prod_ranges, true);
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