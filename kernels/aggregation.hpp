#pragma once

#include <sycl/sycl.hpp>

#include "types.hpp"

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
void perform_operation(T result[], const T a[], const T b[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    // for (int i = 0; i < size; i++)
    BinaryOp op_enum = get_op_from_string(op);
    queue.parallel_for(size, [=](sycl::id<1> i)
                       {
        if (flags[i])
            result[i] = element_operation(a[i], b[i], op_enum); });
}

template <typename T>
void perform_operation(T result[], T a, const T b[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    // for (int i = 0; i < size; i++)
    BinaryOp op_enum = get_op_from_string(op);
    queue.parallel_for(size, [=](sycl::id<1> i)
                       {
        if (flags[i])
            result[i] = element_operation(a, b[i], op_enum); });
}

template <typename T>
void perform_operation(T result[], const T a[], T b, bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    // for (int i = 0; i < size; i++)
    BinaryOp op_enum = get_op_from_string(op);
    queue.parallel_for(size, [=](sycl::id<1> i)
                       {
        if (flags[i])
            result[i] = element_operation(a[i], b, op_enum); });
}

template <typename T, typename U>
void aggregate_operation(U *result, const T a[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    auto start = std::chrono::high_resolution_clock::now();
    U *result_d = sycl::malloc_device<U>(1, queue);
    queue.memset(result_d, 0, sizeof(U)).wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memset_time = end - start;
    std::cout << "Memset time: " << memset_time.count() << " ms" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    queue.parallel_for(
             size,
             sycl::reduction(result_d, sycl::plus<>()),
             [=](sycl::id<1> idx, auto &sum)
             {
                 if (flags[idx])
                 {
                     sum.combine(a[idx]);
                 }
             })
        .wait();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_time = end - start;
    std::cout << "Aggregate kernel time: " << kernel_time.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    queue.memcpy(result, result_d, sizeof(U)).wait();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memcpy_time = end - start;
    sycl::free(result_d, queue);
    std::cout << "Memcpy time: " << memcpy_time.count() << " ms" << std::endl;
}

/*unsigned long long aggregate_operation(const int a[], bool flags[], int size, const std::string &op, sycl::queue &queue)
{
    unsigned long long result = 0;
    unsigned long long *d_result = sycl::malloc_device<unsigned long long>(1, queue);
    queue.memset(d_result, 0, sizeof(unsigned long long)).wait();
    queue.parallel_for(size, sycl::reduction(d_result, sycl::plus<>()), [=](sycl::id<1> idx, auto &sum) {
        if (flags[idx]) { sum.combine(a[idx]); }
    });
    queue.memcpy(&result, d_result, sizeof(unsigned long long)).wait();
    sycl::free(d_result, queue);
    return result;
}*/

std::tuple<int *, unsigned long long, bool *, uint64_t *> group_by_aggregate(ColumnData<int> *group_columns, int *agg_column, bool *flags, int col_num, int col_len, const std::string &agg_op, sycl::queue &queue)
{
    unsigned long long prod_ranges = 1;

    for (int i = 0; i < col_num; i++)
    {
        prod_ranges *= group_columns[i].max_value - group_columns[i].min_value + 1;
    }

    int *results = sycl::malloc_shared<int>(col_num * prod_ranges, queue);
    queue.fill(results, 0, col_num * prod_ranges).wait();

    uint64_t *agg_result = sycl::malloc_shared<uint64_t>(prod_ranges, queue);
    queue.fill(agg_result, (uint64_t)0, prod_ranges).wait();

    unsigned *res_flags = sycl::malloc_shared<unsigned>(prod_ranges, queue);
    queue.fill(res_flags, (unsigned)0, prod_ranges).wait();

    queue.parallel_for(
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

                     sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>
                         flag_obj(res_flags[hash]);
                     if (flag_obj.fetch_add(1) == 0)
                     {
                         for (int j = 0; j < col_num; j++)
                             results[j * prod_ranges + hash] = group_columns[j].content[i];
                     }

                     // if (agg_op == "SUM")
                     auto sum_obj =
                         sycl::atomic_ref<uint64_t,
                                          sycl::memory_order::relaxed,
                                          sycl::memory_scope::device,
                                          sycl::access::address_space::global_space>(agg_result[hash]);
                     sum_obj.fetch_add(agg_column[i]);
                 }
             })
        .wait();

    // for (int i = 0; i < prod_ranges; i++)
    //     std::cout << i << " :: " << agg_result[i] << std::endl;

    // std::cout << "prod ranges :: " << prod_ranges << "\nStarting group by aggregation kernel..." << std::endl;

    // for (int i = 0; i < col_len; i++)
    // {
    //     if (flags[i])
    //     {
    //         unsigned hash = 0, mult = 1;
    //         for (int j = 0; j < col_num; j++)
    //         {
    //             hash += (group_columns[j].content[i] - group_columns[j].min_value) * mult;
    //             mult *= group_columns[j].max_value - group_columns[j].min_value + 1;
    //         }
    //         hash %= prod_ranges;

    //         if (hash == 99)
    //         {
    //             if (group_columns[1].content[i] == 1995 && group_columns[0].content[i] == 24)
    //             {
    //                 std::cout << agg_column[i] << " -> " << agg_result[hash] << " (" << hash << ")" << std::endl;
    //             }
    //             else
    //             {
    //                 std::cout << "Unexpected group by result: " << group_columns[0].content[i] << ", " << group_columns[1].content[i] << " -> " << agg_column[i] << " (" << hash << ")" << std::endl;
    //             }
    //         }

    //         if (res_flags[hash] == 0)
    //         {
    //             res_flags[hash] = 1;
    //             for (int j = 0; j < col_num; j++)
    //                 results[j * prod_ranges + hash] = group_columns[j].content[i];
    //         }

    //         if (agg_op == "SUM")
    //             agg_result[hash] += agg_column[i];
    //         else
    //         {
    //             // std::cout << "Unsupported aggregate operation: " << agg_op << std::endl;
    //         }
    //     }
    // }

    // int *h_results = sycl::malloc_shared<int>((col_num + (sizeof(uint64_t) / sizeof(int))) * prod_ranges, queue);
    // bool *h_res_flags = sycl::malloc_shared<bool>(prod_ranges, queue);
    // queue.memcpy(h_results, results, sizeof(int) * (col_num + (sizeof(uint64_t) / sizeof(int))) * prod_ranges).wait();
    // queue.memcpy(h_res_flags, res_flags, sizeof(bool) * prod_ranges).wait();
    // sycl::free(results, queue);
    // sycl::free(res_flags, queue);

    bool *final_flags = sycl::malloc_shared<bool>(prod_ranges, queue);
    queue.parallel_for(
             prod_ranges,
             [=](sycl::id<1> idx)
             {
                 final_flags[idx] = res_flags[idx] != 0;
             })
        .wait();
    sycl::free(res_flags, queue);

    return std::make_tuple(results, prod_ranges, final_flags, agg_result);
}