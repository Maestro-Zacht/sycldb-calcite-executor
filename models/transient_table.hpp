#pragma once

#include <sycl/sycl.hpp>

#include "table.hpp"
#include "../operations/memory_manager.hpp"
#include "../gen-cpp/calciteserver_types.h"


class TransientTable
{
private:
    bool *flags_cpu, *flags_gpu;
    Table *base_table;
    sycl::queue gpu_queue, cpu_queue;
public:
    TransientTable(Table *base_table, sycl::queue &gpu_queue, sycl::queue &cpu_queue)
        : base_table(base_table), gpu_queue(gpu_queue), cpu_queue(cpu_queue)
    {
        uint64_t nrows = base_table->get_nrows();

        flags_gpu = sycl::malloc_device<bool>(nrows, gpu_queue);
        auto e1 = gpu_queue.fill<bool>(flags_gpu, true, nrows);

        flags_cpu = sycl::malloc_device<bool>(nrows, cpu_queue);
        auto e2 = cpu_queue.fill<bool>(flags_cpu, true, nrows);

        e1.wait();
        e2.wait();
    }

    ~TransientTable()
    {
        sycl::free(flags_gpu, gpu_queue);
        sycl::free(flags_cpu, cpu_queue);
    }

    std::vector<sycl::event> apply_filter(
        const ExprType &expr,
        std::string parent_op,
        memory_manager &gpu_allocator,
        memory_manager &cpu_allocator,
        const std::vector<sycl::event> &dependencies)
    {
        // Recursive parsing of EXRP types. LITERAL and COLUMN are handled in parent EXPR type.
        if (expr.exprType != ExprOption::EXPR)
        {
            std::cerr << "Filter condition: Unsupported parsing ExprType " << expr.exprType << std::endl;
            return {};
        }

        std::vector<sycl::event> events;

        if (expr.op == "SEARCH")
        {

        }
    }
};