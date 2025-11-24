#pragma once

#include <vector>
#include <memory>
#include "../kernels/common.hpp"
#include "../kernels/selection.hpp"
#include "../kernels/projection.hpp"
#include "../kernels/aggregation.hpp"
#include "../kernels/join.hpp"

enum class KernelType : uint8_t
{
    EmptyKernel,
    LogicalKernel,
    SelectionKernelColumns,
    SelectionKernelLiteral,
    FillKernel,
    PerformOperationKernelColumns,
    PerformOperationKernelLiteralFirst,
    PerformOperationKernelLiteralSecond,
    BuildKeysHTKernel,
    FilterJoinKernel,
    BuildKeyValsHTKernel,
    FullJoinKernel,
    AggregateOperationKernel,
    GroupByAggregateKernel,
    SyncFlagsKernel,
    CopyFlagsKernel,
};

class KernelData
{
private:
    KernelType kernel_type;
    std::shared_ptr<KernelDefinition> kernel_def;
public:
    KernelData(KernelType kt, KernelDefinition *kd)
        : kernel_type(kt), kernel_def(std::shared_ptr<KernelDefinition>(kd))
    {}

    sycl::event execute(
        sycl::queue gpu_queue,
        sycl::queue cpu_queue,
        const std::vector<sycl::event> &gpu_dependencies,
        const std::vector<sycl::event> &cpu_dependencies,
        bool on_device
    ) const
    {
        sycl::queue queue = on_device ? gpu_queue : cpu_queue;
        const std::vector<sycl::event> &dependencies = on_device ? gpu_dependencies : cpu_dependencies;

        switch (kernel_type)
        {
        case KernelType::EmptyKernel:
        {
            EmptyKernel *kernel = static_cast<EmptyKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.single_task(*kernel);
                }
            );
        }
        case KernelType::LogicalKernel:
        {
            LogicalKernel *kernel = static_cast<LogicalKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::SelectionKernelColumns:
        {
            SelectionKernelColumns *kernel = static_cast<SelectionKernelColumns *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::SelectionKernelLiteral:
        {
            SelectionKernelLiteral *kernel = static_cast<SelectionKernelLiteral *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::FillKernel:
        {
            FillKernel *kernel = static_cast<FillKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::PerformOperationKernelColumns:
        {
            PerformOperationKernelColumns *kernel = static_cast<PerformOperationKernelColumns *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::PerformOperationKernelLiteralFirst:
        {
            PerformOperationKernelLiteralFirst *kernel = static_cast<PerformOperationKernelLiteralFirst *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::PerformOperationKernelLiteralSecond:
        {
            PerformOperationKernelLiteralSecond *kernel = static_cast<PerformOperationKernelLiteralSecond *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::BuildKeysHTKernel:
        {
            BuildKeysHTKernel *kernel = static_cast<BuildKeysHTKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::FilterJoinKernel:
        {
            FilterJoinKernel *kernel = static_cast<FilterJoinKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::BuildKeyValsHTKernel:
        {
            BuildKeyValsHTKernel *kernel = static_cast<BuildKeyValsHTKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::FullJoinKernel:
        {
            FullJoinKernel *kernel = static_cast<FullJoinKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::AggregateOperationKernel:
        {
            AggregateOperationKernel *kernel = static_cast<AggregateOperationKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::GroupByAggregateKernel:
        {
            GroupByAggregateKernel *kernel = static_cast<GroupByAggregateKernel *>(kernel_def.get());
            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
        }
        case KernelType::SyncFlagsKernel:
        {
            SyncFlagsKernel *kernel = static_cast<SyncFlagsKernel *>(kernel_def.get());

            bool *src = kernel->get_src();
            bool *tmp = kernel->get_tmp();
            int len = kernel->get_col_len();

            sycl::event e = gpu_queue.memcpy(
                tmp, src, len * sizeof(bool),
                on_device ? cpu_dependencies : gpu_dependencies
            );

            return queue.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(dependencies);
                    cgh.depends_on(e);
                    cgh.parallel_for(len, *kernel);
                }
            );
        }
        case KernelType::CopyFlagsKernel:
        {
            CopyFlagsKernel *kernel = static_cast<CopyFlagsKernel *>(kernel_def.get());

            bool *src = kernel->get_src();
            bool *dst = kernel->get_dst();
            int len = kernel->get_col_len();

            return queue.memcpy(dst, src, len * sizeof(bool), dependencies);
        }
        default:
            std::cerr << "Unknown kernel type in KernelData::execute()" << std::endl;
            throw std::invalid_argument("Unknown kernel type");
        }
    }
};

class KernelBundle
{
private:
    std::vector<KernelData> kernels;
    bool on_device;
public:
    KernelBundle(bool on_device)
        : on_device(on_device)
    {}

    bool is_on_device() const
    {
        return on_device;
    }

    void add_kernel(KernelData kernel)
    {
        kernels.push_back(kernel);
    }

    sycl::event execute(
        sycl::queue gpu_queue,
        sycl::queue cpu_queue,
        const std::vector<sycl::event> &gpu_dependencies,
        const std::vector<sycl::event> &cpu_dependencies
    ) const
    {
        std::vector<sycl::event> deps = on_device ? gpu_dependencies : cpu_dependencies;
        sycl::event e;

        for (const KernelData &kernel : kernels)
        {
            e = kernel.execute(
                gpu_queue,
                cpu_queue,
                on_device ? deps : gpu_dependencies,
                on_device ? cpu_dependencies : deps,
                on_device
            );
            deps.clear();
            deps.push_back(e);
        }

        return e;
    }
};