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
    AggregateOperationKernel
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

    sycl::event execute(sycl::queue &queue, const std::vector<sycl::event> &dependencies) const
    {
        switch (kernel_type)
        {
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
public:
    void add_kernel(KernelData kernel)
    {
        kernels.push_back(kernel);
    }

    sycl::event execute(sycl::queue &queue, const std::vector<sycl::event> &dependencies) const
    {
        std::vector<sycl::event> deps = dependencies;
        sycl::event e;

        for (const KernelData &kernel : kernels)
        {
            e = kernel.execute(queue, deps);
            deps.clear();
            deps.push_back(e);
        }

        return e;
    }
};