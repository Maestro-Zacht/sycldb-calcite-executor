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

    std::vector<sycl::event> execute(
        sycl::queue &queue,
        const std::vector<sycl::event> &dependencies
    ) const
    {
        // std::cout << "  - Executing kernel of type "
        //     << static_cast<int>(kernel_type)
        //     << std::endl;

        switch (kernel_type)
        {
        case KernelType::EmptyKernel:
        {
            return dependencies;
        }
        case KernelType::LogicalKernel:
        {
            LogicalKernel *kernel = static_cast<LogicalKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::SelectionKernelColumns:
        {
            SelectionKernelColumns *kernel = static_cast<SelectionKernelColumns *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::SelectionKernelLiteral:
        {
            SelectionKernelLiteral *kernel = static_cast<SelectionKernelLiteral *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::FillKernel:
        {
            FillKernel *kernel = static_cast<FillKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::PerformOperationKernelColumns:
        {
            PerformOperationKernelColumns *kernel = static_cast<PerformOperationKernelColumns *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::PerformOperationKernelLiteralFirst:
        {
            PerformOperationKernelLiteralFirst *kernel = static_cast<PerformOperationKernelLiteralFirst *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::PerformOperationKernelLiteralSecond:
        {
            PerformOperationKernelLiteralSecond *kernel = static_cast<PerformOperationKernelLiteralSecond *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::BuildKeysHTKernel:
        {
            BuildKeysHTKernel *kernel = static_cast<BuildKeysHTKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::FilterJoinKernel:
        {
            FilterJoinKernel *kernel = static_cast<FilterJoinKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::BuildKeyValsHTKernel:
        {
            BuildKeyValsHTKernel *kernel = static_cast<BuildKeyValsHTKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::FullJoinKernel:
        {
            FullJoinKernel *kernel = static_cast<FullJoinKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::AggregateOperationKernel:
        {
            AggregateOperationKernel *kernel = static_cast<AggregateOperationKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::GroupByAggregateKernel:
        {
            GroupByAggregateKernel *kernel = static_cast<GroupByAggregateKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
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
    int device_index;
public:
    KernelBundle(bool on_device, int device_index)
        : on_device(on_device), device_index(device_index)
    {}

    bool is_on_device() const
    {
        return on_device;
    }

    int get_device_index() const
    {
        return device_index;
    }

    void add_kernel(KernelData kernel)
    {
        kernels.push_back(kernel);
    }

    std::vector<sycl::event> execute(
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        const std::vector<sycl::event> &cpu_dependencies,
        const std::vector<std::vector<sycl::event>> &device_dependencies
    ) const
    {
        std::vector<sycl::event> deps = on_device ? device_dependencies[device_index] : cpu_dependencies;

        for (const KernelData &kernel : kernels)
        {
            deps = kernel.execute(
                on_device ? device_queues[device_index] : cpu_queue,
                deps
            );
            // sycl::event::wait(deps);
            // std::cout << "    - Kernel executed" << std::endl;
        }

        return deps;
    }
};