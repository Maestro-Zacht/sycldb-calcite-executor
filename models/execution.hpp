#pragma once

#include <vector>
#include <memory>
#include "../kernels/common.hpp"


enum class KernelType : uint8_t
{
    LogicalKernel,
    SelectionKernelColumns,
    SelectionKernelLiteral,
    FillKernel,
    PerformOperationKernelColumns,
    PerformOperationKernelLiteralFirst,
    PerformOperationKernelLiteralSecond
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
};

class KernelBundle
{
private:
    std::vector<KernelData> kernels;
public:
    void add_kernel(KernelData kernel)
    {
        kernels.push_back(std::move(kernel));
    }
};