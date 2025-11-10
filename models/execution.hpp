#pragma once

#include <vector>
#include <memory>
#include "../kernels/common.hpp"

class KernelBundle
{
private:
    std::vector<std::unique_ptr<KernelDefinition>> kernels;
public:
    void add_kernel(std::unique_ptr<KernelDefinition> kernel)
    {
        kernels.push_back(std::move(kernel));
    }
};