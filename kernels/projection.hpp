#pragma once

#include <sycl/sycl.hpp>

#include "common.hpp"

class FillKernel : public KernelDefinition
{
private:
    int *target;
    int value;
public:
    FillKernel(int *tgt, int val, int len)
        : KernelDefinition(len), target(tgt), value(val)
    {}

    void operator()(sycl::id<1> idx) const
    {
        target[idx] = value;
    }
};
