#pragma once

#include <sycl/sycl.hpp>

#define ALLOC_ON_HOST 0

class memory_manager
{
private:
    char *memory_region, *current_free;
    uint64_t size, allocated;
    sycl::queue &queue;
public:
    memory_manager(sycl::queue &queue, uint64_t size);
    ~memory_manager();

    template <typename T>
    T *alloc(uint64_t count);
};

memory_manager::memory_manager(sycl::queue &queue, uint64_t size)
    : size(size), allocated(0), queue(queue)
{
    memory_region =
        #if ALLOC_ON_HOST
        sycl::malloc_host<char>
        #else
        sycl::malloc_device<char>
        #endif
        (size, queue);
    queue.memset(memory_region, 0, size).wait();
    current_free = memory_region;
}


memory_manager::~memory_manager()
{
    sycl::free(memory_region, queue);
}

template <typename T>
T *memory_manager::alloc(uint64_t count)
{
    uint64_t bytes = count * sizeof(T);

    if (allocated + bytes > size)
    {
        throw std::bad_alloc();
    }

    T *ptr = reinterpret_cast<T *>(current_free);

    current_free += bytes;
    allocated += bytes;

    return ptr;
}