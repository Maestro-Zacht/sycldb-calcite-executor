#pragma once

#include <sycl/sycl.hpp>

class memory_manager
{
private:
    char *memory_region, *current_free;
    uint64_t size, allocated;
    sycl::queue &queue;
public:
    memory_manager(sycl::queue &queue, uint64_t size, bool alloc_on_host);
    ~memory_manager();

    template <typename T>
    T *alloc(uint64_t count);
};

memory_manager::memory_manager(sycl::queue &queue, uint64_t size, bool alloc_on_host)
    : size(size), allocated(0), queue(queue)
{

    if (alloc_on_host)
        memory_region = sycl::malloc_host<char>(size, queue);
    else
        memory_region = sycl::malloc_device<char>(size, queue);

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

    bytes = (bytes + 7) & (~7); // align to 8 bytes

    if (allocated + bytes > size)
    {
        std::cerr << "Memory manager out of memory: requested " << bytes << " bytes, "
            << (size - allocated) << " bytes available." << std::endl;
        throw std::bad_alloc();
    }

    T *ptr = reinterpret_cast<T *>(current_free);

    current_free += bytes;
    allocated += bytes;

    return ptr;
}
