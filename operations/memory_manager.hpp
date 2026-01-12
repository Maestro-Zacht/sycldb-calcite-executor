#pragma once

#include <sycl/sycl.hpp>

#define MEMORY_MANAGER_DEBUG_INFO 0

class memory_manager
{
private:
    void *memory_region_device, *current_free_device,
        *memory_region_zero_device, *current_free_zero_device,
        *memory_region_host, *current_free_host;
    uint64_t size_device, allocated_device,
        size_zero_device, allocated_zero_device,
        size_host, allocated_host;
    sycl::queue &queue;
public:
    memory_manager(sycl::queue &queue, uint64_t size);
    ~memory_manager();

    template <typename T>
    T *alloc(uint64_t count, bool on_device);

    template <typename T>
    T *alloc_zero(uint64_t count);

    sycl::event reset();
};

memory_manager::memory_manager(sycl::queue &queue, uint64_t size)
    : size_device(size), size_zero_device(size >> 6), size_host(size), queue(queue)
{
    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Allocating memory region of size " << size_device << " bytes on device + " << size_zero_device << " bytes zero on device and " << size_host << " bytes on host." << std::endl;
    #endif

    memory_region_device = sycl::malloc_device<uint8_t>(size_device, queue);
    memory_region_zero_device = sycl::malloc_device<uint8_t>(size_zero_device, queue);
    memory_region_host = sycl::malloc_host<uint8_t>(size_host, queue);

    reset().wait();
}


memory_manager::~memory_manager()
{
    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Freeing memory region of size "
        << (allocated_device >> 20) << "/" << (size_device >> 20) << " MB on device + "
        << (allocated_zero_device >> 20) << "/" << (size_zero_device >> 20) << " MB zero on device and "
        << (allocated_host >> 20) << "/" << (size_host >> 20) << " MB on host."
        << std::endl;
    #endif

    sycl::free(memory_region_device, queue);
    sycl::free(memory_region_zero_device, queue);
    sycl::free(memory_region_host, queue);

    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Memory regions freed." << std::endl;
    #endif
}

template <typename T>
T *memory_manager::alloc(uint64_t count, bool on_device)
{
    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7); // align to 8 bytes

    uint64_t &allocated = on_device ? allocated_device : allocated_host,
        &size = on_device ? size_device : size_host;
    void *&current_free = on_device ? current_free_device : current_free_host;

    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Memory manager allocating " << bytes << " bytes. "
        << size << " bytes total, " << allocated << " bytes allocated." << std::endl;
    #endif

    if (allocated + bytes > size)
    {
        std::cerr << "Memory manager out of memory on "
            << (on_device ? "device" : "host") << ": requested " << bytes << " bytes, "
            << (size - allocated) << " bytes available." << std::endl;
        throw std::bad_alloc();
    }

    T *ptr = reinterpret_cast<T *>(current_free);

    current_free = static_cast<void *>(static_cast<uint8_t *>(current_free) + bytes);
    allocated += bytes;

    return ptr;
}

template <typename T>
T *memory_manager::alloc_zero(uint64_t count)
{
    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7); // align to 8 bytes

    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Memory manager allocating " << bytes << " bytes zeroed on device." << std::endl;
    #endif

    if (allocated_zero_device + bytes > size_zero_device)
    {
        std::cerr << "Memory manager out of zeroed memory on device: requested " << bytes << " bytes, "
            << (size_zero_device - allocated_zero_device) << " bytes available." << std::endl;
        throw std::bad_alloc();
    }

    T *ptr = reinterpret_cast<T *>(current_free_zero_device);

    current_free_zero_device = static_cast<void *>(static_cast<uint8_t *>(current_free_zero_device) + bytes);
    allocated_zero_device += bytes;

    return ptr;
}

sycl::event memory_manager::reset()
{
    // #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "reset "
        << (allocated_device >> 20) << "/" << (size_device >> 20) << " MB on device + "
        << (allocated_zero_device >> 20) << "/" << (size_zero_device >> 20) << " MB zero on device and "
        << (allocated_host >> 20) << "/" << (size_host >> 20) << " MB on host"
        << std::endl;
    // #endif

    auto e1 = queue.memset(memory_region_zero_device, 0, size_zero_device);

    current_free_device = memory_region_device;
    current_free_zero_device = memory_region_zero_device;
    current_free_host = memory_region_host;

    allocated_device = 0;
    allocated_zero_device = 0;
    allocated_host = 0;

    return e1;
}