#pragma once

#include <sycl/sycl.hpp>

#define MEMORY_MANAGER_DEBUG_INFO 0

class memory_region
{
private:
    void *memory_ptr, *current_free;
    uint64_t size, allocated;
    sycl::queue &queue;
    bool zero;
public:
    memory_region(sycl::queue &queue, uint64_t size, bool on_device, bool zero);
    ~memory_region();

    template <typename T>
    bool can_alloc(uint64_t count) const;

    template <typename T>
    T *alloc(uint64_t count);

    sycl::event reset();
};

memory_region::memory_region(sycl::queue &queue, uint64_t size, bool on_device, bool zero)
    : size(size), queue(queue), zero(zero)
{
    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Allocating memory region of size " << size << " bytes on "
        << (on_device ? "device" : "host") << (zero ? " with zeroing." : ".") << std::endl;
    #endif

    if (on_device)
        memory_ptr = sycl::malloc_device<uint8_t>(size, queue);
    else
        memory_ptr = sycl::malloc_host<uint8_t>(size, queue);

    if (memory_ptr == nullptr)
    {
        std::cerr << "Memory region failed to allocate memory." << std::endl;
        throw std::bad_alloc();
    }

    allocated = 0;
    reset();
}

memory_region::~memory_region()
{
    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Freeing memory region of size "
        << (allocated >> 20) << "/" << (size >> 20) << " MB on "
        << (zero ? "zeroed " : "") << (queue.get_device().is_host() ? "host." : "device.")
        << std::endl;
    #endif

    sycl::free(memory_ptr, queue);

    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Memory region freed." << std::endl;
    #endif
}

template <typename T>
bool memory_region::can_alloc(uint64_t count) const
{
    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7); // align to 8 bytes
    return (allocated + bytes <= size);
}

template <typename T>
T *memory_region::alloc(uint64_t count)
{
    uint64_t bytes = count * sizeof(T);
    bytes = (bytes + 7) & (~7); // align to 8 bytes

    #if MEMORY_MANAGER_DEBUG_INFO
    std::cout << "Memory manager allocating " << bytes << " bytes. "
        << size << " bytes total, " << allocated << " bytes allocated." << std::endl;
    #endif

    if (allocated + bytes > size)
    {
        std::cerr << "Memory manager out of memory: requested " << bytes << " bytes, "
            << (size - allocated) << " bytes available." << std::endl;
        throw std::bad_alloc();
    }

    T *ptr = reinterpret_cast<T *>(current_free);

    current_free = static_cast<void *>(static_cast<uint8_t *>(current_free) + bytes);
    allocated += bytes;

    return ptr;
}

sycl::event memory_region::reset()
{
    sycl::event e;

    current_free = memory_ptr;
    allocated = 0;

    if (zero)
        e = queue.memset(memory_ptr, 0, size);

    return e;
}

class memory_manager
{
private:
    std::vector<memory_region> regions_device;
    std::vector<memory_region> regions_host;
    std::vector<memory_region> regions_zero_device;
public:
    memory_manager(sycl::queue &queue, uint64_t size, uint64_t max_region_size);

    template <typename T>
    T *alloc(uint64_t count, bool on_device);

    template <typename T>
    T *alloc_zero(uint64_t count);

    std::vector<sycl::event> reset();
};

memory_manager::memory_manager(sycl::queue &queue, uint64_t size, uint64_t max_region_size)
{
    uint64_t size_left = size >> 6,
        num_regions = (size_left + max_region_size - 1) / max_region_size;
    regions_zero_device.reserve(num_regions);
    while (size_left > 0)
    {
        uint64_t region_size = (size_left > max_region_size) ? max_region_size : size_left;

        regions_zero_device.emplace_back(queue, region_size, true, true);

        size_left -= region_size;
    }

    size_left = size;
    num_regions = (size_left + max_region_size - 1) / max_region_size;
    regions_device.reserve(num_regions);
    regions_host.reserve(num_regions);
    while (size_left > 0)
    {
        uint64_t region_size = (size_left > max_region_size) ? max_region_size : size_left;

        regions_device.emplace_back(queue, region_size, true, false);
        regions_host.emplace_back(queue, region_size, false, false);

        size_left -= region_size;
    }

    queue.wait();
}

template <typename T>
T *memory_manager::alloc(uint64_t count, bool on_device)
{
    std::vector<memory_region> &regions = on_device ? regions_device : regions_host;

    for (auto &region : regions)
    {
        if (region.can_alloc<T>(count))
        {
            return region.alloc<T>(count);
        }
    }

    std::cerr << "Memory manager out of memory regions on " << (on_device ? "device" : "host") << std::endl;
    throw std::bad_alloc();
}

template <typename T>
T *memory_manager::alloc_zero(uint64_t count)
{
    for (auto &region : regions_zero_device)
    {
        if (region.can_alloc<T>(count))
        {
            return region.alloc<T>(count);
        }
    }

    std::cerr << "Memory manager out of zeroed memory regions on device" << std::endl;
    throw std::bad_alloc();
}

std::vector<sycl::event> memory_manager::reset()
{
    std::vector<sycl::event> events;
    events.reserve(
        regions_device.size() +
        regions_host.size() +
        regions_zero_device.size()
    );

    for (auto &region : regions_device)
    {
        events.push_back(region.reset());
    }

    for (auto &region : regions_host)
    {
        events.push_back(region.reset());
    }

    for (auto &region : regions_zero_device)
    {
        events.push_back(region.reset());
    }

    return events;
}