#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

#include "../operations/memory_manager.hpp"

/**
 * Globally Stable Asynchronous copy_if.
 * * Algorithm:
 * 1. Identify valid elements and calc local offsets (Up-sweep).
 * 2. Scan the per-group totals to get global group offsets (Spine scan).
 * 3. Scatter elements to final position (Down-sweep).
 */
template <typename T, typename Predicate>
sycl::event copy_if(
    sycl::queue &q,
    const T *input,
    T *output,
    size_t *global_count, // Output: Total elements copied. needs to be on device
    size_t n,
    Predicate pred,
    memory_manager &allocator,
    const std::vector<sycl::event> &deps = {})
{

    size_t local_size = 256;
    size_t num_groups = (n + local_size - 1) / local_size;

    // We need temporary memory for:
    // 1. local_indices: The offset of an item within its own workgroup (size N)
    // 2. group_sums: The total number of valid items in each workgroup (size num_groups)
    size_t *temp_local_indices = allocator.alloc<size_t>(n, true);
    size_t *temp_group_sums = allocator.alloc<size_t>(num_groups, true);

    // --- KERNEL 1: Local Scan & Reduction ---
    auto e1 = q.submit([&](sycl::handler &h)
        {
            h.depends_on(deps);

            // Create local accessor for the group scan
            sycl::local_accessor<size_t, 1> scratch(sycl::range<1>(local_size), h);

            h.parallel_for(sycl::nd_range<1>(num_groups * local_size, local_size),
                [=](sycl::nd_item<1> item)
                {
                    size_t global_id = item.get_global_id(0);
                    size_t local_id = item.get_local_id(0);

                    // 1. Check predicate
                    bool keep = (global_id < n) ? pred(input[global_id]) : false;
                    size_t val = keep ? 1 : 0;

                    // 2. Exclusive Scan within the work-group
                    // This gives us the index relative to the start of the group
                    size_t local_idx = sycl::exclusive_scan_over_group(item.get_group(), val, sycl::plus<size_t>());

                    // Store local index for the final scatter pass
                    if (global_id < n)
                    {
                        temp_local_indices[global_id] = local_idx;
                    }

                    // 3. Last thread in group stores the total count for this group
                    // We use the result of the scan + the last thread's value
                    size_t group_total = sycl::reduce_over_group(item.get_group(), val, sycl::plus<size_t>());

                    if (local_id == 0)
                    {
                        temp_group_sums[item.get_group(0)] = group_total;
                    }
                });
        });

    // --- KERNEL 2: Spine Scan (Group Offsets) ---
    // We scan the 'temp_group_sums' array so that group I knows where group I-1 ended.
    // Note: For massive arrays, this single-group approach might need recursion, 
    // but a single work-group can easily scan thousands of group-sums (covering millions of input items).
    auto e2 = q.submit([&](sycl::handler &h)
        {
            h.depends_on(e1);

            // We run this as a single work-group to avoid recursive dependency logic for this snippet.
            // We load chunks of the group_sums, scan them, and carry over the offset.
            h.single_task([=]()
                {
                    size_t running_total = 0;
                    for (size_t i = 0; i < num_groups; ++i)
                    {
                        size_t val = temp_group_sums[i];
                        temp_group_sums[i] = running_total; // Exclusive scan write
                        running_total += val;
                    }
                    // Identify total count
                    if (global_count) *global_count = running_total;
                });
        });

    // --- KERNEL 3: Scatter (Down-sweep) ---
    auto e3 = q.submit([&](sycl::handler &h)
        {
            h.depends_on(e2);

            h.parallel_for(sycl::nd_range<1>(num_groups * local_size, local_size),
                [=](sycl::nd_item<1> item)
                {
                    size_t global_id = item.get_global_id(0);

                    if (global_id < n)
                    {
                        bool keep = pred(input[global_id]);
                        if (keep)
                        {
                            // Global Index = Group Start Offset + Local Offset
                            size_t group_start = temp_group_sums[item.get_group(0)];
                            size_t local_offset = temp_local_indices[global_id];

                            output[group_start + local_offset] = input[global_id];
                        }
                    }
                });
        });
    return e3;
}