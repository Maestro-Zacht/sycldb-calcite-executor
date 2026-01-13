#pragma once

class KernelDefinition
{
private:
    int col_len;
public:
    KernelDefinition(int col_len) : col_len(col_len) {}
    int get_col_len() const { return col_len; }
};


uint64_t count_true_flags(
    const bool *flags,
    int len,
    sycl::queue &queue,
    const std::vector<sycl::event> &dependencies = {})
{
    uint64_t *count = sycl::malloc_shared<uint64_t>(1, queue);

    queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                sycl::range<1>(len),
                sycl::reduction(
                    count,
                    sycl::plus<>(),
                    sycl::property::reduction::initialize_to_identity()
                ),
                [=](sycl::id<1> idx, auto &sum)
                {
                    sum.combine(flags[idx[0]]);
                }
            );
        }
    ).wait();
    uint64_t result = *count;
    sycl::free(count, queue);
    return result;
}

sycl::event count_true_flags(
    const bool *flags,
    int len,
    sycl::queue &queue,
    memory_manager &allocator,
    uint64_t *result,
    const std::vector<sycl::event> &dependencies = {})
{
    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(dependencies);
            cgh.parallel_for(
                sycl::range<1>(len),
                sycl::reduction(result, sycl::plus<>()),
                [=](sycl::id<1> idx, auto &sum)
                {
                    // sycl::atomic_ref<
                    //     uint64_t,
                    //     sycl::memory_order::relaxed,
                    //     sycl::memory_scope::device,
                    //     sycl::access::address_space::global_space
                    // > count_atomic(*count);
                    // count_atomic.fetch_add(flags[idx[0]]);
                    sum.combine(flags[idx[0]]);
                }
            );
        }
    );
}