#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/device_buffer.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_profiler_api.h>

using namespace cooperative_groups;

// Dummy compute just for the sake of having one
template <typename T>
__device__
static void compute(thread_group g, T shared[])
{
    shared[g.thread_rank()] += shared[g.thread_rank() + g.size()];
}

template <typename T, int block_width>
__global__
static void basic(cuda_tools::device_buffer<T> result,
                  cuda_tools::device_buffer<T> global1,
                  cuda_tools::device_buffer<T> global2)
{
    __shared__ T shared[block_width * 2];
    auto group = cooperative_groups::this_thread_block();

    const int tx = threadIdx.x;
    const int gx = tx + blockIdx.x * blockDim.x;
    if (gx >= result.size_) return;
 
    shared[group.thread_rank()               ] = global1[gx];
    shared[group.size() + group.thread_rank()] = global2[gx];
 
    group.sync(); // Wait for all copies to complete
 
    compute(group, shared);

    result[gx] = shared[group.thread_rank()];
}

template <typename T, int block_width>
__global__
static void cooperative_basic(cuda_tools::device_buffer<T> result,
                              cuda_tools::device_buffer<T> global1,
                              cuda_tools::device_buffer<T> global2,
                              const int subset_count)
{
    __shared__ T shared[block_width * 2];
    auto group = cooperative_groups::this_thread_block();

    const int tx = threadIdx.x;
    const int gx = tx + blockIdx.x * blockDim.x;
    if (gx * subset_count >= result.size_) return;
    const int grid_size = blockDim.x * gridDim.x;
 
    for (int subset = 0; subset < subset_count; ++subset)
    {
        shared[group.thread_rank()               ] = global1[subset * grid_size + gx];
        shared[group.size() + group.thread_rank()] = global2[subset * grid_size + gx];
 
        group.sync(); // Wait for all copies to complete
 
        compute(group, shared);

        result[gx + subset * grid_size] = shared[group.thread_rank()];

        group.sync();
    }
}

template <typename T, int block_width>
__global__
static void cooperative_async(cuda_tools::device_buffer<T> result,
                              cuda_tools::device_buffer<T> global1,
                              cuda_tools::device_buffer<T> global2,
                              const int subset_count)
{
    __shared__ T shared[block_width * 2];
    auto group = cooperative_groups::this_thread_block();

    const int tx = threadIdx.x;
    const int gx = tx + blockIdx.x * blockDim.x;
    if (gx * subset_count >= result.size_) return;
    const int grid_size = blockDim.x * gridDim.x;
 
    for (int subset = 0; subset < subset_count; ++subset)
    {
        cooperative_groups::memcpy_async(group, shared,
            &global1[subset * grid_size + gx], sizeof(T) * group.size());
        cooperative_groups::memcpy_async(group, shared + group.size(),
            &global2[subset * grid_size + gx], sizeof(T) * group.size());

        cooperative_groups::wait(group); // Wait for all copies to complete
 
        compute(group, shared);

        result[gx + subset * grid_size] = shared[group.thread_rank()];

        group.sync();
    }
}

void basic(cuda_tools::host_shared_ptr<int> _result,
           cuda_tools::host_shared_ptr<int> _global1,
           cuda_tools::host_shared_ptr<int> _global2)
{
    constexpr int TILE_WIDTH  = 64;
    constexpr int TILE_HEIGHT = 1;

    cudaProfilerStart();
    cudaFuncSetCacheConfig(basic<int, TILE_WIDTH>, cudaFuncCachePreferShared);
    
    cuda_tools::device_buffer<int> result(_result);
    cuda_tools::device_buffer<int> global1(_global1);
    cuda_tools::device_buffer<int> global2(_global2);

    const int gx             = (result.size_ + TILE_WIDTH - 1) / (TILE_WIDTH);
    const int gy             = 1;

    const dim3 block(TILE_WIDTH, TILE_HEIGHT);
    const dim3 grid(gx, gy);

    basic<int, TILE_WIDTH><<<grid, block>>>(result, global1, global2);
    kernel_check_error();

    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void cooperative_basic(cuda_tools::host_shared_ptr<int> _result,
                       cuda_tools::host_shared_ptr<int> _global1,
                       cuda_tools::host_shared_ptr<int> _global2)
{
    constexpr int TILE_WIDTH  = 64;
    constexpr int TILE_HEIGHT = 1;
    constexpr int SUBSET_COUNT = 4;

    cudaProfilerStart();
    cudaFuncSetCacheConfig(cooperative_basic<int, TILE_WIDTH>, cudaFuncCachePreferShared);

    cuda_tools::device_buffer<int> result(_result);
    cuda_tools::device_buffer<int> global1(_global1);
    cuda_tools::device_buffer<int> global2(_global2);

    const int gx             = (result.size_ + TILE_WIDTH - 1) / (TILE_WIDTH * SUBSET_COUNT);
    const int gy             = 1;
    
    const dim3 block(TILE_WIDTH, TILE_HEIGHT);
    const dim3 grid(gx, gy);

    cooperative_basic<int, TILE_WIDTH><<<grid, block>>>(result, global1, global2, SUBSET_COUNT);
    kernel_check_error();

    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void cooperative_async(cuda_tools::host_shared_ptr<int> _result,
    cuda_tools::host_shared_ptr<int> _global1,
    cuda_tools::host_shared_ptr<int> _global2)
{
    constexpr int TILE_WIDTH  = 64;
    constexpr int TILE_HEIGHT = 1;
    constexpr int SUBSET_COUNT = 1;

    cudaProfilerStart();
    cudaFuncSetCacheConfig(cooperative_async<int, TILE_WIDTH>, cudaFuncCachePreferShared);

    cuda_tools::device_buffer<int> result(_result);
    cuda_tools::device_buffer<int> global1(_global1);
    cuda_tools::device_buffer<int> global2(_global2);

    const int gx             = (result.size_ + TILE_WIDTH - 1) / (TILE_WIDTH * SUBSET_COUNT);
    const int gy             = 1;

    const dim3 block(TILE_WIDTH, TILE_HEIGHT);
    const dim3 grid(gx, gy);

    cooperative_async<int, TILE_WIDTH><<<grid, block>>>(result, global1, global2, SUBSET_COUNT);
    kernel_check_error();

    cudaDeviceSynchronize();
    cudaProfilerStop();
}