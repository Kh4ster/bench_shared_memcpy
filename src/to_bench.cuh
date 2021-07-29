#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

void basic(cuda_tools::host_shared_ptr<int> result,
    cuda_tools::host_shared_ptr<int> global1,
    cuda_tools::host_shared_ptr<int> global2);

void cooperative_basic(cuda_tools::host_shared_ptr<int> result,
    cuda_tools::host_shared_ptr<int> global1,
    cuda_tools::host_shared_ptr<int> global2);

void cooperative_async(cuda_tools::host_shared_ptr<int> _result,
        cuda_tools::host_shared_ptr<int> _global1,
        cuda_tools::host_shared_ptr<int> _global2);