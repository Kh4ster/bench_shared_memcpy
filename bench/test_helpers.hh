#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

template <typename FUNC>
void check_buffer(cuda_tools::host_shared_ptr<int> buffer, FUNC func);

#include "test_helpers.hxx"