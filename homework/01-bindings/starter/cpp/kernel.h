#pragma once
#include <cstddef>

// The binding layer validates arrays and owns the buffers.
// For size == 0, do not dereference any pointer.
void mac_kernel(const double* a, const double* b, const double* c,
                double* out, std::size_t size);
