#ifndef QUEUE_BASE_CUDA_HPP
#define QUEUE_BASE_CUDA_HPP

#include <cstddef>

class QueueBase {
public:
    __host__ __device__ virtual ~QueueBase() = default;
    __host__ __device__ virtual bool enqueue(int item) = 0;
    __host__ __device__ virtual bool dequeue(int& item) = 0;
    __host__ __device__ virtual bool empty() const = 0;
    __host__ __device__ virtual size_t size() const = 0;
};

#endif 
