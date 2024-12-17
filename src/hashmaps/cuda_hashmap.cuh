#ifndef CUDA_HASHMAP_CUH
#define CUDA_HASHMAP_CUH

#include <cuda_runtime.h>
#include <cstdint>

__device__ __host__ inline static uint32_t integerHash(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

class CUDAHashMap {
public:
    CUDAHashMap(uint32_t arraySize = 1U << 22) : m_arraySize(arraySize) {
   
        cudaMalloc(&d_keys, arraySize * sizeof(uint32_t));
        cudaMalloc(&d_values, arraySize * sizeof(uint32_t));
        
        cudaMemset(d_keys, 0, arraySize * sizeof(uint32_t));
        cudaMemset(d_values, 0, arraySize * sizeof(uint32_t));
    }

    ~CUDAHashMap() {
        cudaFree(d_keys);
        cudaFree(d_values);
    }

    uint32_t getArraySize() const { return m_arraySize; }
    uint32_t* getKeys() const { return d_keys; }
    uint32_t* getValues() const { return d_values; }

private:
    uint32_t* d_keys;
    uint32_t* d_values;
    uint32_t m_arraySize;
};

__global__ void insertKernel(uint32_t* keys, uint32_t* values, 
                            const uint32_t* input_keys, const uint32_t* input_values,
                            uint32_t num_items, uint32_t array_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    uint32_t key = input_keys[tid];
    uint32_t value = input_values[tid];
    
    if (key == 0 || value == 0) return;

    uint32_t start = integerHash(key);
    uint32_t i = start;

    while (true) {
        i &= (array_size - 1);
        uint32_t existing_key = atomicCAS(&keys[i], 0, key);
        
        if (existing_key == 0) {
            values[i] = value;
            break;
        } else if (existing_key == key) {
            values[i] = value;
            break;
        }
        
        i++;
    }
}

__global__ void findKernel(const uint32_t* keys, const uint32_t* values,
                          const uint32_t* search_keys, bool* results,
                          uint32_t num_items, uint32_t array_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) return;

    uint32_t key = search_keys[tid];
    if (key == 0) {
        results[tid] = false;
        return;
    }

    uint32_t start = integerHash(key);
    uint32_t i = start;

    while (true) {
        i &= (array_size - 1);
        uint32_t probedKey = keys[i];
        
        if (probedKey == key) {
            results[tid] = true;
            return;
        }
        if (probedKey == 0) {
            results[tid] = false;
            return;
        }
        i++;
    }
}

#endif 