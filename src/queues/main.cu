#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "cuda_ms_queue.hpp"

#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    }

__global__ void producer_kernel(CUDAMSQueue* queue, int items_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = 0; i < items_per_thread; ++i) {
        int item = idx * items_per_thread + i;
        queue->enqueue(item);
       }
}

__global__ void consumer_kernel(CUDAMSQueue* queue, int items_per_thread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = 0; i < items_per_thread; ++i) {
        int backoff = 1;
        bool success = false;
        int item;
        success = queue->dequeue(item);
    }
}

void test_cuda_queue(const char* test_name, int num_blocks, int threads_per_block, int items_per_thread) {
    const int total_threads = num_blocks * threads_per_block;
    const int total_items = total_threads * items_per_thread;

    std::cout << "\n=== " << test_name << " ===\n";
    std::cout << "Blocks: " << num_blocks 
              << "\nThreads per block: " << threads_per_block
              << "\nItems per thread: " << items_per_thread
              << "\nTotal items: " << total_items << "\n";

    // Create queue
    CUDAMSQueue* device_queue = CUDAMSQueue::createOnDevice();
    if (!device_queue) {
        std::cerr << "Failed to create queue!\n";
        return;
    }

    // Allocate counters
    int *processed_count, *failed_enqueues, *failed_dequeues;
    CHECK_CUDA_ERROR(cudaMalloc(&processed_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&failed_enqueues, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&failed_dequeues, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemset(processed_count, 0, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(failed_enqueues, 0, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(failed_dequeues, 0, sizeof(int)));

    auto start = std::chrono::high_resolution_clock::now();

    producer_kernel<<<num_blocks, threads_per_block>>>(device_queue, items_per_thread);
    consumer_kernel<<<num_blocks, threads_per_block>>>(device_queue, items_per_thread);
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    int host_processed_count, host_failed_enqueues, host_failed_dequeues;


    double elapsed_time = std::chrono::duration<double>(end - start).count();
    
    std::cout << "\nResults:\n"
              << "Time elapsed: " << elapsed_time * 1000 << " ms\n"
              << "Throughput: " << (total_items / elapsed_time) << " items/second\n";

    CUDAMSQueue::destroyOnDevice(device_queue);
    CHECK_CUDA_ERROR(cudaFree(processed_count));
    CHECK_CUDA_ERROR(cudaFree(failed_enqueues));
    CHECK_CUDA_ERROR(cudaFree(failed_dequeues));
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name 
              << "\nCompute Capability: " << prop.major << "." << prop.minor 
              << "\nSMs: " << prop.multiProcessorCount << "\n";

    const int BLOCKS = 128;
    const int THREADS = 160;
    const int ITEMS_PER_THREAD = 28;

    test_cuda_queue("CUDA MS Queue Test", BLOCKS, THREADS, ITEMS_PER_THREAD);

    return 0;
}