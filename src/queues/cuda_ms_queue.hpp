#ifndef CUDA_MS_QUEUE_HPP
#define CUDA_MS_QUEUE_HPP

#include <cuda_runtime.h>

class CUDAMSQueue;
__global__ void initQueueKernel(CUDAMSQueue* queue, void* pool);
__global__ void cleanupQueueKernel(CUDAMSQueue* queue);

class CUDAMSQueue {
private:
    static const int POOL_SIZE = 10240 * 1024;  

    struct Node {
        int data;
        Node* next;
        __device__ Node() : data(0), next(nullptr) {}
        __device__ explicit Node(int value) : data(value), next(nullptr) {}
    };

    Node* pool_nodes;     
    unsigned int pool_head; 
    Node* head;
    Node* tail;

    __device__ Node* allocateNode() {
        unsigned int index = atomicAdd(&pool_head, 1);
        if (index < POOL_SIZE) {
            return &pool_nodes[index];
        }
        return nullptr; 
    }

public:
 
    static CUDAMSQueue* createOnDevice() {
        
        CUDAMSQueue* device_queue;
        cudaMalloc(&device_queue, sizeof(CUDAMSQueue));

        Node* device_pool;
        cudaMalloc(&device_pool, POOL_SIZE * sizeof(Node));

        initQueueKernel<<<1, 1>>>(device_queue, device_pool);
        cudaDeviceSynchronize();

        return device_queue;
    }

    static void destroyOnDevice(CUDAMSQueue* queue) {
        if (queue) {
            cleanupQueueKernel<<<1, 1>>>(queue);
            cudaDeviceSynchronize();
            
            Node* device_pool;
            cudaMemcpy(&device_pool, &(queue->pool_nodes), sizeof(Node*), cudaMemcpyDeviceToHost);
            cudaFree(device_pool);
            cudaFree(queue);
        }
    }

    __device__ bool enqueue(int value) {
        Node* node = allocateNode();
        if (!node) {
            return false;  
        }
        new(node) Node(value);

        while (true) {
            Node* last = tail;
            Node* next = last->next;

            if (last == tail) {
                if (next == nullptr) {
                    if (atomicCAS(
                        reinterpret_cast<unsigned long long*>(&last->next),
                        reinterpret_cast<unsigned long long>(nullptr),
                        reinterpret_cast<unsigned long long>(node)
                    ) == reinterpret_cast<unsigned long long>(nullptr)) {
                        
                        atomicCAS(
                            reinterpret_cast<unsigned long long*>(&tail),
                            reinterpret_cast<unsigned long long>(last),
                            reinterpret_cast<unsigned long long>(node)
                        );
                        return true;
                    }
                } else {
                    atomicCAS(
                        reinterpret_cast<unsigned long long*>(&tail),
                        reinterpret_cast<unsigned long long>(last),
                        reinterpret_cast<unsigned long long>(next)
                    );
                }
            }
        }
    }

    __device__ bool dequeue(int& result) {
        while (true) {
            Node* first = head;
            Node* last = tail;
            Node* next = first->next;

            if (first == head) {
                if (first == last) {
                    if (next == nullptr) {
                        return false; 
                    }
                    atomicCAS(
                        reinterpret_cast<unsigned long long*>(&tail),
                        reinterpret_cast<unsigned long long>(last),
                        reinterpret_cast<unsigned long long>(next)
                    );
                } else {
                    result = next->data;
                    if (atomicCAS(
                        reinterpret_cast<unsigned long long*>(&head),
                        reinterpret_cast<unsigned long long>(first),
                        reinterpret_cast<unsigned long long>(next)
                    ) == reinterpret_cast<unsigned long long>(first)) {
                        return true;
                    }
                }
            }
        }
    }

    __device__ bool empty() const {
        return head == tail;
    }

    friend __global__ void initQueueKernel(CUDAMSQueue* queue, void* pool);
    friend __global__ void cleanupQueueKernel(CUDAMSQueue* queue);
};

__global__ void initQueueKernel(CUDAMSQueue* queue, void* pool) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        queue->pool_nodes = static_cast<CUDAMSQueue::Node*>(pool);
        queue->pool_head = 0;

        CUDAMSQueue::Node* dummy = queue->allocateNode();
        if (dummy) {
            new(dummy) CUDAMSQueue::Node();
            queue->head = dummy;
            queue->tail = dummy;
        }
    }
}

__global__ void cleanupQueueKernel(CUDAMSQueue* queue) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        queue->head = nullptr;
        queue->tail = nullptr;
        queue->pool_nodes = nullptr;
    }
}

#endif