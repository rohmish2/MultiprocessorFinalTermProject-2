#ifndef MPMC_QUEUE_WRAPPER_HPP
#define MPMC_QUEUE_WRAPPER_HPP

#include "queue_base.hpp"
#include "MPMCQueue/include/rigtorp/MPMCQueue.h"
#include <memory>  

class MPMCQueueWrapper : public QueueBase {
private:
    std::unique_ptr<rigtorp::MPMCQueue<int>> queue;

public:
    explicit MPMCQueueWrapper(size_t capacity)
        : queue(new rigtorp::MPMCQueue<int>(capacity)) {}

    MPMCQueueWrapper() 
        : queue(new rigtorp::MPMCQueue<int>(10000000)) {} 

    ~MPMCQueueWrapper() override = default;

    bool enqueue(int item) override {
        return queue->try_push(item);
    }

    bool dequeue(int &item) override {
        return queue->try_pop(item);
    }

    bool empty() const override {
        return queue->empty();
    }

    size_t size() const  {
        return static_cast<size_t>(queue->size());
    }
};

#endif 
