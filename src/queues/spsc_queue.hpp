#ifndef SPSC_QUEUE_HPP
#define SPSC_QUEUE_HPP

#include "queue_base.hpp"
#include <atomic>
#include <vector>

class SPSCQueue : public QueueBase {
private:
    static const size_t QUEUE_SIZE = 10000000;
    std::vector<int> buffer;
    std::atomic<size_t> head;  
    std::atomic<size_t> tail;  

public:
    SPSCQueue() : 
        buffer(QUEUE_SIZE),
        head(0),
        tail(0) {}

    bool enqueue(int item) override {
        const size_t current_tail = tail.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) % QUEUE_SIZE;
        
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false;
        }

        buffer[current_tail] = item;
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    bool dequeue(int& item) override {
        const size_t current_head = head.load(std::memory_order_relaxed);
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false;
        }

        item = buffer[current_head];
        head.store((current_head + 1) % QUEUE_SIZE, std::memory_order_release);
        return true;
    }

    bool empty() const override {
        return head.load(std::memory_order_acquire) == 
               tail.load(std::memory_order_acquire);
    }
};

#endif 