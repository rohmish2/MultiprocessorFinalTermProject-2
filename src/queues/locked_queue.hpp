#ifndef LOCKED_QUEUE_HPP
#define LOCKED_QUEUE_HPP

#include <queue>
#include <mutex>
#include "queue_base.hpp"

class LockedQueue : public QueueBase {
private:
    std::queue<int> queue;
    mutable std::mutex mutex;

public:
    bool enqueue(int item) override {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(item);
        return true;
    }

    bool dequeue(int& item) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) return false;
        item = queue.front();
        queue.pop();
        return true;
    }

    bool empty() const override {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    size_t size() const  {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};
#endif