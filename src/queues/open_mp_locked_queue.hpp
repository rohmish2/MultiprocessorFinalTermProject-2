#ifndef LOCKED_OPENMP_QUEUE_HPP
#define LOCKED_OPENMP_QUEUE_HPP

#include <queue>
#include <omp.h>
#include "queue_base.hpp"

class OMLockedQueue : public QueueBase {
private:
    std::queue<int> queue;
    mutable omp_lock_t lock;

public:
    OMLockedQueue() {
        omp_init_lock(&lock);
    }

    ~OMLockedQueue() {
        omp_destroy_lock(&lock);
    }

    bool enqueue(int item) override {
        omp_set_lock(&lock);
        queue.push(item);
        omp_unset_lock(&lock);
        return true;
    }

    bool dequeue(int& item) override {
        omp_set_lock(&lock);
        if (queue.empty()) {
            omp_unset_lock(&lock);
            return false;
        }
        item = queue.front();
        queue.pop();
        omp_unset_lock(&lock);
        return true;
    }

    bool empty() const override {
        omp_set_lock(&lock);
        bool is_empty = queue.empty();
        omp_unset_lock(&lock);
        return is_empty;
    }

    size_t size() const  {
        omp_set_lock(&lock);
        size_t size = queue.size();
        omp_unset_lock(&lock);
        return size;
    }
};

#endif