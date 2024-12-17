#ifndef QUEUE_BASE_HPP
#define QUEUE_BASE_HPP

#include <cstddef>

class QueueBase {
public:
    virtual ~QueueBase() = default;
    virtual bool enqueue(int item) = 0;
    virtual bool dequeue(int& item) = 0;
    virtual bool empty() const = 0;
};

#endif