#ifndef MS_QUEUE_LIB_HPP
#define MS_QUEUE_LIB_HPP

#include <boost/lockfree/queue.hpp>
#include "queue_base.hpp"

class MSQueueLib : public QueueBase {
private:
    mutable boost::lockfree::queue<int> queue;
    


public:
    explicit MSQueueLib(size_t initial_capacity = 64) 
        : queue(initial_capacity) {}

    ~MSQueueLib() override = default;

    bool enqueue(int item) override {
       return queue.push(item);
    }

    bool dequeue(int& item) override {
        return  queue.pop(item);
       
    }

   
    bool empty() const  {
        return queue.empty();
    }



};

#endif 