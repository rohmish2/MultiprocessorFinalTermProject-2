
#include <iostream>
#include <omp.h>
#include <chrono>
#include <atomic>
#include "queue_base.hpp"
#include "locked_queue.hpp"
#include "spsc_queue.hpp"
#include "open_mp_locked_queue.hpp"
#include "ms_queue_lib.hpp"
#include "mpmc_queue_wrapper.hpp"

#include <queue>
#include <iomanip>  

void test_stl_queue_baseline(int num_items) {
    std::queue<int> queue;
    
    double start_time = omp_get_wtime();
    
    for (int i = 0; i < num_items; ++i) {
        queue.push(i);
    }
    double enqueue_end_time = omp_get_wtime();
    
    for (int i = 0; i < num_items; ++i) {
        queue.pop();
    }
    double dequeue_end_time = omp_get_wtime();

    double enqueue_duration = (enqueue_end_time - start_time) * 1000;
    double dequeue_duration = (dequeue_end_time - enqueue_end_time) * 1000;
    double total_duration = (dequeue_end_time - start_time) * 1000;
    
    std::cout << "\n=== STL Queue Baseline (Single-Threaded) ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Items processed: " << num_items << "\n";
    std::cout << "Enqueue time: " << enqueue_duration << " ms\n";
    std::cout << "Dequeue time: " << dequeue_duration << " ms\n";
    std::cout << "Total time: " << total_duration << " ms\n";
    std::cout << "Enqueue throughput: " << num_items / (enqueue_duration/1000) << " items/second\n";
    std::cout << "Dequeue throughput: " << num_items / (dequeue_duration/1000) << " items/second\n";
    std::cout << "Overall throughput: " << num_items / (total_duration/1000) << " items/second\n";
    std::cout << "Queue empty?: " << (queue.empty() ? "yes" : "no") << "\n";
    std::cout << "\n----------------------------------------\n";
}

void test_queue(QueueBase* queue, const char* queue_name, int num_producers, 
                int num_consumers, int items_per_producer) {
    const int total_items = num_producers * items_per_producer;
    std::atomic<long long> enqueued_count{0};
    std::atomic<long long> dequeued_count{0};

    double start_time, enqueue_end_time, dequeue_end_time;

    if (num_producers == 1 && num_consumers == 1) {
        start_time = omp_get_wtime();
        
        for (int i = 0; i < total_items; ++i) {
            while (!queue->enqueue(i)) { }
        }
        enqueue_end_time = omp_get_wtime();

        int item;
        for (int i = 0; i < total_items; ++i) {
            while (!queue->dequeue(item)) { }
        }
        dequeue_end_time = omp_get_wtime();
    }
    else {
        start_time = omp_get_wtime();

        #pragma omp parallel num_threads(num_producers)
        {
            for (int i = 0; i < items_per_producer; ++i) {
                int item = omp_get_thread_num() * items_per_producer + i;
                while (!queue->enqueue(item)) { }
                enqueued_count.fetch_add(1);
            }
        }
        enqueue_end_time = omp_get_wtime();

        #pragma omp barrier

        #pragma omp parallel num_threads(num_consumers)
        {
            int item;
            while (dequeued_count.load() < total_items) {
                if (queue->dequeue(item)) {
                    dequeued_count.fetch_add(1);
                }
            }
        }
        dequeue_end_time = omp_get_wtime();
    }

    double enqueue_duration = (enqueue_end_time - start_time) * 1000;
    double dequeue_duration = (dequeue_end_time - enqueue_end_time) * 1000;
    double total_duration = (dequeue_end_time - start_time) * 1000;
    
    std::cout << "\nResults for " << queue_name << ":\n";
    std::cout << "Total Threads: " << (num_producers + num_consumers) << "\n";
    std::cout << "Enqueue time: " << enqueue_duration << " ms\n";
    std::cout << "Dequeue time: " << dequeue_duration << " ms\n";
    std::cout << "Total time: " << total_duration << " ms\n";
    std::cout << "Enqueue throughput: " << total_items / (enqueue_duration/1000) << " items/second\n";
    std::cout << "Dequeue throughput: " << total_items / (dequeue_duration/1000) << " items/second\n";
    std::cout << "Overall throughput: " << total_items / (total_duration/1000) << " items/second\n";
    std::cout << "Equque empty?: " << queue->empty() << "\n";
std::cout << "Total items enqueued: " << (num_producers == 1 ? (long long)total_items : enqueued_count.load()) << "\n";
std::cout << "Total items dequeued: " << (num_producers == 1 ? (long long)total_items : dequeued_count.load()) << "\n";

    if (num_producers > 1 && enqueued_count != dequeued_count) {
        std::cout << "WARNING: Item mismatch! Enqueued: " << enqueued_count
                  << ", Dequeued: " << dequeued_count << "\n";
    } else {
        std::cout << "All items were successfully processed.\n";
    }
}

int main() {
    // Initialize OpenMP settings
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    
    std::cout << "Queue Performance Benchmark" << std::endl;
    std::cout << "Running on " << omp_get_max_threads() << " OpenMP threads" << std::endl;
    std::cout << "Nested parallelism: " << (omp_get_nested() ? "enabled" : "disabled") << "\n";

    struct TestConfig {
        const char* name;
        int producers;
        int consumers;
        int items_per_producer;
    };

    const int BASE_ITEMS = 1000000;
    
    TestConfig configs[] = {
        {"Single Producer-Consumer", 1, 1, BASE_ITEMS},              
        {"Low Contention (4x4)", 4, 4, BASE_ITEMS/4},               
        {"Medium Contention (8x8)", 8, 8, BASE_ITEMS/8},            
        {"High Contention (16x16)", 16, 16, BASE_ITEMS/16},         
        {"More High Contention (32x32)", 32, 32, BASE_ITEMS/32},
    };
         test_stl_queue_baseline(BASE_ITEMS);

    // Run tests for all configurations
    for (const auto& config : configs) {
        std::cout << "\n=== " << config.name << " Test ===" << std::endl;
        std::cout << "Producers: " << config.producers 
                  << ", Consumers: " << config.consumers 
                  << ", Items per producer: " << config.items_per_producer << std::endl;

        // Test all queue implementations
        {
            LockedQueue locked_queue;
            test_queue(&locked_queue, "Locked Queue", 
                      config.producers, config.consumers, config.items_per_producer);
        }
        {
            OMLockedQueue om_locked_queue;
            test_queue(&om_locked_queue, "OM Locked Queue", 
                      config.producers, config.consumers, config.items_per_producer);
        }
        {
            MSQueueLib ms_queue_lib;
            test_queue(&ms_queue_lib, "MS Lib Boost Queue", 
                      config.producers, config.consumers, config.items_per_producer);
        }
        {
            MPMCQueueWrapper mpmc_queue_wrapper;
            test_queue(&mpmc_queue_wrapper, "rigtorp Queue", 
                      config.producers, config.consumers, config.items_per_producer);
        }
       
        if (config.producers == 1 && config.consumers == 1) {
            SPSCQueue spsc_queue;
            test_queue(&spsc_queue, "SPSC Queue", 
                      config.producers, config.consumers, config.items_per_producer);
        }
        
        std::cout << "\n----------------------------------------\n";
    }

    return 0;
}