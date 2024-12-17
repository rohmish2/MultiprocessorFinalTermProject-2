#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <omp.h>
#include <map>

#include "hashmap_base.hpp"
#include "locked_hashmap.hpp"
#include "omp_locked_hashmap.hpp"
#include "libcuckoo_hashmap_wrapper.hpp"
#include "partially_lock_free_hashmap.hpp"

struct BenchmarkResults {
    double insert_time_ms;
    double find_time_ms;
    double remove_time_ms;
    size_t final_size;
    long long insert_retries;
    long long find_retries;
    long long remove_retries;
    double avg_retries;
};

struct TestConfig {
    const char* name;
    int num_threads;
    int num_operations;
};

class HashMapBenchmark {
private:
    
  
   

    static BenchmarkResults run_single_test(HashMapBase* hashmap, int num_threads, int num_operations) {
        BenchmarkResults results{};
        std::atomic<long long> insert_retries{0}, find_retries{0}, remove_retries{0};

        int total_ops = num_threads * num_operations;

        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < total_ops; ++i) {
            int retries = 0;
            while (!hashmap->insert(i,i)) {
                retries++;
                if (retries > 100000) {
                    std::cerr << "Stuck on insert key = " << i << "\n";
                    break;
                }
                std::this_thread::yield();
            }
            insert_retries += retries;
        }
        #pragma omp barrier
        auto end = std::chrono::high_resolution_clock::now();
        results.insert_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < total_ops; ++i) {
            int retries = 0;
            std::string value;
            while (!hashmap->find(i, i)) {
                retries++;
                if (retries > 100000) {
                    std::cerr << "Stuck on find key = " << i << "\n";
                    break;
                }
                std::this_thread::yield();
            }
            find_retries += retries;
        }
        #pragma omp barrier
        end = std::chrono::high_resolution_clock::now();
        results.find_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < total_ops; ++i) {
            int retries = 0;
            while (!hashmap->remove(i)) {
                retries++;
                if (retries > 100000) {
                    std::cerr << "Stuck on remove key = " << i << "\n";
                    break;
                }
                std::this_thread::yield();
            }
            remove_retries += retries;
        }
        #pragma omp barrier
        end = std::chrono::high_resolution_clock::now();
        results.remove_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        results.final_size = hashmap->size();
        results.insert_retries = insert_retries;
        results.find_retries = find_retries;
        results.remove_retries = remove_retries;
        results.avg_retries = (insert_retries + find_retries + remove_retries) / 
                              (3.0 * total_ops);

        return results;
    }

    static void print_results(const std::vector<BenchmarkResults>& results) {
        const char* implementations[] = {
            "Locked Hash Map",
            "OMP Locked Hash Map",
            
            "libcucu Hash Map"
        };

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::cout << "\nResults for " << implementations[i] << ":\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Insertion time: " << r.insert_time_ms << " ms\n";
            std::cout << "Search time: " << r.find_time_ms << " ms\n";
            std::cout << "Deletion time: " << r.remove_time_ms << " ms\n";
            std::cout << "Final size: " << r.final_size << "\n";
            std::cout << "Insert retries: " << r.insert_retries << "\n";
            std::cout << "Find retries: " << r.find_retries << "\n";
            std::cout << "Remove retries: " << r.remove_retries << "\n";
            std::cout << "Average retries per operation: " << r.avg_retries << "\n";
        }
    }
     static BenchmarkResults measure_baseline() {
        BenchmarkResults results{};
        std::map<int, int> test_map;
        
        auto start_insert = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 400000; ++i) {
            test_map[i] = i;
        }
        auto end_insert = std::chrono::high_resolution_clock::now();
        
        auto start_find = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 400000; ++i) {
            volatile auto it = test_map.find(i);
        }
        auto end_find = std::chrono::high_resolution_clock::now();
        
        auto start_remove = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 400000; ++i) {
            test_map.erase(i);
        }
        auto end_remove = std::chrono::high_resolution_clock::now();
        
        results.insert_time_ms = std::chrono::duration<double, std::milli>(end_insert - start_insert).count();
        results.find_time_ms = std::chrono::duration<double, std::milli>(end_find - start_find).count();
        results.remove_time_ms = std::chrono::duration<double, std::milli>(end_remove - start_remove).count();
        results.final_size = 0;  // Should be 0 after all removals
        results.insert_retries = 0;
        results.find_retries = 0;
        results.remove_retries = 0;
        results.avg_retries = 0;
        
        return results;
    }
    
     static void print_summary(const TestConfig configs[],
                            size_t config_count,
                            const std::vector<std::vector<BenchmarkResults>>& all_results) {
        std::cout << "\n====================================================\n";
        std::cout << "              Overall Performance Summary              \n";
        std::cout << "====================================================\n\n";
        std::cout << std::fixed << std::setprecision(2);

        const char* implementations[] = {
            "Locked",
            "OMP Locked",
            "Lib Cucu Hashmap"
        };

        BenchmarkResults baseline = measure_baseline();
        double total_time_base = baseline.insert_time_ms + baseline.find_time_ms + baseline.remove_time_ms;
        
        std::cout << "Baseline (Single Thread) Performance:\n";
        std::cout << "  Insert Time: " << baseline.insert_time_ms << " ms\n";
        std::cout << "  Find Time: " << baseline.find_time_ms << " ms\n";
        std::cout << "  Remove Time: " << baseline.remove_time_ms << " ms\n";
        std::cout << "  Total Time: " << total_time_base << " ms\n\n";

       
        std::cout << std::setw(20) << "Implementation";
        for (size_t i = 0; i < config_count; ++i) {
            std::cout << std::setw(35) << configs[i].name;
        }
        std::cout << "\n";
        
        std::cout << std::string(20 + config_count * 35, '-') << "\n";

        for (size_t impl = 0; impl < 3; ++impl) {
            std::cout << std::setw(20) << implementations[impl];
            
            for (size_t config = 0; config < all_results.size(); ++config) {
                const auto& result = all_results[config][impl];
                double total_time = result.insert_time_ms + result.find_time_ms + result.remove_time_ms;
                double speedup = total_time_base / total_time;
                
                std::cout << std::setw(20) << total_time 
                         << " ms (speedup: " << std::setw(6) << speedup << "x)  ";
            }
            std::cout << "\n";
        }
        
        std::cout << "\nNote: Speedup > 1 indicates improvement over baseline\n";
        std::cout << "====================================================\n";
    }

public:
    static void run_benchmark_suite() {
        std::cout << "\nHash Map Performance Benchmark\n";
        std::cout << "Running on " << omp_get_max_threads() << " OpenMP threads\n";

        const TestConfig configs[] = {
            {"Low Contention", 4, 100000},
            {"Medium Contention", 8, 50000},
            {"High Contention", 16, 25000},
            {"Very High Contention", 32,12500},
            {"Extremely High Contention", 64, 6250}
        };

        const size_t config_count = sizeof(configs) / sizeof(configs[0]);
        std::vector<std::vector<BenchmarkResults>> all_results;

        for (size_t i = 0; i < config_count; ++i) {
            const auto& config = configs[i];
            std::cout << "\n=== " << config.name << " ===\n";
            std::cout << "Threads: " << config.num_threads
                      << ", Operations per thread: " << config.num_operations << "\n";

            std::vector<BenchmarkResults> config_results;

            {
                LockedHashMap locked_map;
                config_results.push_back(run_single_test(&locked_map, 
                                                         config.num_threads, config.num_operations));
            }
            {
                OMP_LockedHashMap omp_locked_map;
                config_results.push_back(run_single_test(&omp_locked_map, 
                                                         config.num_threads, config.num_operations));
            }
           
             {
                LibCuckooHashMapWrapper libcuckoo;
                config_results.push_back(run_single_test(&libcuckoo, 
                                                         config.num_threads, config.num_operations));
            }
            all_results.push_back(config_results);
            print_results(config_results);
            std::cout << "----------------------------------------\n";
        }

        print_summary(configs, config_count, all_results);
    }
};

int main() {
    HashMapBenchmark::run_benchmark_suite();
    return 0;
}
