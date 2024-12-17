#ifndef OMP_LOCKED_HASHMAP_HPP
#define OMP_LOCKED_HASHMAP_HPP

#include <unordered_map>
#include <omp.h>
#include "hashmap_base.hpp"

class OMP_LockedHashMap : public HashMapBase {
private:
    std::unordered_map<int, int> map;
    mutable omp_lock_t lock;

public:
    OMP_LockedHashMap() {
        omp_init_lock(&lock);
    }

    ~OMP_LockedHashMap() {
        omp_destroy_lock(&lock);
    }

    bool insert(int key, int value) override {
        omp_set_lock(&lock);
        bool result = map.emplace(key, value).second;
        omp_unset_lock(&lock);
        return result;
    }

    bool remove(int key) override {
        omp_set_lock(&lock);
        bool result = map.erase(key) > 0;
        omp_unset_lock(&lock);
        return result;
    }

    bool find(int key, int value) override {
        omp_set_lock(&lock);
        auto it = map.find(key);
        bool result = (it != map.end() && it->second == value);
        omp_unset_lock(&lock);
        return result;
    }

    size_t size() const override {
        omp_set_lock(&lock);
        size_t result = map.size();
        omp_unset_lock(&lock);
        return result;
    }

    void clear() override {
        omp_set_lock(&lock);
        map.clear();
        omp_unset_lock(&lock);
    }
};

#endif 