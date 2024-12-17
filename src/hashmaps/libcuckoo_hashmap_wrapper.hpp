#ifndef LIBCUCKOO_HASHMAP_HPP
#define LIBCUCKOO_HASHMAP_HPP

#include "hashmap_base.hpp"
#include "libcuckoo/libcuckoo/cuckoohash_map.hh"
#include <string>

class LibCuckooHashMapWrapper : public HashMapBase {
private:
    libcuckoo::cuckoohash_map<int, int> map; 
public:
    bool insert(int key, int value) override {
        return map.insert(key, value);
    }

    bool remove(int key) override {
        return map.erase(key);
    }

    bool find(int key, int value) override {
        return map.find(key, value);
    }

    size_t size() const override {
        return map.size();
    }

    void clear() override {
        map.clear();
    }
};
#endif
