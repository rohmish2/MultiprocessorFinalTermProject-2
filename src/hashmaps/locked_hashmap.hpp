#ifndef LOCKED_HASHMAP_HPP
#define LOCKED_HASHMAP_HPP

#include <unordered_map>
#include <mutex>
#include "hashmap_base.hpp"

class LockedHashMap : public HashMapBase {
private:
    std::unordered_map<int, int> map;
    mutable std::mutex mtx;

public:
    bool insert(int key, int value) override {
        std::lock_guard<std::mutex> lock(mtx);
        return map.emplace(key, value).second;
    }

    bool remove(int key) override {
        std::lock_guard<std::mutex> lock(mtx);
        return map.erase(key) > 0;
    }

    bool find(int key, int value) override {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = map.find(key);
        if (it != map.end()) {
            return it->second == value;
        }
        return false;
    }

    size_t size() const override {
        std::lock_guard<std::mutex> lock(mtx);
        return map.size();
    }

    void clear() override {
        std::lock_guard<std::mutex> lock(mtx);
        map.clear();
    }
};

#endif // LOCKED_HASHMAP_HPP