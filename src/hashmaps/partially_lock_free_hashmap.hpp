#include <atomic>
#include <cassert>
#include <cstdint>
#include "hashmap_base.hpp"

inline static uint32_t integerHash(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

class HashTable1 : public HashMapBase {
public:
    HashTable1() : HashTable1(1U << 22) {}

    HashTable1(uint32_t arraySize) {
        assert((arraySize & (arraySize - 1)) == 0); 
        m_arraySize = arraySize;
        m_keys = new std::atomic<uint32_t>[m_arraySize];
        m_values = new std::atomic<uint32_t>[m_arraySize];
        clear();
    }

    ~HashTable1() override {
        delete[] m_keys;
        delete[] m_values;
    }

    bool insert(int key, int value) override {
        if (key == 0 || value == 0) {
            return false;
        }

        uint32_t ukey = (uint32_t)key;
        uint32_t uvalue = (uint32_t)value;
        uint32_t start = integerHash(ukey);
        for (uint32_t i = start;; i++) {
            i &= (m_arraySize - 1);
            uint32_t probedKey = m_keys[i].load(std::memory_order_relaxed);

            if (probedKey == 0) {

                uint32_t expected = 0;
                if (m_keys[i].compare_exchange_strong(expected, ukey, std::memory_order_relaxed)) {
                    m_values[i].store(uvalue, std::memory_order_relaxed);
                    return true;
                } else {
                    if (m_keys[i].load(std::memory_order_relaxed) == ukey) {
                        m_values[i].store(uvalue, std::memory_order_relaxed);
                        return true;
                    }
                }
            } else if (probedKey == ukey) {
                m_values[i].store(uvalue, std::memory_order_relaxed);
                return true;
            }
        }
    }

    bool find(int key, int value) override {
        if (key == 0) return false;

        uint32_t ukey = (uint32_t)key;
        uint32_t start = integerHash(ukey);

        for (uint32_t i = start;; i++) {
            i &= (m_arraySize - 1);
            uint32_t probedKey = m_keys[i].load(std::memory_order_relaxed);

            if (probedKey == ukey) {
                return true;
            }
            if (probedKey == 0) {
                return false;
            }
        }
    }

    bool remove(int key) override {
        return true;
    }

    size_t size() const override {
        size_t itemCount = 0;
        for (uint32_t i = 0; i < m_arraySize; i++) {
            uint32_t k = m_keys[i].load(std::memory_order_relaxed);
            uint32_t v = m_values[i].load(std::memory_order_relaxed);
            if (k != 0 && v != 0) {
                itemCount++;
            }
        }
        return itemCount;
    }

    void clear() override {
        for (uint32_t i = 0; i < m_arraySize; i++) {
            m_keys[i].store(0, std::memory_order_relaxed);
            m_values[i].store(0, std::memory_order_relaxed);
        }
    }

private:
    std::atomic<uint32_t>* m_keys;
    std::atomic<uint32_t>* m_values;
    uint32_t m_arraySize;
};
