#ifndef HASHMAP_BASE_HPP
#define HASHMAP_BASE_HPP

class HashMapBase {
public:
    virtual bool insert(int key, int value) = 0;
    virtual bool remove(int key) = 0;
    virtual bool find(int key, int value) = 0;  
    virtual size_t size() const = 0;
    virtual void clear() = 0;
    virtual ~HashMapBase() = default;
};

#endif