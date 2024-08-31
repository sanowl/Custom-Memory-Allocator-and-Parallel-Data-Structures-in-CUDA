#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include "memory_allocator.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ void* d_memory_pool = nullptr;
__device__ size_t d_pool_size = 0;
__device__ BlockHeader* d_free_list = nullptr;
__device__ int d_allocation_count = 0;

__global__ void initializeMemoryPool(void* pool, size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_memory_pool = pool;
        d_pool_size = size;
        d_free_list = (BlockHeader*)pool;
        d_free_list->size = size - sizeof(BlockHeader);
        d_free_list->next = nullptr;
        d_allocation_count = 0;
    }
}

__device__ void* allocateMemory(size_t size) {
    size = (size + sizeof(size_t) - 1) & ~(sizeof(size_t) - 1);  // Align to 8 bytes

    BlockHeader* prev = nullptr;
    BlockHeader* curr = d_free_list;

    while (curr != nullptr) {
        if (curr->size >= size) {
            if (curr->size > size + sizeof(BlockHeader)) {
                BlockHeader* new_block = (BlockHeader*)((char*)curr + sizeof(BlockHeader) + size);
                new_block->size = curr->size - size - sizeof(BlockHeader);
                new_block->next = curr->next;
                curr->size = size;
                curr->next = new_block;
            }

            if (prev == nullptr) {
                d_free_list = curr->next;
            } else {
                prev->next = curr->next;
            }

            atomicAdd(&d_allocation_count, 1);
            return (void*)((char*)curr + sizeof(BlockHeader));
        }

        prev = curr;
        curr = curr->next;
    }

    return nullptr;  // Out of memory
}

__device__ void freeMemory(void* ptr) {
    if (ptr == nullptr) return;

    BlockHeader* block = (BlockHeader*)((char*)ptr - sizeof(BlockHeader));
    
    // Coalesce with previous block if adjacent
    BlockHeader* prev = nullptr;
    BlockHeader* curr = d_free_list;
    while (curr != nullptr && curr < block) {
        prev = curr;
        curr = curr->next;
    }

    if (prev != nullptr && (char*)prev + prev->size + sizeof(BlockHeader) == (char*)block) {
        prev->size += block->size + sizeof(BlockHeader);
        block = prev;
    } else {
        block->next = curr;
        if (prev == nullptr) {
            d_free_list = block;
        } else {
            prev->next = block;
        }
    }

    // Coalesce with next block if adjacent
    if ((char*)block + block->size + sizeof(BlockHeader) == (char*)block->next) {
        block->size += block->next->size + sizeof(BlockHeader);
        block->next = block->next->next;
    }

    atomicSub(&d_allocation_count, 1);
}

__global__ void kernelAllocateMemory(void** result, size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = allocateMemory(size);
    }
}

__global__ void kernelFreeMemory(void* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        freeMemory(ptr);
    }
}

MemoryAllocator::MemoryAllocator(size_t pool_size) : pool_size(pool_size) {
    cudaMalloc(&d_pool, pool_size);
    initializeMemoryPool<<<1, 1>>>(d_pool, pool_size);
    cudaDeviceSynchronize();
}

MemoryAllocator::~MemoryAllocator() {
    cudaFree(d_pool);
}

void* MemoryAllocator::allocate(size_t size) {
    void* result;
    void** d_result;
    cudaMalloc(&d_result, sizeof(void*));
    kernelAllocateMemory<<<1, 1>>>(d_result, size);
    cudaMemcpy(&result, d_result, sizeof(void*), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}

void MemoryAllocator::free(void* ptr) {
    kernelFreeMemory<<<1, 1>>>(ptr);
    cudaDeviceSynchronize();
}

int MemoryAllocator::getAllocationCount() const {
    int count;
    cudaMemcpyFromSymbol(&count, d_allocation_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
    return count;
}