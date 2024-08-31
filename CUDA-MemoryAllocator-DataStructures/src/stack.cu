#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "stack.h"

template<typename T>
__global__ void initializeStack(Stack<T>* stack, size_t capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        stack->data = nullptr;
        stack->size = 0;
        stack->capacity = capacity;
    }
}

template<typename T>
Stack<T>::Stack(MemoryAllocator& allocator, size_t initial_capacity)
    : allocator(allocator), d_stack(nullptr) {
    cudaMalloc(&d_stack, sizeof(Stack<T>));
    initializeStack<<<1, 1>>>(d_stack, initial_capacity);
    cudaDeviceSynchronize();

    T* data = static_cast<T*>(allocator.allocate(sizeof(T) * initial_capacity));
    cudaMemcpy(&(d_stack->data), &data, sizeof(T*), cudaMemcpyHostToDevice);
}

template<typename T>
Stack<T>::~Stack() {
    if (d_stack) {
        T* h_data;
        cudaMemcpy(&h_data, &(d_stack->data), sizeof(T*), cudaMemcpyDeviceToHost);
        allocator.free(h_data);
        cudaFree(d_stack);
    }
}

template<typename T>
__global__ void pushElement(Stack<T>* stack, const T* element) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (stack->size < stack->capacity) {
            stack->data[stack->size++] = *element;
        }
    }
}

template<typename T>
void Stack<T>::push(const T& value) {
    T* d_value;
    cudaMalloc(&d_value, sizeof(T));
    cudaMemcpy(d_value, &value, sizeof(T), cudaMemcpyHostToDevice);
    pushElement<<<1, 1>>>(d_stack, d_value);
    cudaDeviceSynchronize();
    cudaFree(d_value);
}

template<typename T>
__global__ void popElement(Stack<T>* stack, T* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (stack->size > 0) {
            *result = stack->data[--stack->size];
        }
    }
}

template<typename T>
T Stack<T>::pop() {
    T result;
    T* d_result;
    cudaMalloc(&d_result, sizeof(T));
    popElement<<<1, 1>>>(d_stack, d_result);
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_result);
    return result;
}

template<typename T>
__global__ void peekElement(const Stack<T>* stack, T* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (stack->size > 0) {
            *result = stack->data[stack->size - 1];
        }
    }
}

template<typename T>
T Stack<T>::peek() const {
    T result;
    T* d_result;
    cudaMalloc(&d_result, sizeof(T));
    peekElement<<<1, 1>>>(d_stack, d_result);
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_result);
    return result;
}

template<typename T>
__global__ void getStackSize(const Stack<T>* stack, size_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = stack->size;
    }
}

template<typename T>
size_t Stack<T>::size() const {
    size_t result;
    size_t* d_result;
    cudaMalloc(&d_result, sizeof(size_t));
    getStackSize<<<1, 1>>>(d_stack, d_result);
    cudaMemcpy(&result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_result);
    return result;
}

// Explicit instantiation for common types
template class Stack<int>;
template class Stack<float>;
template class Stack<double>;