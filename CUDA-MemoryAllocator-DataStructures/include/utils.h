#pragma once

#include "memory_allocator.h"

void debugPrintMemoryUtilization(MemoryAllocator& allocator);

template<typename Func, typename... Args>
float measureExecutionTime(Func func, Args... args);

template<typename T>
voi