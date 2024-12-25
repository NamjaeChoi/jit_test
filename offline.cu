#include "cuda_runtime.h"

#include <iostream>

__device__  float compute(float a, float x, float y) {
	return a * x + y;
}

void foo()
{
	std::cout << "finished" << std::endl;
}