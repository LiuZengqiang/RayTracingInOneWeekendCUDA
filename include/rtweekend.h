// 常用的常量和函数
// 需要用到的命名空间
#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <cuda_runtime.h>

// Usings
// 用到的命名空间
using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants
// 常用的常数
const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

// Utility Functions
// 常用的函数
// 角度转弧度
inline float degrees_to_radians(float degrees) { return degrees * pi / 180.0f; }
// 生成 [0,1) 随机数的函数
inline float random_double() {
  static std::uniform_real_distribution<float> distribution(0.0, 1.0);
  static std::mt19937 generator;
  return distribution(generator);
}
// 生成 [min, max) 范围内随机数的函数
inline float random_double(float min, float max) {
  // Returns a random real in [min,max).
  return min + (max - min) * random_double();
}

// Common Headers
// 这个interval.h 头文件必须放在这个位置，因为 interval 类需要 infinity 常量
#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif