// TODO:: 在GPU端double精度可能会很耗时, 需要转为float单精度
#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
 public:
  __host__ __device__ ray() {}

  __host__ __device__ ray(const point3& origin, const vec3& direction)
      : orig(origin), dir(direction) {}
  // 取 ray 的原点
  __host__ __device__ point3 origin() const { return orig; }
  // 取 ray 的方向
  __host__ __device__ vec3 direction() const { return dir; }
  // 取 ray 的终点
  __host__ __device__ point3 at(double t) const { return orig + t * dir; }

 private:
  point3 orig;
  vec3 dir;
};

#endif