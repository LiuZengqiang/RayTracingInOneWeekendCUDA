#ifndef HITTABLE_H
#define HITTABLE_H
#include "ray.h"
#include "rtweekend.h"
class material;

class hit_record {
 public:
  point3 p;
  vec3 normal;  // 这个normal总是和ray方向相反，因此不一定是物体表面的外法向
  material* mat = nullptr;
  float t;
  // 用于标记这个交点是在物体的外表面(true) 还是内表面(false)
  bool front_face;
  // 用于设置这个交点的 front_face 和 normal
  __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? (outward_normal) : (-outward_normal);
  }
};
// 物体对象
// 所有物体都要继承这个类
class hittable {
 public:
  // =default 指示编译器生成默认的构造函数
  hittable() = default;

  __device__ virtual bool hit(const ray& r, interval ray_t,
                              hit_record& rec) const = 0;
  __device__ virtual size_t self_size() const = 0;
};

#endif