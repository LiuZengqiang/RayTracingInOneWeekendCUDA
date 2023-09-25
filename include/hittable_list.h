// 多个hittable的封装类
// 使用到了 智能指针 等特性
// 在 CUDA 中不能使用智能指针或者STL容器, 只能使用 朴素的指针
#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>

#include "hittable.h"

class hittable_list : public hittable {
 public:
  // 物体列表
  // std::vector<shared_ptr<hittable>> objects;

  hittable** d_objects = nullptr;
  int d_objects_size = 0;

  // __host__ __device__ hittable_list() {}
  __host__ __device__ hittable_list(hittable** objects, int object_size) {
    d_objects = objects;
    d_objects_size = object_size;
  }

  // hitable_list(hitable** l, int n) {
  //   objects = l;
  //   objects_size = n;
  // }

  // 添加一个 物体
  // hittable_list(shared_ptr<hittable> object) { add(object); }

  // void clear() { objects.clear(); }

  // TODO: 完成该函数
  // void add(shared_ptr<hittable> object) { objects.push_back(object); }

  // __host__ __device__ void add(hittable* object) {}

  // 计算 光线 r 与物体列表第一个合法相交点
  // TODO: 使用 interval

  __host__ __device__ bool hit(const ray& r, interval ray_t,
                               hit_record& rec) const override {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    for (int i = 0; i < d_objects_size; i++) {
      if (d_objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }
    return hit_anything;
  }
};

#endif