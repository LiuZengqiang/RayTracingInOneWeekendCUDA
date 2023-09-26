#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <memory>
#include <vector>

#include "hittable.h"

struct hittable_node {
  hittable* object = nullptr;
  hittable_node* next = nullptr;
};

class hittable_list : public hittable {
 public:
  // 物体列表(链表)
  hittable_node* d_first_object_node = nullptr;
  hittable_node* d_last_object_node = nullptr;

  int d_objects_size = 0;

  __host__ __device__ hittable_list() {}

  __device__ void clear() {
    hittable_node* cur_object_node = d_first_object_node;

    while (cur_object_node != nullptr) {
      delete cur_object_node->object;
      hittable_node* temp_object_node = cur_object_node->next;
      delete cur_object_node;
      cur_object_node = temp_object_node;
    }
  }

  __device__ void add(hittable* object) {
    if (object == nullptr) {
      return;
    }
    if (d_objects_size == 0) {
      d_first_object_node = new hittable_node();
      d_first_object_node->object = object;
      d_last_object_node = d_first_object_node;

      d_objects_size++;
    } else {
      hittable_node* temp_object_node = new hittable_node();
      temp_object_node->object = object;
      d_last_object_node->next = temp_object_node;
      d_last_object_node = temp_object_node;
      d_objects_size++;
    }
  }

  // __host__ __device__ void add(hittable* object) {}

  // 计算 光线 r 与物体列表第一个合法相交点
  __device__ bool hit(const ray& r, interval ray_t,
                      hit_record& rec) const override {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;
    hittable_node* cur_object_node = d_first_object_node;

    while (cur_object_node != nullptr) {
      hittable* object = cur_object_node->object;
      if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
      cur_object_node = cur_object_node->next;
    }

    // for (int i = 0; i < d_objects_size; i++) {
    //   if (d_objects[i]->hit(r, interval(ray_t.min, closest_so_far),
    //   temp_rec)) {
    //     hit_anything = true;
    //     closest_so_far = temp_rec.t;
    //     rec = temp_rec;
    //   }
    // }
    return hit_anything;
  }

  __device__ size_t self_size() const override { return sizeof(*this); };
};

#endif