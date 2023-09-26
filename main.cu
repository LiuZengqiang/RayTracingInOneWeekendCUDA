// TODO:使用CUDA并行运算
// TODO:修改world的构建方法，尽可能的与CPU版本的代码相同

#include <float.h>

#include <iostream>

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "interval.h"
#include "material.h"
#include "ray.h"
#include "rtweekend.h"
#include "sphere.h"
#include "vec3.h"

__global__ void create_world(hittable** list, hittable_list** world) {
  int id = getThreadId();
  if (id > 0) return;
  list[0] = new sphere(point3(0, 0, -1), 0.5, color(0, 1, 0));
  // material* lamber = new lambertian(color(0.5, 0.5, 0.5));
  // list[0] = new sphere(point3(0, 3, 0), 3, lamber);
  list[1] = new sphere(point3(0, -100.5, -1), 100, color(1, 0, 0));
  *world = new hittable_list(list, 2);
}

__global__ void free_world(hittable** list, hittable_list** world) {
  delete (list[0]);
  delete (list[1]);
  delete *world;
}
int main() {
  // hittable_list world;
  // 定义 世界场景
  // 需要一个生成 world 的 kernel 函数

  // auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
  // world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

  // auto material1 = make_shared<dielectric>(1.5);
  // world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

  // auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
  // world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

  // auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
  // world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

  // 新建场景
  // world.to_device();
  // world.device();
  hittable** d_list = nullptr;

  cudaMalloc((void**)&d_list, 2 * sizeof(hittable*));

  hittable_list** d_world = nullptr;
  cudaMalloc((void**)&d_world, 1 * sizeof(hittable_list*));

  create_world<<<1, 1>>>(d_list, d_world);
  camera cam;
  cam.aspect_ratio = 16.0 / 9.0;
  cam.aspect_ratio = 1.0;
  cam.image_width = 400;
  cam.samples_per_pixel = 100;
  cam.max_depth = 2;

  cam.vfov = 90;
  cam.lookfrom = point3(0, 0, 0);
  cam.lookat = point3(0, 0, -1);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0.6;
  cam.focus_dist = 1.0;
  cam.renderEntrance(d_world);
  // cam.render(world);
  checkCudaErrors(cudaDeviceSynchronize());

  free_world<<<1, 1>>>(d_list, d_world);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  cudaFree(d_list);
  cudaFree(d_world);
  return 0;
}