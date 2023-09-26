#include <float.h>

#include <iostream>
#include <curand_kernel.h>
#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "interval.h"
#include "material.h"
#include "ray.h"
#include "rtweekend.h"
#include "sphere.h"
#include "vec3.h"

__global__ void create_world(hittable_list** world, curandState* rand_state) {
  int id = getThreadId();
  if (id > 0) return;
  curand_init(1984, id, 0, rand_state);

  *world = new hittable_list();

  lambertian* ground_material = new lambertian(color(0.5, 0.5, 0.5));

  (*world)->add(new sphere(point3(0, -1000, 0), 1000, ground_material));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = random_double(rand_state);

      point3 center(a + 0.9f * random_double(rand_state), 0.2f,
                    b + 0.9f * random_double(rand_state));
      // 根据随机数 choose_mat 的取值大小随机生成 diffuse, metal, glass 材质的球
      if ((center - point3(4, 0.2f, 0)).length() > 0.9f) {
        material* sphere_material;

        if (choose_mat < 0.8f) {
          // diffuse, 漫反射材质, 反射光线在交点法线方向存在散射(漫反射)和衰减
          auto albedo = random(rand_state) * random(rand_state);

          sphere_material = new lambertian(albedo);

          (*world)->add(new sphere(center, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // metal, 金属材质, 反射光线在理想反射光线方向存在散射(漫反射)和衰减
          auto albedo = random(0.5f, 1.0f, rand_state);
          auto fuzz = random_double(0.0f, 0.5f, rand_state);

          sphere_material = new metal(albedo, fuzz);
          (*world)->add(new sphere(center, 0.2f, sphere_material));
        } else {
          // glass, 玻璃材质, 反射光线存在折射,反射和衰减
          sphere_material = new dielectric(1.5);
          (*world)->add(new sphere(center, 0.2f, sphere_material));
        }
      }
    }
  }

  dielectric* material1 = new dielectric(1.5f);
  (*world)->add(new sphere(point3(0, 1, 0), 1.0f, material1));

  lambertian* material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
  (*world)->add(new sphere(point3(-4, 1, 0), 1.0, material2));

  metal* material3 = new metal(color(0.7, 0.6, 0.5), 0.0f);
  (*world)->add(new sphere(point3(4, 1, 0), 1.0, material3));
}

__global__ void free_world(hittable_list** world) {
  (*world)->clear();
  delete *world;
}
int main() {
  hittable_list** d_world = nullptr;

  curandState* d_rand_state;
  checkCudaErrors(
      cudaMallocManaged((void**)&d_rand_state, sizeof(curandState)));

  cudaMalloc((void**)&d_world, 1 * sizeof(hittable_list*));
  create_world<<<1, 1>>>(d_world, d_rand_state);
  checkCudaErrors(cudaDeviceSynchronize());

  /* 设置相机和输出图像的属性 */
  camera cam;
  cam.aspect_ratio = 16.0 / 9.0;  // 图像的长宽比
  cam.image_width = 800;          // 图像的宽(像素数)
  cam.samples_per_pixel = 100;    // 每个像素的采样光线数
  cam.max_depth = 50;             // 光线的最大深度

  cam.vfov = 20;                    // 视场角
  cam.lookfrom = point3(13, 2, 3);  // 相机位置
  cam.lookat = point3(0, 0, 0);     // 相机观察的点
  cam.vup = vec3(0, 1, 0);          // 相机上方向向量

  cam.defocus_angle = 0.6;  // 模拟实际相机的散射角度(以实现景深效果)
  cam.focus_dist = 10.0;  // 模拟实际相机的理想焦距(以实现景深效果)

  cam.render(d_world);
  checkCudaErrors(cudaDeviceSynchronize());

  free_world<<<1, 1>>>(d_world);
  checkCudaErrors(cudaDeviceSynchronize());

  cudaFree(d_world);
  cudaFree(d_rand_state);
  return 0;
}