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

__global__ void create_world(hittable_list** world) {
  int id = getThreadId();
  if (id > 0) return;
  *world = new hittable_list();
  // list[0] = new sphere(point3(0, 0, -1), 0.5, color(0, 1, 0));
  // material* lamber = new lambertian(color(0.5, 0.5, 0.5));
  // material* metal_material = new metal(color(0.8, 0.8, 0.8), 0.3);
  // material* metal_dielectric = new dielectric(1.2);

  lambertian* material_ground = new lambertian(color(0.8, 0.8, 0.0));
  lambertian* material_center = new lambertian(color(0.7, 0.3, 0.3));
  metal* material_left = new metal(color(0.8, 0.8, 0.8), 0.3);
  metal* material_right = new metal(color(0.8, 0.6, 0.2), 1.0);

  (*world)->add(new sphere(point3(0, -100.5, -1), 100, material_ground));
  (*world)->add(new sphere(point3(0.0, 0.0, -1.0), 0.5, material_center));
  (*world)->add(new sphere(point3(-1.0, 0.0, -1.0), 0.5, material_left));
  (*world)->add(new sphere(point3(1.0, 0.0, -1.0), 0.5, material_right));
}

__global__ void free_world(hittable_list** world) {
  (*world)->clear();
  delete *world;
}
int main() {

  hittable_list** d_world = nullptr;
  cudaMalloc((void**)&d_world, 1 * sizeof(hittable_list*));
  create_world<<<1, 1>>>(d_world);

  camera cam;
  cam.aspect_ratio = 16.0 / 9.0;
  cam.image_width = 800;
  cam.samples_per_pixel = 1000;
  cam.max_depth = 40;

  cam.vfov = 90;
  cam.lookfrom = point3(0, 0, 0);
  cam.lookat = point3(0, 0, -1);
  cam.vup = vec3(0, 1, 0);

  cam.defocus_angle = 0.6;
  cam.focus_dist = 1.0;
  cam.render(d_world);
  checkCudaErrors(cudaDeviceSynchronize());

  free_world<<<1, 1>>>(d_world);

  // checkCudaErrors(cudaGetLastError());
  // checkCudaErrors(cudaDeviceSynchronize());

  // cudaFree(d_list);
  cudaFree(d_world);
  return 0;
}