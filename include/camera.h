#ifndef CAMERA_H
#define CAMERA_H
#include "color.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"

// 根据光线r的y分量计算得到一个颜色
__device__ vec3 ray_color(const ray& r, hittable_list** world) {
  hit_record rec;

  if ((*world)->hit(r, interval(0, infinity), rec)) {
    return (rec.normal + vec3(1, 1, 1)) / 2.0f;
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
  }
}

// 尝试使用 GPU端 的 ray_color
// A __global__ function or function template cannot be a member function
// 因此只能将 render<<<>>> 函数提出来
__global__ void render(vec3* fb, hittable_list** world, int image_width,
                       int image_height, vec3 origin, vec3 lower_left_corner,
                       vec3 horizontal, vec3 vertical, int max_thread) {
  int id = getThreadId();
  if (id >= max_thread) return;

  // 计算 pixel 的 index
  int pixel_index = id;
  int u = pixel_index % image_width;
  int v = pixel_index / image_width;
  vec3 lo = (lower_left_corner + u * horizontal + v * vertical);
  ray r(origin, lo - origin);
  fb[pixel_index] = ray_color(r, world);
}
// 相机类
class camera {
 public:
  double aspect_ratio = 16.0 / 9.0;  // 图片的横纵比例
  int image_width = 400;             // 图片的像素宽度
  int samples_per_pixel = 100;       // 每个像素的采样光线数
  int max_depth = 10;                // 光线在最大深度(反射次数)

  double vfov = 90;  // Vertical view angle, field of view 视场角

  point3 lookfrom = point3(0, 0, -1);  // 相机(眼睛的所在位置)
  point3 lookat = point3(0, 0, 0);     // 相机盯着的点
  vec3 vup =
      vec3(0, 1, 0);  // 相机(视角)的"上"向量，没必要完全跟 lookfrom-lookat
                      // 向量正交，因为之后会使用 向量的叉乘操作使得两个向量正交

  double defocus_angle = 0;  // 景深效果中，一个像素中发出的光线角度
  double focus_dist = 10;  // 理想的焦距
  /* Public Camera Parameters Here */

  // void render(const hittable_list& world) {
  //   // 先初始化相机的各个参数
  //   initialize();
  // Render
  // 渲染图像
  // for (int j = 0; j < image_height; ++j) {
  //   std::clog << "\rScanlines remaining: " << (image_height - j) << ' '
  //             << std::flush;
  //   for (int i = 0; i < image_width; ++i) {
  //     color pixel_color;
  //     for (int sample = 0; sample < samples_per_pixel; sample++) {
  //       // 采样得到一条 视点->图像(i,j)像素的光线，在像素(i,j)中随机采样
  //       auto r = get_ray(i, j);
  //       // 在这里实现一个 计算光线 r 颜色的光线跟踪程序
  //       pixel_color += ray_color(r, max_depth, world);

  //     }
  //   }
  // }

  // 输出渲染结果
  //   write_color(fb, image_width, image_height, std::cout, samples_per_pixel);

  //   std::clog << "\rDone.                 \n";
  // }

  __host__ void renderEntrance(hittable_list** world) {
    initialize();
    int tx = 8;
    int ty = 8;
    int nx = image_width;
    int ny = image_height;

    dim3 grid_size(nx / tx + 1, ny / ty + 1);
    dim3 block_size(tx, ty);

    render<<<grid_size, block_size>>>(fb, world, nx, ny, center, pixel00_loc,
                                      pixel_delta_u, pixel_delta_v,
                                      image_width * image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // 输出渲染结果
    samples_per_pixel = 1;
    write_color(fb, image_width, image_height, std::cout, samples_per_pixel);
  }

  ~camera() {
    if (fb != nullptr) {
      checkCudaErrors(cudaFree(fb));
    }
  }

 private:
  int image_height;    // 图片的像素高度
  point3 center;       // 相机中心点
  point3 pixel00_loc;  // 左上角像素的坐标
  vec3 pixel_delta_u;  // u方向 单位像素的 坐标delta
  vec3 pixel_delta_v;  // v方向 单位像素的 坐标delta
  // u:相机的右/左方向
  // v:相机的上方向
  // w:相机的后方向
  vec3 u, v, w;  // 相机局部坐标系的三个基本坐标轴

  vec3 defocus_disk_u;  // 散焦的 u 方向半径
  vec3 defocus_disk_v;  // 散焦的 v 方向半径

  /* CUDA */
  // 图像内存大小, framebuffer size
  size_t fb_size = 0;
  vec3* fb = nullptr;

  /* Private Camera Variables Here */
  void initialize() {
    // Calculate the image height, and ensure that it's at least 1.
    // image 是视窗(viewport) 映射到屏幕(位图)上的二维矩阵
    image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera
    // 相机参数
    center = lookfrom;

    // auto focal_length =
    //     (lookfrom - lookat).length();  // 焦距 表示 视点 到 被盯着的点 的距离

    auto theta = degrees_to_radians(vfov);  // 视场角 弧度
    auto h = tan(theta / 2.0);

    // Viewport widths less than one are ok since they are real valued.
    // viewport 视窗, 是一个虚拟的窗口，使用世界坐标系下的坐标尺寸表示
    // 这里的 viewport_height 和 viewport_width 必须用实际的 image
    // 像素值计算，不能用 aspect_ratio 计算，因为 image_width/image_height
    // 不一定真的等于 aspect_ratio
    auto viewport_height = 2 * h * focus_dist;  // 视窗 世界坐标系下的值
    auto viewport_width =
        viewport_height * (static_cast<double>(image_width) / image_height);

    // 计算相机的局部坐标 u,v,w
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    // Calculate the vectors across the horizontal and down the vertical
    // viewport edges. 视窗u向量
    auto viewport_u = viewport_width * u;
    // 视窗v向量
    // 这里必须乘以 -v，因为
    // 图片的坐标系中是以左上角为(0,0)，height方向的坐标往下为正
    auto viewport_v = viewport_height * -v;

    // Calculate the horizontal and vertical delta vectors from pixel to
    // pixel. 计算每个像素代表的坐标delta pixel_delta_u 等于 u
    // 方向一个像素表示的坐标 delta 值
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    // 计算 视窗左上角的世界坐标
    // viewport_upper_left = 视点 往z负方向移动到达 视窗中心
    // 再往左移动视窗宽度的一半 再往上移动视窗高度的一半
    auto viewport_upper_left =
        center - focus_dist * w - viewport_u / 2 - viewport_v / 2;

    // 视窗图像像素坐标为(0,0)的世界坐标值
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // 散焦半径
    auto defocus_radius =
        focus_dist * tan(degrees_to_radians(defocus_angle / 2.0));
    // 单位散焦半径 u 方向
    defocus_disk_u = defocus_radius * u;
    // 单位散焦半径 v 方向
    defocus_disk_v = defocus_radius * v;
    // CUDA 相关
    int num_pixels = image_width * image_height;
    fb_size = num_pixels * sizeof(vec3);
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
  }

  // 计算 光线 r 在场景 world 中的颜色
  // color ray_color(const ray& r, int depth, const hittable_list& world) {
  //   if (depth <= 0) {
  //     return color(0, 0, 0);
  //   }
  //   hit_record rec;
  //   // 如果集中物体，就反射物体的颜色*0.5
  //   // 忽略[0,0.001) 范围内的交点，以避免浮点运算的误差
  //   if (world.hit(r, interval(0.001, infinity), rec)) {
  //     // 计算反射结果
  //     // 反射光线
  //     ray scattered;
  //     // 光线的散射
  //     color attenuation;
  //     // 入射光, 交点, 衰减, 散射
  //     if (rec.mat->scatter(r, rec, attenuation, scattered)) {
  //       return attenuation * ray_color(scattered, depth - 1, world);
  //     } else {
  //       // 有一定可能 反射光 与 交点法向 反向，此时
  //       // rec.mat->scatter()函数返回false
  //       return color(0, 0, 0);
  //     }
  //   }
  //   // 如果没有击中物体，就假设击中天空，天空的颜色是一个无意义的随机值
  //   vec3 unit_direction = unit_vector(r.direction());
  //   auto a = 0.5 * (unit_direction.y() + 1.0);
  //   return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
  // }

  // 采样 视点->像素 (i,j) 的一条光线
  ray get_ray(int i, int j) const {
    auto pixel_center = pixel00_loc + i * pixel_delta_u + j * pixel_delta_v;
    auto pixel_sample = pixel_center + pixel_sample_square();

    auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
    auto ray_direction = pixel_sample - ray_origin;
    return ray(ray_origin, ray_direction);
  }
  // 在一个 square 像素中采样得到一个点
  vec3 pixel_sample_square() const {
    auto px = -0.5 + random_double();
    auto py = -0.5 + random_double();
    return px * pixel_delta_u + py * pixel_delta_v;
  }
  // 在一个 disk 中采样一个点
  point3 defocus_disk_sample() const {
    auto p = random_in_unit_disk();
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
  }
};

#endif