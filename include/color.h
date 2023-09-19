#ifndef COLOR_H
#define COLOR_H

#include <iostream>

#include "rtweekend.h"
#include "vec3.h"

using color = vec3;
// gamma 矫正
inline double linear_to_gamma(double linear_component) {
  return sqrt(linear_component);
}

// samples_per_pixel 是一个像素的采样光线数
// 最终的像素颜色 等于 多根采样光线的平均
/**
 * @brief 输出结果
 *
 * @param fb 内存地址
 * @param nx image_width
 * @param ny image_height
 * @param out 输出目标流
 * @param samples_per_pixel
 */
void write_color(float* fb, int nx, int ny, std::ostream& out,
                 int samples_per_pixel) {
  out << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * 3 * nx + i * 3;
      float r = fb[pixel_index + 0];
      float g = fb[pixel_index + 1];
      float b = fb[pixel_index + 2];

      auto scale = 1.0 / samples_per_pixel;
      r *= scale;
      g *= scale;
      b *= scale;

      // gamma 矫正
      r = linear_to_gamma(r);
      g = linear_to_gamma(g);
      b = linear_to_gamma(b);
      static const interval intensity(0.0, 0.999);

      out << static_cast<int>(255.999 * intensity.clamp(r)) << ' '
          << static_cast<int>(255.999 * intensity.clamp(g)) << ' '
          << static_cast<int>(255.999 * intensity.clamp(b)) << '\n';
    }
  }
}

#endif
