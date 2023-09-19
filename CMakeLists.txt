cmake_minimum_required(VERSION 3.10.0)
project(RayTracingInOneWeekend VERSION 0.1.0 LANGUAGES CXX CUDA)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(RayTracingInOneWeekend main.cu)


set_target_properties(RayTracingInOneWeekend PROPERTIES CUDA_ARCHITECTURES "61")

# set_target_properties(
#         RayTracingInOneWeekend
#         PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON)