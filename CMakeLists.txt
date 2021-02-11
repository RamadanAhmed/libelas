# project
cmake_minimum_required (VERSION 3.14)
project (libelas)

find_package(OpenCV CONFIG REQUIRED)
# directories
add_library(elas)

target_sources(elas 
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptor.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/src/elas.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/src/filter.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/src/triangle.cpp)

target_include_directories(elas
                           PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_options(elas
                       PRIVATE
                       $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>)

# build demo program
add_executable(elas_demo src/main.cpp)
target_link_libraries(elas_demo PRIVATE elas)

# build demo program
add_executable(elas_opencv example/main.cpp)
target_link_libraries(elas_opencv PRIVATE elas opencv_core opencv_imgcodecs opencv_highgui)
