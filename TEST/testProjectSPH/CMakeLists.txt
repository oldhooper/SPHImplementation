cmake_minimum_required(VERSION 3.0) #показывает какой минимальной версией можно собрать проект

project(testProjectSPH) #дает имя проекту

set(NANOGUI_BUILD_EXAMPLE ON)
set(NANOGUI_BUILD_PYTHON OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../nanogui nanogui)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../nanogui/include ${NANOGUI_EXTRA_INCS})

add_executable(nanosandbox ${CMAKE_CURRENT_SOURCE_DIR}/testProjectSPH/testProjectSPH.cpp)
target_link_libraries(nanosandbox nanogui)

#дает понять что будет собираться исполняеый файл(exe)
#первый аргумент - это название собираемоего проекта, остальные исходные файлы *.h *.cpp
#все аргументы записываются через пробел
add_executable(testProjectSPH testProjectSPH/testProjectSPH.cpp) 