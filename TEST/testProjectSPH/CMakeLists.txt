cmake_minimum_required(VERSION 3.0) #показывает какой минимальной версией можно собрать проект

project(testProjectSPH) #дает имя проекту

#дает понять что будет собираться исполняеый файл(exe)
#первый аргумент - это название собираемоего проекта, остальные исходные файлы *.h *.cpp
#все аргументы записываются через пробел
add_executable(testProjectSPH testProjectSPH/testProjectSPH.cpp) 