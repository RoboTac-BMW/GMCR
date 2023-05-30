cmake_minimum_required(VERSION 3.16)
project(fgr-baseline VERSION 0.1.0)


find_package (Eigen3 3.3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(FastGlobalRegistrationLib STATIC app.cpp include/app.h)
target_link_libraries(FastGlobalRegistrationLib PRIVATE ${PCL_LIBRARIES} Eigen3::Eigen)

add_library(FGR_Utils STATIC fgr_utils.cpp include/fgr_utils.h)
target_link_libraries(FGR_Utils PRIVATE ${PCL_LIBRARIES} Eigen3::Eigen FastGlobalRegistrationLib)

add_executable(fgr_test fgr_test.cpp)
target_link_libraries(fgr_test PRIVATE ${PCL_LIBRARIES} Eigen3::Eigen FastGlobalRegistrationLib FGR_Utils)

add_executable(fgr_baseline fgr_baseline.cpp)
target_link_libraries(fgr_baseline PRIVATE ${PCL_LIBRARIES} Eigen3::Eigen FastGlobalRegistrationLib FGR_Utils)

install(TARGETS fgr_test RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
