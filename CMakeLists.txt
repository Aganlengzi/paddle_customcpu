cmake_minimum_required(VERSION 3.10)

project(paddle-custom_cpu CXX C)

set(PLUGIN_NAME        "paddle_custom_cpu")
set(PLUGIN_VERSION     "0.0.1")

set(PADDLE_PLUGIN_DIR  "/opt/conda/lib/python3.7/site-packages/paddle-plugins/")
set(PADDLE_INC_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/include/")
set(PADDLE_LIB_DIR     "/opt/conda/lib/python3.7/site-packages/paddle/fluid/")

############ 三方依赖，此Demo中使用Paddle相同依赖
set(BOOST_INC_DIR      "/workspace/tmp/Paddle/build/third_party/boost/src/extern_boost")
set(GFLAGS_INC_DIR     "/workspace/tmp/Paddle/build/third_party/install/gflags/include")
set(GLOG_INC_DIR       "/workspace/tmp/Paddle/build/third_party/install/glog/include")
set(MKLDNN_INC_DIR     "/workspace/tmp/Paddle/build/third_party/install/mkldnn/include")
set(THIRD_PARTY_INC_DIR ${BOOST_INC_DIR} ${GFLAGS_INC_DIR} ${GLOG_INC_DIR} ${MKLDNN_INC_DIR})

include_directories(${PADDLE_INC_DIR} ${THIRD_PARTY_INC_DIR})
link_directories(${PADDLE_LIB_DIR})

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)  # for out fluid seperate
add_definitions(-DPADDLE_WITH_MKLDNN)  # for out MKLDNN compiling

############ 编译插件
add_library(${PLUGIN_NAME} SHARED runtime.cc add_kernel.cc)
target_link_libraries(${PLUGIN_NAME} PRIVATE :core_avx.so)  # special name

############ 打包插件
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
