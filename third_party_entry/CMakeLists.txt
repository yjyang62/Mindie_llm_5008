cmake_minimum_required(VERSION 3.19)
project(third_party_builder)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/default_config.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/compile_options.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/build_open_source.cmake)

add_custom_target(all_third_party DEPENDS ${PKG_LIST})
