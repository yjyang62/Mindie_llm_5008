add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        apply_top_k_top_p_custom_def.cpp
    )
    add_ops_compile_options(
        OP_NAME ApplyTopKTopPCustom
        OPTIONS --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE apply_top_k_top_p_custom ACLNNTYPE aclnn_inner)

    if (TARGET ${OPHOST_NAME}_opapi_obj)
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/apply_top_k_top_p_custom.cpp
        )
        target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()

    if (TARGET ${OPHOST_NAME}_tiling_obj)
        target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()
