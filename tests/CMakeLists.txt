add_compile_options(-Wno-error=ignored-qualifiers)
add_compile_options(-Wno-error=pmf-conversions)
add_compile_options(-Wno-error=sign-compare)
if(USE_PYTHONTEST_TEST)
    add_subdirectory(pythontest)
endif()
if(DOMAIN_LAYERED_TEST)
    add_subdirectory(dlt)
endif()


