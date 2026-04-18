add_executable(acpp-metal-archive-build main.cpp)

target_include_directories(acpp-metal-archive-build PRIVATE
    ${HIPSYCL_SOURCE_DIR}
    ${HIPSYCL_SOURCE_DIR}/include
    ${ACPP_BINARY_ROOT}/include
    ${METAL_INCLUDE_DIR})

target_link_libraries(acpp-metal-archive-build PRIVATE
    "-framework Metal" "-framework Foundation" "-framework CoreFoundation")

set_target_properties(acpp-metal-archive-build PROPERTIES
    INSTALL_RPATH ${base}/../lib/)

install(TARGETS acpp-metal-archive-build DESTINATION bin)
