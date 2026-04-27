cmake_minimum_required(VERSION 3.18)
project(PINN_CPP CXX)

# Usage: cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH="C:/libs/libtorch"
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: match libtorch runtime
if(MSVC)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MD" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Core library
add_library(pinn_core STATIC
    src/model.cpp
    src/physics.cpp
    src/sampling.cpp
    src/train.cpp
    src/eval.cpp
)
target_link_libraries(pinn_core ${TORCH_LIBRARIES})
set_property(TARGET pinn_core PROPERTY CXX_STANDARD 17)

# Executable: full 8-experiment run
add_executable(pinn_experiments src/run_experiments.cpp)
target_link_libraries(pinn_experiments pinn_core ${TORCH_LIBRARIES})
set_property(TARGET pinn_experiments PROPERTY CXX_STANDARD 17)

# Executable: standalone eval CLI
add_executable(pinn_eval src/eval_main.cpp)
target_link_libraries(pinn_eval pinn_core ${TORCH_LIBRARIES})
set_property(TARGET pinn_eval PROPERTY CXX_STANDARD 17)

# Windows: copy libtorch DLLs next to executables
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET pinn_experiments POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:pinn_experiments>
    )
    add_custom_command(TARGET pinn_eval POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:pinn_eval>
    )
endif()
