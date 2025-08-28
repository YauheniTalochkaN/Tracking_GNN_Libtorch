cmake_minimum_required(VERSION 3.11)

#----------------------------------------------------------------------------------------------------------------------

set(CUDASYS $ENV{CUDASYS})
set(CMAKE_CUDA_TOOLKIT_ROOT_DIR "${CUDASYS}")
set(NVCOMPILERS $ENV{NVCOMPILERS})
set(NVARCH $ENV{NVARCH})
set(CMAKE_CUDA_COMPILER "${NVCOMPILERS}/${NVARCH}/25.3/compilers/bin/nvcc")
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA is not found!")
endif()
link_directories("${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/lib64")
find_package(CUDAToolkit 12.8 REQUIRED COMPONENTS cudart cublas cusparse cufft curand cusolver cusparseLt cudss)

#----------------------------------------------------------------------------------------------------------------------

set(CAFFE2_USE_CUDNN ON)
set(CAFFE2_USE_CUSPARSELT ON)
set(USE_CUDSS ON)     

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_C_COMPILER "gcc-14")
set(CMAKE_CXX_COMPILER "g++-14")
set(CMAKE_CUDA_HOST_COMPILER "g++-14")

#----------------------------------------------------------------------------------------------------------------------

project(cutensors LANGUAGES CXX CUDA)

#----------------------------------------------------------------------------------------------------------------------

set(nvtx3_dir "${CUDASYS}/NVTX/c")
include("${nvtx3_dir}/nvtxImportedTargets.cmake")

#----------------------------------------------------------------------------------------------------------------------

set(TORCHSYS $ENV{TORCHSYS})
set(Torch_DIR "${TORCHSYS}/share/cmake/Torch")
find_package(Torch REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

find_package(ROOT 6.30 CONFIG REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

find_package(OpenMP)
#find_package(MPI REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

find_package(yaml-cpp REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wall -O2 -fPIC -ffast-math -mavx")

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

#----------------------------------------------------------------------------------------------------------------------

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB train_source ${PROJECT_SOURCE_DIR}/train.cc)
file(GLOB evaluate_gnn_source ${PROJECT_SOURCE_DIR}/evaluate_gnn.cc)
file(GLOB evaluate_events_source ${PROJECT_SOURCE_DIR}/evaluate_events.cc)
file(GLOB data_preprocessing_source ${PROJECT_SOURCE_DIR}/data_preprocessing.cc)
file(GLOB train_data_preprocessing_source ${PROJECT_SOURCE_DIR}/train_data_preprocessing.cc)

#----------------------------------------------------------------------------------------------------------------------

separate_arguments(ROOT_EXE_LINKER_FLAGS)

#----------------------------------------------------------------------------------------------------------------------

add_library(TorchNetworks SHARED ${sources})
target_include_directories(TorchNetworks PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                ${ROOT_INCLUDE_DIRS})
target_compile_features(TorchNetworks PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(TorchNetworks PUBLIC ${TORCH_LIBRARIES}  
                                           CUDA::cudart 
                                           CUDA::cublas 
                                           CUDA::cusparse 
                                           CUDA::cufft 
                                           CUDA::curand
                                           CUDA::cusolver
                                           cudnn
                                           cusparseLt
                                           cudss
                                           nvtx3-cpp
                                           ${ROOT_LIBRARIES} 
                                           ${ROOT_EXE_LINKER_FLAGS} 
                                           ROOT::Geom 
                                           OpenMP::OpenMP_CXX
                                           #MPI::MPI_CXX
                                           yaml-cpp)
                                 
#----------------------------------------------------------------------------------------------------------------------

add_executable(train ${train_source})
target_include_directories(train PRIVATE ${PROJECT_SOURCE_DIR}/include
                                         ${ROOT_INCLUDE_DIRS})
target_compile_features(train PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(train PRIVATE TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(evaluate_gnn ${evaluate_gnn_source})
target_include_directories(evaluate_gnn PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                ${ROOT_INCLUDE_DIRS})
target_compile_features(evaluate_gnn PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(evaluate_gnn TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(evaluate_events ${evaluate_events_source})
target_include_directories(evaluate_events PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                   ${ROOT_INCLUDE_DIRS})
target_compile_features(evaluate_events PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(evaluate_events TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(data_preprocessing ${data_preprocessing_source})
target_include_directories(data_preprocessing PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                      ${ROOT_INCLUDE_DIRS})
target_compile_features(data_preprocessing PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(data_preprocessing TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(train_data_preprocessing ${train_data_preprocessing_source})
target_include_directories(train_data_preprocessing PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                            ${ROOT_INCLUDE_DIRS})
target_compile_features(train_data_preprocessing PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(train_data_preprocessing TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

set(MAIN_SCRIPTS )

foreach(_script ${MAIN_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()

#----------------------------------------------------------------------------------------------------------------------

install(TARGETS train evaluate_gnn evaluate_events data_preprocessing train_data_preprocessing 
        DESTINATION ${PROJECT_SOURCE_DIR}/install)

#----------------------------------------------------------------------------------------------------------------------

add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

#----------------------------------------------------------------------------------------------------------------------