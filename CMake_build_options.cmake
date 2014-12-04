# Maintainer: harperic

######################
## CUDA related options
find_package(CUDA QUIET)
if (CUDA_FOUND)
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" on)
else (CUDA_FOUND)
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" off)
endif (CUDA_FOUND)

# disable CUDA if the intel compiler is detected
if (CMAKE_CXX_COMPILER MATCHES "icpc")
    set(ENABLE_CUDA OFF CACHE BOOL "Forced OFF by the use of the intel c++ compiler" FORCE)
endif (CMAKE_CXX_COMPILER MATCHES "icpc")

if (ENABLE_CUDA)
    option(ENABLE_NVTOOLS "Enable NVTools profiler integration" off)
endif (ENABLE_CUDA)
