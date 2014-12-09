######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")
string(TOUPPER ${BOOST_PYTHON_COMPONENT} UPPER_BOOST_PYTHON_COMPONENT )
target_link_libraries(${target} ${Boost_${UPPER_BOOST_PYTHON_COMPONENT}_LIBRARY} ${PYTHON_LIBRARIES} ${TBB_LIBRARY})
endmacro(setup_pymodule)
#####################

######################
# Compile a list of ispc files
# this is loosly based on FindCUDA.cmake
macro(compile_ispc generated_files)
# reset the output variable
set(_compile_ispc_generated_files "")

# Iterate over the macro arguments and create custom
# commands for all the .ispc files.
foreach(file ${ARGN})
    get_filename_component( basename ${file} NAME )
    set(generated_file_path ${CMAKE_CURRENT_BINARY_DIR})
    set(generated_file_basename "${basename}")
    set(generated_obj_file "${generated_file_path}/${generated_file_basename}.o")
    set(generated_header_file "${generated_file_path}/${generated_file_basename}.h")

    # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
    get_filename_component(file_path "${file}" PATH)
    if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
    else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
    endif()

    add_custom_command(OUTPUT ${generated_obj_file} ${generated_header_file}
                       IMPLICIT_DEPENDS CXX
                       DEPENDS ${source_file}
                       COMMAND ${ISPC_EXECUTABLE} ${source_file} -o ${generated_obj_file} -h ${generated_header_file}
                      )

    set_source_files_properties(${generated_obj_file} PROPERTIES GENERATED TRUE)
    list(APPEND _compile_ispc_generated_files ${generated_obj_file})
endforeach()

# return the result
set(${generated_files} ${_compile_ispc_generated_files})
endmacro(compile_ispc)

######################
# Compile a list of cuda files
macro(add_source_dir dir)
add_subdirectory(${dir})
# reset the output variable

# Find all cuda src files
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cc)
file(GLOB CU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cu)
file(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cuh)
# add to the source group
list(APPEND _freud_sources ${SRCS} ${HDRS})
list(APPEND _freud_cu_sources ${CU_SRCS})
source_group(${dir} FILES ${CU_SRCS} ${HDRS})

endmacro(add_source_dir)

######################

set(_freud_sources "")
set(_freud_cu_sources "")

# have to manually set density because of templating
set(_dir_list util trajectory bootstrap cluster interface kspace
    lindemann locality molfile order pairing parallel
    pmft shapesplit sphericalharmonicorderparameters viz voronoi)

if (ENABLE_CUDA)
list(APPEND _dir_list cudacell cudapmft)
endif (ENABLE_CUDA)

foreach (dir ${_dir_list})
add_source_dir(${dir})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
endforeach (dir)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/density
                    ${CMAKE_CURRENT_BINARY_DIR}
                    )


list(APPEND _freud_sources
            density/RDF.cc
            density/RDF.h
            density/GaussianDensity.cc
            density/GaussianDensity.h
            density/LocalDensity.h
            density/LocalDensity.cc
            )
set(SOURCES ${_freud_sources})
set_source_files_properties(${SOURCES} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

# compile_ispc(ispc_generated_files viz/colorutil.ispc)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_freud_cu_sources} OPTIONS ${CUDA_ADDTIONAL_OPTIONS} SHARED)
source_group(cuda_generated FILES ${_CUDA_GENERATED_FILES})
endif (ENABLE_CUDA)

# add_library(_freud MODULE ${ispc_generated_files} ${SOURCES} util/num_util.cc util/num_util.h module.cc)
add_library(_freud MODULE ${SOURCES} ${_CUDA_GENERATED_FILES} util/num_util.cc util/num_util.h module.cc)
if (ENABLE_CUDA)
target_link_libraries(_freud ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})
endif (ENABLE_CUDA)
# .dylib is not regognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_freud PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_freud)
setup_pymodule(_freud)

INSTALL(TARGETS _freud
  LIBRARY DESTINATION .
  )
