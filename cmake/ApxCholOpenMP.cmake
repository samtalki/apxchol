function(apxchol_enable_openmp target visibility)
    if(DEFINED APXCHOL_ENABLE_OPENMP AND NOT APXCHOL_ENABLE_OPENMP)
        message(STATUS "apxchol: OpenMP disabled; building serial runtime")
        return()
    endif()

    set(APXCHOL_OPENMP_INCLUDE_DIR "" CACHE PATH
        "OpenMP include directory supplied by the host")
    set(APXCHOL_OPENMP_LIBRARY "" CACHE FILEPATH
        "OpenMP runtime supplied by the host")
    if((APXCHOL_OPENMP_INCLUDE_DIR AND NOT APXCHOL_OPENMP_LIBRARY) OR
       (APXCHOL_OPENMP_LIBRARY AND NOT APXCHOL_OPENMP_INCLUDE_DIR))
        message(FATAL_ERROR
            "APXCHOL_OPENMP_INCLUDE_DIR and APXCHOL_OPENMP_LIBRARY must be set together")
    endif()
    if(APXCHOL_OPENMP_INCLUDE_DIR AND APXCHOL_OPENMP_LIBRARY)
        target_include_directories(${target} ${visibility} ${APXCHOL_OPENMP_INCLUDE_DIR})
        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} ${visibility}
                $<$<COMPILE_LANGUAGE:CXX>:-Xpreprocessor>
                $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
        else()
            target_compile_options(${target} ${visibility}
                $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
        endif()
        target_link_libraries(${target} ${visibility} ${APXCHOL_OPENMP_LIBRARY})
        message(STATUS "apxchol: OpenMP enabled via explicit host runtime")
        return()
    endif()

    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} ${visibility} OpenMP::OpenMP_CXX)
        message(STATUS "apxchol: OpenMP enabled via ${OpenMP_CXX_LIB_NAMES}")
        return()
    endif()

    # AppleClang does not discover Homebrew libomp through FindOpenMP. Locate
    # the headers and runtime explicitly while keeping OpenMP optional.
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_path(APXCHOL_LIBOMP_INCLUDE_DIR omp.h
            HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        find_library(APXCHOL_LIBOMP_LIBRARY omp
            HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
        if(APXCHOL_LIBOMP_INCLUDE_DIR AND APXCHOL_LIBOMP_LIBRARY)
            target_include_directories(${target} ${visibility} ${APXCHOL_LIBOMP_INCLUDE_DIR})
            target_compile_options(${target} ${visibility}
                $<$<COMPILE_LANGUAGE:CXX>:-Xpreprocessor>
                $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
            target_link_libraries(${target} ${visibility} ${APXCHOL_LIBOMP_LIBRARY})
            message(STATUS "apxchol: OpenMP enabled via Homebrew libomp")
            return()
        endif()
    endif()

    message(STATUS "apxchol: OpenMP not found; building serial runtime")
endfunction()
