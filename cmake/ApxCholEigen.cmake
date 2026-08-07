include_guard(GLOBAL)

function(apxchol_find_or_fetch_eigen)
    find_package(Eigen3 3.3 CONFIG QUIET)
    if (Eigen3_FOUND)
        return()
    endif()

    include(FetchContent)
    # Eigen 3.4 predates CMP0077 and otherwise enables its own excluded tests,
    # which makes the parent project's CTest run report them as not built.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_TESTING OFF)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)
endfunction()
