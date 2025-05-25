add_executable(Testbed 
    main.cpp
)

target_include_directories(Testbed PUBLIC 
    ${PROJECT_SOURCE_DIR}/NDArray
)