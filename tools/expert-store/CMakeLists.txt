set(TARGET llama-expert-store)
add_executable(${TARGET} expert-store.cpp)
target_link_libraries(${TARGET} PRIVATE llama ggml vendor::nlohmann ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
