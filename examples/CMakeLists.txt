add_executable(trxinfo trxinfo.cpp)
target_link_libraries(trxinfo PRIVATE trx cxxopts::cxxopts)
target_compile_features(trxinfo PRIVATE cxx_std_17)

install(TARGETS trxinfo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
