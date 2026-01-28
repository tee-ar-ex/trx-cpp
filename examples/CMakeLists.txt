add_executable(trxinfo trxinfo.cpp)
target_link_libraries(trxinfo PRIVATE trx CLI11::CLI11)
target_compile_features(trxinfo PRIVATE cxx_std_11)

install(TARGETS trxinfo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
