add_executable(trxinfo trxinfo.cpp)
target_link_libraries(trxinfo PRIVATE trx cxxopts::cxxopts)
target_compile_features(trxinfo PRIVATE cxx_std_17)

if(TRX_ENABLE_NIFTI)
    add_executable(trx-update-affine trx_nifti_affine.cpp)
    target_link_libraries(trx-update-affine PRIVATE trx trx-nifti)
    target_compile_features(trx-update-affine PRIVATE cxx_std_17)
endif()

install(TARGETS trxinfo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
