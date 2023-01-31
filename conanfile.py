from conans import ConanFile, CMake


class TRX(ConanFile):
    name = "trx"
    version = "1.0.0"
    description = "TRX (tee-ar-ex), a community-oriented unified tractography file"
    license = "BSD-2-Clause"
    url = "https://tee-ar-ex.github.io/"
    settings = "os", "arch", "compiler", "build_type"
    generators = "cmake"
    requires = [
        "eigen/3.4.0@#3bc2bf84eff697283b6bd64d8262c423",
        "spdlog/1.10.0@#1e0f4eb6338d05e4bd6fcc6bf4734172",
        "nlohmann_json/3.11.2@#a35423bb6e1eb8f931423557e282c7ed",
        "libzip/1.8.0@#5a0a692ec9d7d8a4337eb79e55869639",
        "mio/cci.20201220",
        "gtest/cci.20210126",
    ]