from conans import ConanFile, CMake


class TRX(ConanFile):
    name = "trx"
    version = "1.0.0"
    description = "TRX (tee-ar-ex), a community-oriented unified tractography file"
    license = "BSD-2-Clause"
    url = "https://tee-ar-ex.github.io/"
    settings = "os", "arch", "compiler", "build_type"
    generators = "cmake"
