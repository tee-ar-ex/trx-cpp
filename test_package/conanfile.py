import os

from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout


class TrxCppTestPackage(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "CMakeDeps", "CMakeToolchain"
    test_type = "explicit"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        exe_name = "example.exe" if self.settings.os == "Windows" else "example"
        self.run(os.path.join(self.build_folder, exe_name), env="conanrun")
