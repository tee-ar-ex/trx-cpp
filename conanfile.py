from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import get, copy
import os


class TrxCppConan(ConanFile):
    name = "trx-cpp"
    version = "0.1.0"
    package_type = "library"
    license = "BSD-2-Clause"
    url = "https://github.com/tractdata/trx-cpp"
    homepage = "https://github.com/tractdata/trx-cpp"
    description = "C++ library for reading and writing the TRX tractography format."
    topics = ("tractography", "mmap", "neuroimaging")
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_tests": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_tests": False,
    }
    generators = ("CMakeDeps",)
    exports_sources = (
        "CMakeLists.txt",
        "src/*",
        "include/*",
        "third_party/*",
        "cmake/*",
        "tests/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def requirements(self):
        self.requires("libzip/1.10.1")
        self.requires("nlohmann_json/3.11.3")
        self.requires("eigen/3.4.0")
        self.requires("spdlog/1.12.0")

    def build_requirements(self):
        if self.options.with_tests:
            # Only needed for building/running tests, not for consumers.
            self.test_requires("gtest/1.14.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_CXX_STANDARD"] = 11
        tc.variables["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"
        tc.variables["CMAKE_CXX_EXTENSIONS"] = "OFF"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(variables={"TRX_BUILD_TESTS": self.options.with_tests})
        cmake.build()
        if self.options.with_tests:
            cmake.ctest()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        # Ensure libzip headers are available to consumers even when the
        # libzip Conan package does not expose include dirs via CMakeDeps.
        libzip_dep = self.dependencies.get("libzip")
        if libzip_dep and libzip_dep.package_folder:
            libzip_include = os.path.join(libzip_dep.package_folder, "include")
            copy(self, "zip.h", src=libzip_include, dst=os.path.join(self.package_folder, "include"))
            copy(self, "zipconf.h", src=libzip_include, dst=os.path.join(self.package_folder, "include"))

        nlohmann_dep = self.dependencies.get("nlohmann_json")
        if nlohmann_dep and nlohmann_dep.package_folder:
            nlohmann_include = os.path.join(nlohmann_dep.package_folder, "include")
            copy(self, "nlohmann/*", src=nlohmann_include, dst=os.path.join(self.package_folder, "include"))
        eigen_dep = self.dependencies.get("eigen")
        if eigen_dep and eigen_dep.package_folder:
            eigen_include = os.path.join(eigen_dep.package_folder, "include", "eigen3")
            copy(self, "Eigen/*", src=eigen_include, dst=os.path.join(self.package_folder, "include"))

        mio_include = os.path.join(self.source_folder, "third_party", "mio", "include")
        copy(self, "mio/*", src=mio_include, dst=os.path.join(self.package_folder, "include"))

        spdlog_dep = self.dependencies.get("spdlog")
        if spdlog_dep and spdlog_dep.package_folder:
            spdlog_include = os.path.join(spdlog_dep.package_folder, "include")
            copy(self, "spdlog/*", src=spdlog_include, dst=os.path.join(self.package_folder, "include"))
        fmt_dep = self.dependencies.get("fmt")
        if fmt_dep and fmt_dep.package_folder:
            fmt_include = os.path.join(fmt_dep.package_folder, "include")
            copy(self, "fmt/*", src=fmt_include, dst=os.path.join(self.package_folder, "include"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "trx-cpp")
        self.cpp_info.set_property("cmake_target_name", "trx-cpp::trx")
        self.cpp_info.set_property("pkg_config_name", "trx-cpp")
        self.cpp_info.components["trx"].set_property("cmake_target_name", "trx-cpp::trx")
        self.cpp_info.components["trx"].set_property("pkg_config_name", "trx-cpp")
        self.cpp_info.components["trx"].libs = ["trx"]
        self.cpp_info.components["trx"].requires = [
            "libzip::libzip",
            "spdlog::spdlog",
            "nlohmann_json::nlohmann_json",
            "eigen::Eigen3::Eigen",
        ]
        extra_includes = []
        for dep_name in ("nlohmann_json", "eigen"):
            dep = self.dependencies.get(dep_name)
            if dep and dep.package_folder:
                dep_include = os.path.join(dep.package_folder, "include")
                if os.path.isdir(dep_include):
                    extra_includes.append(dep_include)
        if extra_includes:
            self.cpp_info.components["trx"].includedirs.extend(extra_includes)
        if not self.options.shared and self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs.append("pthread")
