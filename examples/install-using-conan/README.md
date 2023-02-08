
```bash
conan editable add ../.. trx/1.0.0@user/channel
```

```bash
cd .. && rm -Rf build && mkdir build && cd build
conan profile new ./.conan --detect && conan profile update settings.compiler.libcxx=libstdc++11 ./.conan
conan install --build=missing --settings=build_type=Debug --profile ./.conan ..
cmake -S .. -B . && make
```
