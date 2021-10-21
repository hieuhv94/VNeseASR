## Build intructions
```
mkdir -p build && cd build
cmake ..
make -j$(nproc) && make install
```

``

## Run examples
- With cpu
```
cd examples
cmake .
make -j2
```
