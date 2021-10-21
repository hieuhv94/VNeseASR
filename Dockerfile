# ==================================================================
# module list
# ------------------------------------------------------------------
# Ubuntu           20.04
# OpenMPI          latest       (apt)
# cmake            3.16.3       (apt)
# arrayfire        3.7.3        (git, CPU backend)
# libsndfile       latest       (apt)
# oneDNN           v2.0         (git)
# Gloo             1da2117      (git)
# FFTW             latest       (apt)
# KenLM            0c4dd4e      (git)
# GLOG             latest       (apt)
# gflags           latest       (apt)
# ==================================================================

#############################################################################
#                             APT IMAGE + CMAKE                             #
#############################################################################

FROM ubuntu:20.04 as cpu_base_builder

ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        curl \
        git \
        g++ \
        cmake \
        # for kenlm
        libboost-thread-dev libboost-test-dev libboost-system-dev libboost-program-options-dev \
        # for arrayfire CPU backend
        libboost-stacktrace-dev \
        # OpenBLAS
        libopenblas-dev liblapacke-dev \
        # ATLAS
        libatlas3-base libatlas-base-dev liblapacke-dev \
        # FFTW
        libfftw3-dev \
        # ssh for OpenMPI
        openssh-server openssh-client \
        # for OpenMPI
        libopenmpi-dev openmpi-bin \
        # for kenlm
        zlib1g-dev libbz2-dev liblzma-dev && \
        # libssl
        libssl-dev \
        # libav and alsa-lib
        libavutil-dev libavcodec-dev libavformat-dev libswresample-dev libasound2-dev \
        # libasio
        libasio-dev \
        # libjsoncpp
        libjsoncpp-dev \
# ==================================================================
# clean up everything
# ------------------------------------------------------------------
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

#############################################################################
#                                DEPS IMAGES                                #
#############################################################################

FROM cpu_base_builder as cpu_arrayfire
# ==================================================================
# arrayfire with CPU backend https://github.com/arrayfire/arrayfire/wiki/
# ------------------------------------------------------------------
RUN cd /tmp && git clone --branch v3.7.3 --depth 1 --recursive --shallow-submodules https://github.com/arrayfire/arrayfire.git && \
    mkdir -p arrayfire/build && cd arrayfire/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/opt/arrayfire \
             -DAF_BUILD_CPU=ON \
             -DAF_BUILD_CUDA=OFF \
             -DAF_BUILD_OPENCL=OFF \
             -DAF_BUILD_EXAMPLES=OFF \
             -DAF_WITH_IMAGEIO=OFF \
             -DBUILD_TESTING=OFF \
             -DAF_BUILD_DOCS=OFF && \
    make install -j$(nproc)

FROM cpu_base_builder as cpu_onednn
# ==================================================================
# oneDNN https://github.com/oneapi-src/oneDNN
# ------------------------------------------------------------------
RUN cd /tmp && git clone --branch v2.0 --depth 1 https://github.com/oneapi-src/onednn.git && \
    mkdir -p onednn/build && cd onednn/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/opt/onednn \
             -DDNNL_BUILD_EXAMPLES=OFF && \
    make install -j$(nproc)

FROM cpu_base_builder as cpu_gloo
# ==================================================================
# Gloo https://github.com/facebookincubator/gloo.git
# ------------------------------------------------------------------
RUN cd /tmp && git clone https://github.com/facebookincubator/gloo.git && \
    cd gloo && git checkout 1da2117 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/opt/gloo \
             -DUSE_MPI=ON && \
    make install -j$(nproc)

FROM cpu_base_builder as cpu_kenlm
# ==================================================================
# KenLM https://github.com/kpu/kenlm
# ------------------------------------------------------------------
RUN cd /tmp && git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && git checkout 0c4dd4e && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/opt/kenlm \
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j$(nproc)

#############################################################################
#                             FINAL IMAGE                                   #
#############################################################################

FROM cpu_base_builder as cpu_final

COPY --from=cpu_arrayfire  /opt/arrayfire  /opt/arrayfire
COPY --from=cpu_onednn     /opt/onednn     /opt/onednn
COPY --from=cpu_gloo       /opt/gloo       /opt/gloo
COPY --from=cpu_kenlm      /opt/kenlm      /opt/kenlm

ENV KENLM_ROOT=/opt/kenlm
ENV LD_LIBRARY_PATH=/opt/arrayfire/lib/:/opt/onednn/lib/:
# ==================================================================
# flashlight
# ------------------------------------------------------------------
COPY fl_build_shared_library.patch /tmp/
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    # for glog
    libgoogle-glog-dev libgoogle-glog0v5 \
    # libsndfile
    libsndfile1-dev \
    # libboost-locale-dev
    libboost-locale-dev
RUN cd /tmp && git clone --recursive https://github.com/facebookresearch/flashlight.git && \
    cd flashlight && git apply /tmp/fl_build_shared_library.patch && mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DFL_BACKEND=CPU -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC" \
    -DFL_LIBRARIES_USE_MKL=OFF -DFL_BUILD_TESTS=OFF -DFL_BUILD_EXAMPLES=OFF -DFL_BUILD_RECIPES=OFF \
    -DFL_BUILD_APP_ASR=ON  -DDNNL_DIR=/opt/onednn/lib/cmake/dnnl && \
    make -j2 && make install
# ==================================================================
# libtensorflow 1.14.0
# ------------------------------------------------------------------
RUN cd /tmp && mkdir clib && curl -L "https://storage.googleapis.com/tensorflow-nightly/github/tensorflow/lib_package/libtensorflow-cpu-linux-x86_64.tar.gz" | tar -C clib -xz \
    cp -r clib/include/* /usr/local/include && cp -r clib/lib/* /usr/local/lib
# ==================================================================
# websocketcpp
# ------------------------------------------------------------------
RUN cd /tmp && git clone https://github.com/zaphoyd/websocketpp && cd websocketpp && cmake . && make install
# ==================================================================
# pnks2t library
# ------------------------------------------------------------------
RUN mkdir /tmp/pnk_s2t
COPY pnk_s2t /tmp/pnk_s2t
RUN export KENLM_ROOT_DIR=/opt/kenlm/ &&  export KENLM_INC=/opt/kenlm/include/kenlm/ && \
    cd /tmp/pnk_s2t && mkdir -p build && cd build && \
    cmake .. && make -j4 && make install
# ==================================================================
# clean up everything
# ------------------------------------------------------------------
RUN    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*
