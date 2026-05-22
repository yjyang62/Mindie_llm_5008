#============================================= Build Args ==============================================#
# 所有对外暴露的参数及默认值，可以在执行 docker build 时，通过 --build-arg 覆盖。
# example: docker build --build-arg --build-arg  ...
#=======================================================================================================#
ARG TORCH_VERSION=2.6.0
ARG TEMP_DIR=/tmp/build_cache
ARG FILE_SERVER=http://localhost:8000
ARG UBUNTU_MIRROR=http://mirrors.tools.huawei.com/ubuntu-ports/
ARG PIP_INDEX_URL=http://mirrors.tools.huawei.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.tools.huawei.com
ARG PYTHON_REQUIREMENTS_FILE=""
ARG TORCH_NPU_WHL_PATH
ARG CANN_TOOLKIT_FROM_PATH
ARG CANN_KERNELS_FROM_PATH
ARG CANN_NNAL_FROM_PATH
ARG ASCEND_INSTALL_DIR=/usr/local/Ascend

ARG MINDIE_LLM_FROM_PATH
ARG MINDIE_LLM_INSTALL_DIR=/usr/local/Ascend


#====================================== Stage 1: Build Base Image ======================================#
# 基于 ubuntu 镜像，安装和 Ascend 无关的 Linux 常用工具，制作基础镜像。
#=======================================================================================================#
FROM ubuntu:24.04 AS mindie_base

ARG TEMP_DIR
ARG UBUNTU_MIRROR
ARG FILE_SERVER
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG TORCH_VERSION
ARG PYTHON_REQUIREMENTS_FILE

# 更换 APT 源
RUN echo "Types: deb" > /etc/apt/sources.list.d/ubuntu.sources \
    && echo "URIs: http://mirrors.huaweicloud.com/ubuntu-ports/" >> /etc/apt/sources.list.d/ubuntu.sources \
    && echo "Suites: noble noble-updates noble-security" >> /etc/apt/sources.list.d/ubuntu.sources \
    && echo "Components: main restricted universe multiverse" >> /etc/apt/sources.list.d/ubuntu.sources \
    && echo "Architectures: arm64" >> /etc/apt/sources.list.d/ubuntu.sources

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PATH="/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}"

# 安装开发常用的软件包，可根据实际需要增删
# 分多次 apt-get install 安装，避免内存不足导致构建失败
RUN rm -f /var/cache/apt/archives/*.deb \
    && rm -f /var/cache/apt/archives/partial/*.deb \
    && rm -f /var/cache/apt/*.bin \
    && apt clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends tzdata locales apt-utils sudo xz-utils ca-certificates \
    && apt-get install -y --no-install-recommends python3.12-dev python3-pip python3.12-venv \
    && apt-get install -y --no-install-recommends wget curl aria2 unzip \
    && apt-get install -y --no-install-recommends zlib1g zlib1g-dev libbz2-dev liblzma-dev libffi-dev libssl-dev libsqlite3-dev \
    && apt-get install -y --no-install-recommends libxml2 libxslt1-dev pkg-config \
    && apt-get install -y --no-install-recommends libblas-dev liblapack-dev libopenblas-dev libhdf5-dev \
    && apt-get install -y --no-install-recommends pciutils net-tools ipmitool numactl \
    && apt-get install -y --no-install-recommends distro-info-data hwdata lsb-release media-types usb.ids \
    && apt-get install -y --no-install-recommends libglx-mesa0 libgmpxx4ldbl \
    && apt-get install -y --no-install-recommends vim lcov bc openssh-server git patch file \
    && apt-get install -y --no-install-recommends build-essential gcc g++ \
    && apt-get install -y --no-install-recommends gfortran gdb \
    && apt-get install -y --no-install-recommends make cmake \
    && apt-get install -y --no-install-recommends ninja-build \
    && apt-get install -y --no-install-recommends autoconf libtool pkg-config \
    && apt-get install -y --no-install-recommends libboost-system-dev libboost-thread-dev libboost-filesystem-dev libboost-chrono-dev \
    && apt-get install -y --no-install-recommends libspdlog-dev nlohmann-json3-dev libprotobuf-dev \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && update-ca-certificates -f \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# 设置字符集、编码等，避免中文显示乱码。
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Ubuntu24建议独立管理Python环境
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/torch.libs:$LD_LIBRARY_PATH"

# 设置 PIP 源为内部源，需要设置特定代理才能访问。如果外部不能访问，可根据实际情况自行修改。
# 安装常用的 Python 包，可根据实际需要增删。
RUN echo "[global]" > /etc/pip.conf \
    && echo "index-url = ${PIP_INDEX_URL}" >> /etc/pip.conf \
    && echo "trusted-host = ${PIP_TRUSTED_HOST}" >> /etc/pip.conf \
    && pip install --timeout=600 \
        numpy==1.26.0 \
        numba==0.63.1 \
        pandas==2.3.3 \
        matplotlib==3.10.8 \
        scipy==1.16.3 \
        tqdm==4.67.1 \
        requests==2.32.5 \
        pillow==12.0.0 \
        pybind11==3.0.1 \
        torch==2.6.0 \
        transformers==4.57.3 \
        wheel==0.46.1 \
        psutil==7.2.0 \
        pydantic==2.12.5 \
        posix_ipc==1.3.2 \
        networkx==3.6.1 \
        sentencepiece==0.2.1 \
        attrs==25.4.0 \
        decorator==5.2.1 \
        protobuf==6.33.2 \
        loguru==0.7.3 \
        multiprocess==0.70.18 \
        tabulate==0.9.0 \
        accelerate==1.12.0 \
        ipaddress==1.0.23

# 增加自定义的 requirements.txt，如果不传入，则
RUN if [ -n "${PYTHON_REQUIREMENTS_FILE}" ]; then \
        echo "Installing Python requirements from ${PYTHON_REQUIREMENTS_FILE}" \
        && mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
        && wget --no-proxy -nv ${FILE_SERVER}/${PYTHON_REQUIREMENTS_FILE} \
        && pip install --no-cache-dir -r "${PYTHON_REQUIREMENTS_FILE}"; \
    else \
        echo "No additional pip requirements provided"; \
    fi

# 编译其他依赖
WORKDIR /opt/src

# ===============================
# 1. gRPC
# ===============================
RUN git clone -b v1.76.0 --depth 1 https://gitcode.com/gh_mirrors/gr/grpc.git \
    && cd grpc \
    && sed -i '/#include <stddef.h>/a #include <string>' include/grpc/event_engine/memory_request.h \
    && sed -i '/#include <string>/a #include <limits>' src/core/channelz/v2tov1/property_list.cc \
    && sed -i '/#include "absl\/strings\/string_view.h"/a #include <algorithm>' src/core/util/glob.cc \
    && cd third_party \
    && git clone https://gitcode.com/gh_mirrors/zl/zlib.git \
    && cd zlib && git checkout f1f503da85d52e56aae11557b4d79a42bcaa2b86 && cd .. \
    && git clone -b v25.1 --depth 1 https://gitcode.com/GitHub_Trending/pr/protobuf.git \
    && git clone -b 20250814.1 --depth 1 https://gitcode.com/GitHub_Trending/ab/abseil-cpp.git \
    && git clone -b 2025-11-05 --depth 1 https://gitcode.com/gh_mirrors/re21/re2.git \
    && git clone -b v0.3.0 --depth 1 https://gitcode.com/grpc-mirror/opencensus-proto.git \
    && git clone -b v1.34.5 --depth 1 https://gitcode.com/gh_mirrors/ca/c-ares.git \
    && cp -r c-ares/* cares/cares \
    && cd .. \
    && cmake -S . -B build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DgRPC_BUILD_TESTS=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_C_FLAGS_RELEASE="-O2 -fstack-protector-strong" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O2 -fstack-protector-strong" \
        -DCMAKE_INSTALL_DO_STRIP=ON \
    && cmake --build build --parallel 32 \
    && cmake --install build \
    && ldconfig \
    && cd .. \
    && rm -rf /opt/src/grpc

# ===============================
# 2. prometheus-cpp
# ===============================
RUN git clone -b v1.3.0 --depth 1 https://gitcode.com/gh_mirrors/pr/prometheus-cpp.git \
    && cd prometheus-cpp \
    && cd 3rdparty/ \
    && git clone -b v1.17.0 --depth 1 https://gitcode.com/GitHub_Trending/go/googletest.git \
    && git clone -b v1.16 --depth 1 https://gitcode.com/gh_mirrors/civ/civetweb.git \
    && cd .. \
    && cmake -S . -B build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DENABLE_TESTING=OFF \
        -DENABLE_PUSH=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_C_FLAGS_RELEASE="-O2 -fstack-protector-strong" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O2 -fstack-protector-strong" \
        -DCMAKE_INSTALL_DO_STRIP=ON \
    && cmake --build build --parallel 32 \
    && cmake --install build \
    && ldconfig \
    && cd .. \
    && rm -rf /opt/src/prometheus-cpp

# ===============================
# 3. gloo
# ===============================
RUN git clone -b main --depth 1 https://gitcode.com/gh_mirrors/gloo/gloo.git \
    && cd gloo \
    && git checkout dc507d1eb822c4396aaca284efff498aba33c7dc \
    && cmake -S . -B build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TEST=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DCMAKE_C_FLAGS_RELEASE="-O2 -fstack-protector-strong" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O2 -fstack-protector-strong" \
        -DCMAKE_INSTALL_DO_STRIP=ON \
    && cmake --build build --parallel 32 \
    && cmake --install build \
    && ldconfig \
    && cd .. \
    && rm -rf /opt/src/gloo

# ===============================
# 4. libboundscheck (Huawei style)
# ===============================
# 常见 Ascend / Huawei 实现（若你有内部版本，可替换 URL）
RUN git clone -b v1.1.16 --depth 1 https://gitcode.com/openeuler/libboundscheck.git \
    && cd libboundscheck \
    && make \
    && cp lib/libboundscheck.so /usr/local/lib \
    && cp include/* /usr/local/include/ \
    && ldconfig \
    && cd .. \
    && rm -rf /opt/src/libboundscheck

# ===============================
# 5. httplib (header-only)
# ===============================
RUN git clone -b v0.28.0 --depth 1 https://gitcode.com/GitHub_Trending/cp/cpp-httplib.git \
    && cp cpp-httplib/httplib.h /usr/local/include/httplib.h \
    && rm -rf /opt/src/cpp-httplib

# ===============================
# 6. spdlog (header-only)
# ===============================
RUN git clone -b v1.15.3 --depth 1 https://gitcode.com/GitHub_Trending/sp/spdlog.git \
    && cd spdlog \
    && cmake -S . -B build \
        -G Ninja \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --parallel 32 \
    && cmake --install build \
    && rm -rf /opt/src/cpp-httplib


# 创建 HwHiAiUser 普通用户和组
RUN groupadd -g 2000 HwHiAiUser \
    && useradd -m -u 2000 -g HwHiAiUser -s /bin/bash HwHiAiUser \
    && echo 'HwHiAiUser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# 默认使用普通用户。
USER HwHiAiUser
WORKDIR /home/HwHiAiUser

CMD [ "/bin/bash" ]

#====================================== Stage 2: Build CANN Image ======================================#
# 在基础镜像中安装 CANN。
#=======================================================================================================#
FROM mindie_base AS cann

ARG TEMP_DIR
ARG FILE_SERVER
ARG ASCEND_INSTALL_DIR
ARG CANN_TOOLKIT_FROM_PATH
ARG CANN_KERNELS_FROM_PATH
ARG CANN_NNAL_FROM_PATH

USER root

ENV ASCEND_HOME_PATH=${ASCEND_INSTALL_DIR}/ascend-toolkit/latest \
    ASCEND_TOOLKIT_HOME=${ASCEND_INSTALL_DIR}/ascend-toolkit/latest \
    ATB_HOME_PATH=${ASCEND_INSTALL_DIR}/nnal/atb/latest/atb/cxx_abi_1

# 下载并安装 CANN，由于 CANN 体积较大，安装后镜像膨胀 10G 左右。
RUN mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
    && wget --no-proxy -nv ${FILE_SERVER}/${CANN_TOOLKIT_FROM_PATH} \
    && wget --no-proxy -nv ${FILE_SERVER}/${CANN_KERNELS_FROM_PATH} \
    && wget --no-proxy -nv ${FILE_SERVER}/${CANN_NNAL_FROM_PATH} \
    && bash ${CANN_TOOLKIT_FROM_PATH} --install --install-path=${ASCEND_INSTALL_DIR} --quiet \
    && bash ${CANN_KERNELS_FROM_PATH} --install --install-path=${ASCEND_INSTALL_DIR} --quiet \
    && bash ${CANN_NNAL_FROM_PATH} --install --install-path=${ASCEND_INSTALL_DIR} --quiet \
    && cd - && rm -rf ${TEMP_DIR} \
    && echo "source ${ASCEND_INSTALL_DIR}/cann/set_env.sh" >> /etc/profile \
    && echo "source ${ASCEND_INSTALL_DIR}/nnal/atb/set_env.sh --cxx_abi=1" >> /etc/profile

USER HwHiAiUser

#====================================== Stage 3: Build PTA Image =======================================#
# 在安装 CANN 的镜像基础上，安装 PTA，至此，完成MindIE编译所需依赖。
#=======================================================================================================#
FROM cann AS pta

ARG TEMP_DIR
ARG FILE_SERVER
ARG TORCH_NPU_WHL_PATH

USER root

# 安装 torch-npu
RUN mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
    && wget --no-proxy -nv ${FILE_SERVER}/${TORCH_NPU_WHL_PATH} \
    && pip install ${TORCH_NPU_WHL_PATH} --no-cache-dir \
    && cd - && rm -rf ${TEMP_DIR}

ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch_npu/lib:$LD_LIBRARY_PATH"

USER HwHiAiUser

#===================================== Stage 4: Build MindIE Image =====================================#
# 在安装 pta 的镜像基础上，编译安装 MindIE-LLM
#=======================================================================================================#
FROM pta AS mindie

ARG TEMP_DIR
ARG FILE_SERVER
ARG MINDIE_LLM_FROM_PATH
ARG MINDIE_LLM_INSTALL_DIR

USER root

# 安装 mindie
RUN mkdir -p ${TEMP_DIR} && cd ${TEMP_DIR} \
    && wget --no-proxy -nv ${FILE_SERVER}/${MINDIE_LLM_FROM_PATH} \
    && bash ${MINDIE_LLM_FROM_PATH} --install --install-path=${MINDIE_LLM_INSTALL_DIR} \
    && cd - && rm -rf ${TEMP_DIR}

ENV PATH="${MINDIE_LLM_INSTALL_DIR}/bin:$bin" \
    LD_LIBRARY_PATH="${MINDIE_LLM_INSTALL_DIR}/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="${MINDIE_LLM_INSTALL_DIR}/lib/python3.12/site-packages:$PYTHONPATH"

USER HwHiAiUser


# For quickly debug, use root user.
USER root
