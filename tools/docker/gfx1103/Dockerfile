# Strata's Inkling-2-Mini engine for the Radeon 780M / 760M (gfx1103), built from this checkout on ROCm 10.1
# (the stack docs/RADEON_780M.md was measured with). Experimental, like the 780M port itself.
#
# Build once on the host, from the repository root (Dockhand only starts the image, it cannot build from a host folder):
#
#   docker build -t strata-780m:0.1.67 -f tools/docker/gfx1103/Dockerfile .
#
# Both stages use the same ROCm image, so the second one adds only the Python packages, the Terminal and the engine.
# The full image is needed at run time too: hipBLAS, hipBLASLt and rocBLAS with their gfx1103 kernel files.

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:10.1.0-full

# ---- 1. compile strata-inkling for gfx1103
FROM ${ROCM_IMAGE} AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY engine engine
# the HIP build globs ../terminal/tools/vision (engine/cmake/backends/hip.cmake)
COPY terminal terminal
ARG HIP_ARCH=gfx1103
ARG JOBS=8
RUN cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
        -DCMAKE_HIP_ARCHITECTURES=${HIP_ARCH} \
        -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        "-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode" \
    && grep -q '^CMAKE_HIP_FLAGS_RELEASE:STRING=-O3' build/CMakeCache.txt \
    && cmake --build build --target strata-inkling -j ${JOBS} \
    && build/src/inkling2/strata-inkling --version

# ---- 2. the runtime: the same ROCm image, Strata's server and the engine
FROM ${ROCM_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv tini \
    && rm -rf /var/lib/apt/lists/* \
    && echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig
COPY terminal/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
COPY terminal /opt/strata/terminal
COPY --from=build /src/build/src/inkling2/strata-inkling /opt/strata/bin/strata-inkling
COPY tools/docker/gfx1103/start.py /opt/strata/start.py
ENV HOME=/data/home XDG_STATE_HOME=/data/state XDG_CONFIG_HOME=/data/config PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    LANG=C.UTF-8 STRATA_IN_DOCKER=1 STRATA_EXPERIMENTAL_GFX1103=1
VOLUME ["/data"]
EXPOSE 8080
LABEL org.opencontainers.image.title="Strata Inkling-2-Mini for Radeon 780M (gfx1103)"
HEALTHCHECK --interval=30s --timeout=8s --start-period=15m --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import json,urllib.request,sys; h=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5)); sys.exit(0 if h.get('loaded') else 1)"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/venv/bin/python", "/opt/strata/start.py"]
