FROM ubuntu:20.04 AS toolchain
RUN apt-get update -y && apt-get install -y clang cmake ninja-build libtbb-dev

FROM toolchain AS build-debug
COPY . .
ENV TARGET Debug
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=$TARGET -B build
RUN cmake --build build -j

FROM ubuntu:20.04 as run-debug
RUN apt-get update -y && apt-get install -y libtbb2
COPY --from=build-debug /build/generator .
ENTRYPOINT ["./generator"]

FROM toolchain AS build-release
COPY . .
ENV TARGET Release
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=$TARGET -B build
RUN cmake --build build -j

FROM ubuntu:20.04 as run-release
RUN apt-get update -y && apt-get install -y libtbb2
COPY --from=build-release /build/generator .
ENTRYPOINT ["./generator"]
