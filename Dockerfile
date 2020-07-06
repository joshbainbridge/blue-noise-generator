FROM ubuntu:20.04 AS toolchain
RUN apt-get update -y && apt-get install -y clang cmake ninja-build libtbb-dev

FROM toolchain AS debug-build
COPY . .
ENV TARGET Debug
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=$TARGET -B build
RUN cmake --build build -j

FROM ubuntu:20.04 as debug-deploy
RUN apt-get update -y && apt-get install -y libtbb2
COPY --from=debug-build /build/generator .
ENTRYPOINT ["./generator"]

FROM toolchain AS release-build
COPY . .
ENV TARGET Release
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=$TARGET -B build
RUN cmake --build build -j

FROM ubuntu:20.04 as release-deploy
RUN apt-get update -y && apt-get install -y libtbb2
COPY --from=release-build /build/generator .
ENTRYPOINT ["./generator"]
