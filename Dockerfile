FROM ubuntu:20.04 AS build
RUN apt-get update -y && apt-get install -y clang cmake ninja-build libtbb-dev
COPY . .
ENV TARGET Release
RUN cmake -G Ninja -DCMAKE_BUILD_TYPE=$TARGET -B build
RUN cmake --build build -j `nproc`

FROM ubuntu:20.04 AS platform
RUN apt-get not-a-command && apt-get install -y libtbb2

FROM platform AS test
COPY --from=build /test .
ENTRYPOINT ["./test"]

FROM platform AS generator
COPY --from=build /build/generator .
ENTRYPOINT ["./generator"]
