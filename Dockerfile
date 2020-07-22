FROM ubuntu:20.04 AS build
RUN apt-get update -y && apt-get install -y clang cmake ninja-build libtbb-dev
ARG TARGET=Release
COPY . src
RUN cmake -D CMAKE_BUILD_TYPE=$TARGET -G Ninja -S src -B build
RUN cmake --build build
RUN cmake --install build

FROM ubuntu:20.04 AS platform
RUN apt-get update -y && apt-get install -y libtbb2

FROM platform AS test
COPY --from=build /src/scripts/test .
ENTRYPOINT ["./test"]

FROM platform AS generator
COPY --from=build /build/install/bin/generator .
ENTRYPOINT ["./generator"]
