FROM ubuntu:20.04 AS builder
WORKDIR /usr/src/project/
COPY . .
RUN apt-get update -y && apt-get install -y clang cmake libtbb-dev
RUN cmake -B build . && cmake --build build -- -j

FROM ubuntu:20.04
WORKDIR /root/
COPY --from=builder /usr/src/project/build/generator .
RUN apt-get update -y && apt-get install -y libtbb2
CMD ["./generator"]
