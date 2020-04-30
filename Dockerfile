FROM ubuntu:20.04
COPY . /usr/src/project/
WORKDIR /usr/src/project/
RUN apt-get update -y && apt-get install -y clang cmake libtbb-dev
RUN cmake -B build . && cmake --build build -- -j
CMD ["./build/generator"]
