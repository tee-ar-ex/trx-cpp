# Use Ubuntu as base image
FROM ubuntu:latest

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Update Ubuntu packages and install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    libeigen3-dev \
    nlohmann-json3-dev \
    libspdlog-dev \
    build-essential \
    zlib1g-dev\
    && rm -rf /var/lib/apt/lists/*

# Install Libzip
RUN git clone https://github.com/nih-at/libzip.git \
    && cd libzip \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install

# Install Google Test v1.12.1
RUN git clone -b release-1.12.1 https://github.com/google/googletest.git \
    && cd googletest \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install

# Install mio
RUN git clone -b master https://github.com/vimpunk/mio.git \
    && cd mio \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install

#setting the working directory
WORKDIR /app

#Copy the project to the container
COPY . .

#Run and build the project
RUN mkdir build && \
    cd build && \
    cmake .. && \
    make


