#!/bin/bash
.PHONY: all, clean, clear, instrument

SHELL := /bin/bash
PWD := $(shell pwd)
LIB_OPENBLAS := ${PWD}/openblas/lib/

CXX = clang++
CXXFLAGS += -c -Wall $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)
CXXARGS += 

OPENCV_VERSION := 4.5.3

all: poc.out
	@echo Done!
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_OPENBLAS}:/usr/local/lib; \
	./$<


clean:
	rm -f *.o

clear: clean
	rm -rf *.out
	rm -rf dump
	rm -rf opencv-{OPENCV_VERSION}
	rm -rf openblas


poc.out: main.o myresnet.o mydataloader.o
	${CXX} main.o myresnet.o mydataloader.o -o $@ \
	-L${LIB_OPENBLAS} ${LDFLAGS} -lopenblas -lm

main.o: main.cpp tensor.h myresnet.h mydataloader.h utils.h common.h
	${CXX} -c -Wall -DSTOP_AT=0 -DBATCH_SIZE=1024 $< -o $@

myresnet.o: myresnet.cpp myresnet.h tensor.h utils.h common.h
	${CXX} -c -Wall $< -o $@

mydataloader.o: loader.o mydataloader.h tensor.h utils.h common.h
	cp $< $@


resnet.out: myresnet.cpp myresnet.h tensor.h utils.h
	${CXX} -Wall -o $@ $< -L${LIB_OPENBLAS} -lopenblas -lm

loader: loader.out
	mkdir -p ./dump
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib; \
	./$<

loader.out: loader.o
	${CXX} $< -o $@ ${LDFLAGS}

loader.o: mydataloader.cpp mydataloader.h common.h tensor.h
	${CXX} $< -o $@ ${CXXFLAGS} ${CXXARGS}

install_opencv: opencv-${OPENCV_VERSION}
	sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"; \
	sudo apt-get -y install build-essential cmake git libgtk2.0-dev pkg-config \
		libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
		libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
	mkdir -p opencv-${OPENCV_VERSION}/build/
	cd opencv-${OPENCV_VERSION}/build/; \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON ..; \
	# make -j4; \
	sudo make install

# v2: use apt to install
# reference: https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/
install_opencv_v2:
	sudo apt update
#	python3-opencv installs opencv for system's `python3` command (not for anaconda)
	sudo apt install libopencv-dev python3-opencv

opencv-${OPENCV_VERSION}: opencv.zip
	unzip opencv.zip -d ./

opencv.zip:
	curl -L -o opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip