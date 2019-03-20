#########################
#Makefile for MEMPTX
#2016-2017 Tong Zhang<ztong@vt.edu>
#

#CXX=/opt/toolchain/gcc/bin/g++
CXX=g++
CXXFLAGS=-g -Ofast -fPIC -march=native \
	 -fomit-frame-pointer \
	 -std=c++11
#	 -funroll-loops \
#	 -fprefetch-loop-arrays

CFLAGS=-g -Ofast -fPIC -march=native \
	-fomit-frame-pointer
#	-fprefetch-loop-arrays \
#	-funroll-loops

all:
	$(MAKE) all-nogpu

all-nogpu:
	$(CXX) $(CXXFLAGS) interceptor.cpp -c -o interceptor.o
	$(CXX) $(CXXFLAGS) scan.c -c -o scan.o
	$(CXX) $(CXXFLAGS) pmap.c -c -o pmap.o
	rm -f libinterceptor.a
	ar cr libinterceptor.a \
		interceptor.o scan.o pmap.o
	$(CXX) interceptor.o scan.o pmap.o \
		$(CXXFLAGS) -shared -o libinterceptor.so -ldl

clean:
	rm -f interceptor.o scan.o pmap.o
	rm -f libinterceptor.a libinterceptor.so 

.PHONY: clean all-nogpu all

