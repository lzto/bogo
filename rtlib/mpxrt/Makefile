#This is Makefile for mpxrt
CFLAGS= -fPIC -g -Ofast -march=native
LDFLAGS=-Wl,--version-script=libmpx.map -lpthread

CC=/opt/toolchain/gcc/bin/gcc

all: dynamic static

dynamic:
	${CC} ${CFLAGS} mpxrt.c -c -fPIC -DPIC -o mpxrt_with_pic.o
	${CC} ${CFLAGS} mpxrt-utils.c -c -fPIC -DPIC -o mpxrt-utils_with_pic.o
	${CC} -shared -fPIC -DPIC ${LDFLAGS} \
		mpxrt_with_pic.o \
		mpxrt-utils_with_pic.o \
		-o libmpxrt.so

static:
	${CC} ${CFLAGS} -c mpxrt.c -o mpxrt_without_pic.o
	${CC} ${CFLAGS} -c mpxrt-utils.c -o mpxrt-utils_without_pic.o
	rm -f libmpxrt.a
	${AR} -rcs libmpxrt.a mpxrt_without_pic.o mpxrt-utils_without_pic.o
	ranlib libmpxrt.a


clean:
	rm -f mpxrt_with_pic.o mpxrt-utils_with_pic.o libmpxrt.so
	rm -f mpxrt_without_pic.o mpxrt-utils_without_pic.o libmpxrt.a

