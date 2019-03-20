#Makefile for llmpx runtime library
CFLAGS=-fPIC -Wall -g -Ofast -march=native

all: static dynamic

dynamic:
	${CC} ${CFLAGS} -fPIC -DPIC -c llmpx_rt.c -o llmpx_rt_with_pic.o
	${CC} ${LDFLAGS} -shared llmpx_rt_with_pic.o -o libllmpx_rt.so

static:
	${CC} ${CFLAGS} -c llmpx_rt.c -o llmpx_rt_without_pic.o
	rm -f libllmpx_rt.a
	${AR} -rcs libllmpx_rt.a llmpx_rt_without_pic.o
	ranlib libllmpx_rt.a

clean:
	rm -f llmpx_rt_with_pic.o libllmpx_rt.so
	rm -f llmpx_rt_without_pic.o libllmpx_rt.a

