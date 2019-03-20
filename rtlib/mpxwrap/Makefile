#This is Makefile for mpxwrappers
CFLAGS=-Ofast -march=native \
	-fcheck-pointer-bounds -mmpx -fno-chkp-check-read \
	-fno-chkp-check-write -fno-chkp-use-wrappers -fPIC -I..
LDFLAGS=-Wl,--version-script=libmpxwrappers.map -fPIC

CC=gcc

all: static dynamic

dynamic:
	${CC} ${CFLAGS} -fPIC -DPIC -c mpx_wrappers.c -o mpx_wrappers_with_pic.o
	${CC} ${LDFLAGS} -shared mpx_wrappers_with_pic.o -o libmpxwrappers.so

static:
	${CC} ${CFLAGS} -c mpx_wrappers.c -o mpx_wrappers_without_pic.o
	rm -f libmpxwrappers.a
	${AR} -rcs libmpxwrappers.a mpx_wrappers_without_pic.o
	ranlib libmpxwrappers.a


clean:
	rm -f mpx_wrappers_with_pic.o libmpxwrappers.so
	rm -f mpx_wrappers_without_pic.o libmpxwrappers.a 

