CXX=/opt/toolchain/gcc/bin/g++
CC=/opt/toolchain/gcc/bin/gcc

GLOBAL_OPT_FLAG=-g -O0

MPX_CFLAG=-fcheck-pointer-bounds -mmpx
CFLAGS=${GLOBAL_OPT_FLAG} -I../ ${ADDITIONAL_CFLAGS}
LDFLAGS=${GLOBAL_OPT_FLAG} -Wl,-rpath=/opt/toolchain/gcc/lib64
LDFLAGS+=-ldl ../libinterceptor.so
#LDFLAGS+=../libmpxutils.a -ldl

all: test_c test_cpp

test_c:
	$(CXX) ${CFLAGS} ${MPX_CFLAG} -c test.c 
	$(CXX) test.o ${LDFLAGS} ${MPX_CFLAG} -o test

test_cpp:
	$(CXX) ${CFLAGS} ${MPX_CFLAG} -c test.cpp -o test_cpp.o
	$(CXX) test_cpp.o ${LDFLAGS} ${MPX_CFLAG} -o test_cpp


DOTSRC=$(wildcard *chkp.dot)

cfg: $(DOTSRC)
	@$(foreach dotfile, $(DOTSRC), dot $(dotfile) -Tpng -o $(dotfile).png;)

clean:
	rm -f test test[0-9] *.o test_cpp test_cpp.o
	rm -f test.c.[0-9]* test[0-9].c.[0-9]*
	rm -f *.png

.PHONY: clean all test test_cpp


