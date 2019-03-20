#Makefile for bound cache
TOOLCHAIN_PREFIX=/opt/toolchain/llvm/bin/
CC=$(TOOLCHAIN_PREFIX)clang
OPT=$(TOOLCHAIN_PREFIX)opt
LLVM_DIS=$(TOOLCHAIN_PREFIX)llvm-dis
CFLAGS=-Wall -g -O2 -emit-llvm
OPTFLAGS=-mem2reg \
		-simplifycfg \
		-loops \
		-lcssa \
		-loop-simplify \
		-loop-rotate \
		-loop-unroll \
		-unroll-count=3 \
		-unroll-allow-partial 

all:
	@echo compile bndcache
	$(CC) -c ${CFLAGS} bndcache.c -o bndcache.bc
	#@echo unroll loop optimization
	#$(OPT) bndcache.bc ${OPTFLAGS} -o bndcache.bc
	@echo disassamble bc file
	$(LLVM_DIS) bndcache.bc
	@echo bndcache.bc is ready
	@echo this file need to be linked together with main bc file

clean:
	rm -f bndcache.bc bndcache.ll

