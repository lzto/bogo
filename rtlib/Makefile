#Makefile for llmpx runtime library

all: libllmpxrt libmpxrt libmpxwrap libbndcache libmemptx

libllmpxrt:
	$(MAKE) -C llmpxrt

libmpxrt:
	$(MAKE) -C mpxrt

libmpxwrap:
	$(MAKE) -C mpxwrap

libbndcache:
	$(MAKE) -C bndcache

libmemptx:
	$(MAKE) -C memptx

clean:
	$(MAKE) clean -C llmpxrt
	$(MAKE) clean -C mpxrt
	$(MAKE) clean -C mpxwrap
	$(MAKE) clean -C bndcache
	$(MAKE) clean -C memptx
