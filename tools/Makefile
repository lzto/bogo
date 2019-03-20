CFLAGS=-std=c++11 -Ofast -march=native

all: process_dbglog

process_dbglog: process_dbglog.o
	$(CXX) $(LDFLAGS) $(CFLAGS) $< -o $@

process_dbglog.o: process_dbglog.cpp
	$(CXX) -c $(CFLAGS) $<

clean:
	rm -f process_dbglog.o process_dbglog

