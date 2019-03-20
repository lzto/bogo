#Visualization of bndldx/bndstx request

* Compile and Run Original Program

enable llmpx debug option by setting
```
-llmpx_dump_bndldstx=true
```

then run the program with
```
2>input.err
```

* preprocess intput file
```
grep bnd input.err > input.raw
make
./process_dbglog input.raw praw.csv finalbin
```

praw.csv is the input file for Processing 3 script (animate.pde)

finalbin stores the number of requests for each MPX page

* Open animate.pde using Processing 3 and hit run

