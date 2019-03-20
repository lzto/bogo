
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# BOGO with LLVM-MPX

## Build
checkout and build mpx branch(https://github.com/lzto/llvm)
```bash
mkdir build
cd build
bash ../build.sh
cd ../rtlib/
make
```

## llvm-mpx Usage
```bash
opt -load=libllmpx.so input.bc -llmpx -o output.bc
```

## Runtime Library

static library recommended, dynamic library may not work properly.

### BOGO library
 * rtlib/memptx/libinterceptor.so
 * rtlib/memptx/libinterceptor.a

### llvm-mpx static
 * rtlib/mpxwrap/libmpxwrappers.a
 * rtlib/llmpxrt/libllmpx_rt.a
 * rtlib/mpxrt/libmpxrt.a

### llvm-mpx dynamic
 * rtlib/mpxwrap/libmpxwrappers.so
 * rtlib/llmpxrt/libllmpx_rt.so
 * rtlib/mpxrt/libmpxrt.so

