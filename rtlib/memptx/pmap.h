/*
 * accelrated pmap getter
 * 2016-2017 Tong Zhang<ztong@vt.edu>
 */
#ifndef _PMAP_
#define _PMAP_

#include "debugutil.h"
#include "mpx.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/syscall.h>

#define _NR_SYS_GETPMAP (399)
#define _NR_SYS_IS_PAGE_CLEAN (400)
#define _NR_SYS_MPROTECT_MPX (401)

struct pmap_segment
{
    unsigned long addr;
    unsigned short length;
};

struct pmap
{
    unsigned long cnt;
    struct pmap_segment segs[0];
}__attribute__((aligned(16)));

extern struct pmap* pmap;

#define PMAP_MAX_ITEM 16383
#define PMAP_SIZE (sizeof(unsigned long) + \
    sizeof(struct pmap_segment)*(PMAP_MAX_ITEM) )

enum {
    PMAP_SUCCESS,
    PMAP_OVERFLOW
};

void pmap_init();
void pmap_uninit();
void pmap_get(struct pmap* x);
void pmap_dump();
bool pmap_mpx_btp_exists(void*);

inline bool is_page_clean(void *x)
{
    return syscall(_NR_SYS_IS_PAGE_CLEAN, x)==1;
}

inline int mprotect_mpx(void* addr, int prot)
{
    return syscall(_NR_SYS_MPROTECT_MPX, addr, prot);
}

#endif //_PMAP_

