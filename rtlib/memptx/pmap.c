/*
 * pmap accelerator
 * 2017 Tong Zhang<ztong@vt.edu>
 */

#include "pmap.h"
#include "mpx.h"
#include "debugutil.h"
#include <sys/mman.h>

struct pmap* pmap = NULL;

void pmap_init()
{
    pmap = (struct pmap*)mmap(NULL,
                PMAP_SIZE,
                PROT_READ|PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pmap==MAP_FAILED)
    {
        fprintf(stderr,"pmap init failed! can not allocate memory!\n");
        exit(-1);
    }
}

void pmap_uninit()
{
    void* ptr = pmap;
    pmap = NULL;
    munmap(ptr, PMAP_SIZE);
}

void pmap_get(struct pmap* x)
{
    if (x==NULL)
    {
        fprintf(stderr,"pmap_get NULL?\n");
        return;
    }
    pmap->cnt = PMAP_MAX_ITEM;
    switch(syscall(_NR_SYS_GETPMAP, x))
    {
        case(PMAP_SUCCESS):
        {
            break;
        }
        case(PMAP_OVERFLOW):
        {
            WARN_ONCE(fprintf(stderr,"pmap overflow, consider enlarge PMAP_SIZE\n"););
            break;
        }
        default:
        {
            fprintf(stderr,"pmap syscall failed!\n");
            exit(-1);
            break;
        }
    }
}

void pmap_dump()
{
    fprintf(stderr,"=======PMAP======== @%p\n", pmap);
    pmap_get(pmap);
    fprintf(stderr,"%ld segments in total\n", pmap->cnt);
    for (int i=0;i<pmap->cnt;i++)
    {
        fprintf(stderr,"    0x%lx + %d pages\n",
                    pmap->segs[i].addr,
                    pmap->segs[i].length);
    }
}

/*
 * is this page an mpx bound table page?
 */
bool pmap_mpx_btp_exists(void* qaddr)
{
    /*
     * pmap segments are ordered
     */
    for (int i=0;i<pmap->cnt;i++)
    {
        void* addr = (void*)(pmap->segs[i].addr);
        size_t chunk_size = pmap->segs[i].length * PAGE_SIZE;
        void* addr_up = (void*)(pmap->segs[i].addr + chunk_size);
        if ((qaddr>=addr) && (qaddr<addr_up))
        {
            return true;
        }
        if (qaddr>addr_up)
        {
            return false;
        }
    }
    return false;
}

