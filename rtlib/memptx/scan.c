/*
 * MPX BD&BT SCANNER
 * DO NOT COMPILE THIS FILE WITH MPX !!!
 * 2016-2017 Tong Zhang <ztong@vt.edu>
 */
#include "mpx.h"
#include "pmap.h"
#include "debugutil.h"
#include "scan.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <errno.h>
/*
 * for mincore
 */
#include <unistd.h>
#include <sys/mman.h>

#include <signal.h>

//AVX speedup
#include <immintrin.h>

#include <pthread.h>

///////////////////////////////////////////////////////////////////////////////

struct free_queue free_queue =
{
    .items = {0},
    .water_level = 0
};

size_t bfq_allowed_max = BFQ_ALLOWED_INITIAL;
////////////////////////////////////////////////////////////////////////////////
/*
 * hp_node maintains hot pages touched by bndldx and bndstx
 */
//struct hp_node hp_nodes[HOT_PAGES_MAX];
struct hp_node *hp_nodes = NULL;
int hpn_head = -1;//have small pf_cnt
int hpn_tail = -1;//have large pf_cnt
int free_hp_node = -1;//head of free list
int hpn_free_cnt = HOT_PAGES_MAX-2;
int hpn_free_quota = HOT_PAGES_QUOTA_INITIAL;
/*
 * ------------------------------
 * |head |tail |free_hp_node|...
 * ------------------------------
 * |0    |1    |2           |...
 * ------------------------------
 */

/*
 * DEBUG FUNCTIONS
 */
/*
 * how many entries is used in this bt page?
 */
int mpx_bt_page_utilization(struct mpx_bt_entry* bt)
{
    unsigned used_entry = 0;
    for (unsigned i=0; i<(PAGE_SIZE/(sizeof(struct mpx_bt_entry))); ++i)
    {
        if (bt[i].p!=NULL)
        {
            used_entry++;
            //fprintf(stderr, "%p-%p-%p\n", bt[i].lb, (void*)~(unsigned long)bt[i].ub, bt[i].p);
        }
    }
    return used_entry;
}

/*
 * validate hot page list
 */
void hpn_checker()
{
    assert(hp_nodes[hpn_head].next!=-1);
    assert(hp_nodes[hpn_tail].prev!=-1);
    assert(hp_nodes[hpn_head].prev==(unsigned short)(-1));
    assert(hp_nodes[hpn_tail].next==(unsigned short)(-1));
    unsigned short hpn_cur = hp_nodes[hpn_head].next;
    fprintf(stderr,KRED "--hpn_checker--\n" KNRM);
    fprintf(stderr,"hp_nodes[hpn_head].next=%d\n", hp_nodes[hpn_head].next);
    fprintf(stderr,"hp_nodes[hpn_tail].prev=%d\n", hp_nodes[hpn_tail].prev);
    while(hpn_cur!=hpn_tail)
    {
        unsigned short hpn_next = hp_nodes[hpn_cur].next;

        fprintf(stderr,"(%d).next=%d (%d).prev=%d\n",
            hpn_cur,
            hp_nodes[hpn_cur].next,
            hpn_next,
            hp_nodes[hpn_next].prev);

        assert(hp_nodes[hpn_cur].next == hpn_next);
        assert(hp_nodes[hpn_next].prev = hpn_cur);
        hpn_cur = hpn_next;
    }
}

void dump_hpn()
{
    unsigned short hpn_cur = hp_nodes[hpn_head].next;
    fprintf(stderr,"----hpn----\n");
    while(hpn_cur!=hpn_tail)
    {
        fprintf(stderr,"%d:0x%lx, pf_cnt=%d used:%d\n",
            hpn_cur,
            hp_nodes[hpn_cur].page_addr,
            hp_nodes[hpn_cur].pf_cnt,
            mpx_bt_page_utilization((struct mpx_bt_entry*)hp_nodes[hpn_cur].page_addr));
        hpn_cur = hp_nodes[hpn_cur].next;
    }
}
/*
 * find the hot page list node using given page address
 */
unsigned short hpn_find_page(void* page_addr)
{
    unsigned short hpn_cur = hp_nodes[hpn_head].next;
    while (hpn_cur != hpn_tail)
    {
        if (hp_nodes[hpn_cur].page_addr==(unsigned long)page_addr)
        {
            return hpn_cur;
        }
        hpn_cur = hp_nodes[hpn_cur].next;
    }
    return hpn_cur;
}
/*
 * get page permission from hot page list, given page address
 * return PROT_NONE is not found
 */
unsigned short hpn_get_page_permission(void *page_addr)
{
    unsigned short hpn_cur = hp_nodes[hpn_head].next;
    while (hpn_cur != hpn_tail)
    {
        if (hp_nodes[hpn_cur].page_addr==(unsigned long)page_addr)
        {
            return hp_nodes[hpn_cur].permission;
        }
        hpn_cur = hp_nodes[hpn_cur].next;
    }
    return PROT_NONE;
}

/*
 * init hp_nodes
 */
void hpn_init()
{
    hp_nodes = (struct hp_node*)mmap(NULL,
                sizeof(struct hp_node) * HOT_PAGES_MAX,
                PROT_READ|PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (hp_nodes==MAP_FAILED)
    {
        fprintf(stderr,"hp_nodes init failed! can not allocate memory!\n");
        exit(-1);
    }
    if (((unsigned long)hp_nodes) & 0x0FFF != 0)
    {
        fprintf(stderr,"hp_nodes not aligned to page boundray???\n");
        exit(-1);
    }

    /*
     * head and tail
     */
    hpn_head = 0;
    hpn_tail = 1;
    hp_nodes[hpn_head].page_addr = ~0UL;
    hp_nodes[hpn_head].pf_cnt = -1;
    hp_nodes[hpn_head].next = hpn_tail;
    hp_nodes[hpn_head].prev = -1;

    hp_nodes[hpn_tail].page_addr = ~0UL;
    hp_nodes[hpn_tail].pf_cnt = -1;
    hp_nodes[hpn_tail].next = -1;
    hp_nodes[hpn_tail].prev = hpn_head;

    /*
     * free node list
     * prev filed is not maintained for free node list
     */
    for (int i=2;i<HOT_PAGES_MAX-1;i++)
    {
        hp_nodes[i].next = i+1;
    }
    //head of free node list
    free_hp_node = 2;
    hp_nodes[HOT_PAGES_MAX-1].next = -1;
}

void hpn_uninit()
{
    munmap(hp_nodes, sizeof(struct hp_node) * HOT_PAGES_MAX);
    hp_nodes = NULL;
}

/*
 * get new node from hp_nodes
 */
unsigned short hpn_alloc()
{
    int used = HOT_PAGES_MAX - hpn_free_cnt;
    if (used>=hpn_free_quota)
    {
        return -1;
    }

    unsigned short ret = free_hp_node;
    if (ret!=HPN_NULL)
    {
        hpn_free_cnt--;
        free_hp_node = hp_nodes[free_hp_node].next;
    }
    return ret;
}
/*
 * put back the node to the free list
 */
void hpn_free(unsigned short n)
{
    hp_nodes[n].next = free_hp_node;
    free_hp_node = n;
    hpn_free_cnt++;
}
/*
 * insert a node into the hot page list ordered by its pf_cnt
 */
void hpn_insert_ordered(unsigned short node)
{
#if USE_PFCNT
    unsigned short npf_cnt = hp_nodes[node].pf_cnt;
    unsigned short cur_node = hpn_head;
    unsigned short nex_node;
    //fprintf(stderr,"hpn_insert_ordered %d:0x%lx - (%d):\n",
    //            node, hp_nodes[node].page_addr, npf_cnt);
    while (1)
    {
        nex_node = hp_nodes[cur_node].next;
        //fprintf(stderr,"   %d->%d\n",cur_node, nex_node);
        unsigned short cur_pfcnt = hp_nodes[cur_node].pf_cnt;
        unsigned short nex_pfcnt = hp_nodes[nex_node].pf_cnt;
        //fprintf(stderr,"         (%d)->(%d)\n", cur_pfcnt, nex_pfcnt);
        if (npf_cnt<nex_pfcnt)
        {
            hp_nodes[cur_node].next = node;
            hp_nodes[node].next = nex_node;
            break;
        }
        cur_node = nex_node;
    }
#else
    //FIFO, insert before hpn_tail
    unsigned short prev = hp_nodes[hpn_tail].prev;
    unsigned short next = hpn_tail;
    hp_nodes[node].prev = prev;
    hp_nodes[node].next = next;
    
    hp_nodes[prev].next = node;
    hp_nodes[next].prev = node;
#endif
}
/*
 * evict a node based on pf_cnt, return the first element in the list
 * assuming that the list is not empty
 */
unsigned short hpn_evict()
{
    unsigned short ret = hp_nodes[hpn_head].next;
    unsigned short new_first = hp_nodes[ret].next;
    hp_nodes[hpn_head].next = new_first;
    hp_nodes[new_first].prev = hpn_head;
    return ret;
}
/*
 * set quota
 */
void hpn_set_quota(int new_quota)
{
    if (new_quota>=hpn_free_quota)
    {
        /*
         * growing quota, only affect free node list
         */
        hpn_free_quota = new_quota;
        if (new_quota>HOT_PAGES_MAX)
            hpn_free_quota = HOT_PAGES_MAX;
        return;
    }
    //new_quota<hpn_free_quota
    //fprintf(stderr,"hpn_set_quota %d\n", new_quota);
    /*
     * we are shrinking quota
     * evict items from used item list if they exceed quota,
     * and add them to the free node list
     */
    //int used = (HOT_PAGES_MAX-2)-hpn_free_cnt;
    int used = hpn_free_quota - hpn_free_cnt;
    while (used>new_quota)
    {
        //evict and free
        unsigned evt = hpn_evict();
        struct mpx_bt_page* victim_bt_page
            = (struct mpx_bt_page*)hp_nodes[evt].page_addr;
        //fprintf(stderr,"...kick %d:%p permission: %d\n", evt,
        //                victim_bt_page, hp_nodes[evt].permission);
        /*
         * we need to start scan against free_queue from wl_kick
         */
        victim_bt_page->wl_kick = free_queue.water_level;
        //mprotect(victim_bt_page, PAGE_SIZE, PROT_NONE);
        mprotect_mpx(victim_bt_page, PROT_NONE);
        hpn_free(evt);
        used--;
    }
    hpn_free_quota = new_quota;
}

inline int hpn_used()
{
    int used = hpn_free_quota - hpn_free_cnt;
    return used;
}

//////////////////////////////////////////////////////////////////////////////
/*
 * memory layout
 * -------------- low address
 * | main image |
 * --------------
 * | heap       |
 * --------------
 * | mpx BD & BT|
 * --------------
 * | library    |
 * --------------
 * | stack      |
 * --------------
 * | vvar       |
 * --------------
 * | vdso       |
 * --------------
 * | vsyscall   |
 * -------------- high address
 */

///////////////////////////////////////////////////////////////////////////////
/*
 * scan bound table
 * ----------------
 *  bt - pointer to the starting point of bound table
 *  bound_table_entry_count - how many bt entries are there
 *  p - the pointer to be found out - if not using bulk free
 */
inline void scan_bt_and_invalid_single(struct mpx_bt_entry* bt,
		unsigned bound_table_entry_count, void* p)
{
	unsigned long ptr = (unsigned long)p;
#if 1
    #if 0
    __m128i invbnd;
    ((unsigned long*)&invbnd)[0] = 0;
    ((unsigned long*)&invbnd)[1] = ~0UL;
    #endif
	for (unsigned i=0; i<bound_table_entry_count; ++i)
	{
        //__builtin_prefetch(&bt[i+1], 0, 0);
        unsigned long lb = (unsigned long)bt[i].lb;
        unsigned long ub = ~(unsigned long)bt[i].ub;
		/*
		 * we don't want to do this for init bnd
		 */
        if (lb==0)
            continue;
        /*
         * NOTE: this sucks as it will invalidate any bound that have the ptr in
         * it. The correct way to do it is to maintain the malloc'ed memory range
         * bnd0=[ptr,ptr+size] and do intersection of these two pieces of bound
         * to see if the bound falls into bnd0 completely.
         * the quick fix -- avoid the invalidate the infinite bound is shown 
         * above, i.e. to check the lb to see whether this is an infinite bound
         * or not, but this does not work for other cases
         */
        if ((ptr<lb)||((ptr>ub)))
        {
            continue;
        }
        #if 1
        bt[i].ub = (void*)~0UL;
		bt[i].lb = 0;
        #else
        _mm_store_si128((__m128i*)&bt[i],invbnd);
        #endif
	}
#else
    //Is cmov faster ?
    //lb, ub, ptr_val, res 4*8=32 Bytes
    unsigned long invbnd_lb = 0;
    unsigned long invbnd_ub = ~(0UL);
    for (unsigned i=0; i<bound_table_entry_count; ++i)
    {

        unsigned long lb = (unsigned long)bt[i].lb;
        unsigned long ub = ~(unsigned long)bt[i].ub;
        int predicate_condition = (lb!=0) && (ptr>=lb) & (ptr<ub);
        //int predicate_condition =!((lb==0) || (ptr<lb) || (ptr>ub));
        asm("mov %3, %%r14\n\t"
            "mov %4, %%r15\n\t"
            "test %2, %2\n\t"
            "cmovnz %5, %%r14\n\t"
            "cmovnz %6, %%r15\n\t"
            "mov %%r14, %0\n\t"
            "mov %%r15, %1\n\t"
            : "=m"(bt[i].lb), "=m"(bt[i].ub)
            : "r"(predicate_condition),
                "m"(bt[i].lb), "m"(bt[i].ub),
                "r"(invbnd_lb), "r"(invbnd_ub)
            : "cc", "r14", "r15");
    }
#endif
}
/*
 * only scan from free_queue(wl_kick, water_level)
 */
inline bool search_free_queue(unsigned long lb,
        unsigned long ub,
        unsigned short wl_kick)
{
    unsigned free_queue_wl = free_queue.water_level;
    for (unsigned short i=wl_kick;i<free_queue_wl;i++)
    {
        unsigned long the_fptr = free_queue.items[i];

        if ((the_fptr>=lb) && (the_fptr<ub))
        {
            return true;
        }
    }
    return false;
}

void scan_bt_and_invalid_bulk_way(struct mpx_bt_entry* bt,
		unsigned bound_table_entry_count)
{
    struct mpx_bt_page * bt_page = (struct mpx_bt_page*)bt;
    unsigned short wl_kick = bt_page->wl_kick;

	for (unsigned i=0; i<bound_table_entry_count; i++)
	{
		unsigned long lb = (unsigned long)bt[i].lb;
		unsigned long ub = ~(unsigned long)bt[i].ub;
		/*
		 * we don't want to do this for init bnd
		 */
		if (lb==0)
			continue;
        if(!search_free_queue(lb,ub,wl_kick))
        {
            continue;
        }
        __builtin_prefetch(&bt[i+1], 0, 1);
		bt[i].ub = (void*)~0UL;
		bt[i].lb = 0UL;
        __builtin_ia32_clflush(&bt[i]);
	}
}

/*
 * scan all mpx bound table pages, return the number of pages scanned
 *
 * this is very slow/expensive since there could be lots of used mpx bt pages!
 * --------------------------------------------------------------
 * Optimization: only access MPX pages that is dirty(at least bndstx'd)
 * Kernel takes care of bound table population,
 * we only need to scan the bound table and don't care about bound directory
 * so that we can save some time scanning bound directory and querying 
 * page avaiability
 * FIXME: only need to scan cold mpx bt pages!!!
 */
size_t scan_all_bps_and_invalid()
{
    pmap_get(pmap);
    size_t pages_scanned = 0;
	///////////////////////////////////////////////////////////////////////////
    //scan collected mpx pages
    for (int i=0;i<pmap->cnt;i++)
    {
        void* addr = (void*)(pmap->segs[i].addr);
        size_t chunk_size = pmap->segs[i].length * PAGE_SIZE;

        struct mpx_bt_page * bt_page = (struct mpx_bt_page*)addr;
        /*
         * populate page to avoid page fault
         */
        mprotect(&bt_page[0], pmap->segs[i].length * PAGE_SIZE,
            PROT_READ|PROT_WRITE|PROT_POPULATE);
        for (int j=0;j<pmap->segs[i].length;j++)
        {
            if (bt_page[j].wl_kick==-1)
            {
                //dont care hot pages
                continue;
            }
            scan_bt_and_invalid_bulk_way(
                (struct mpx_bt_entry*)(&bt_page[j]),
                PAGE_SIZE / (unsigned int)sizeof(struct mpx_bt_entry));
            //__builtin_prefetch(&bt_page[j+2], 0, 0);
            pages_scanned++;
            bt_page[j].wl_kick = 0;
        }
        __builtin_prefetch(&pmap->segs[i+2],0,0);
        mprotect(&bt_page[0], pmap->segs[i].length * PAGE_SIZE, PROT_NONE);
    }
	///////////////////////////////////////////////////////////////////////////
    return pages_scanned;
}
/*
 * fast path, scan hot bt pages
 */
#if USE_THREAD

static unsigned short hot_bps_cur_node;
static pthread_cond_t worker_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t worker_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t worker_thread[WORKERS_N];

struct worker_arg 
{
    volatile unsigned int start;
};

static struct worker_arg workers_arg[WORKERS_N];
static volatile int running_workers;

static void* stub_ptr;

inline void scan_hot_bps_and_invalid_single(void* ptr)
{
    
    for (unsigned short node = hp_nodes[hpn_head].next;
            node!=hpn_tail; node = hp_nodes[node].next)
    {
        void* addr = (void*)hp_nodes[node].page_addr;
        scan_bt_and_invalid_single((struct mpx_bt_entry*)addr,
                PAGE_SIZE/(unsigned int)sizeof(struct mpx_bt_entry), ptr);
    }
}

inline void scan_hpn_stub()
{
    while(hot_bps_cur_node != hpn_tail)
    {
        unsigned short old_hot_bps_cur_node = hot_bps_cur_node;
        unsigned short node = hp_nodes[old_hot_bps_cur_node].next;
        if (!__sync_bool_compare_and_swap(&hot_bps_cur_node,
                        old_hot_bps_cur_node,
                        node))
        {
            continue;
        }
        if (node==hpn_tail)
        {
            break;
        }
        void* addr = (void*)hp_nodes[node].page_addr;
        scan_bt_and_invalid_single((struct mpx_bt_entry*)addr,
                PAGE_SIZE/(unsigned int)sizeof(struct mpx_bt_entry), stub_ptr);
    }
}

size_t scan_hot_bps_and_invalid(void* ptr)
{
    int used = hpn_used();
#if 0
    stub_ptr = ptr;
    hot_bps_cur_node = hpn_head;
    if (used>=16)
    {
        //pthread_mutex_lock(&worker_lock);
        running_workers++;
        //signal worker thread to start scan
        //pthread_cond_signal(&worker_cond);
        //pthread_cond_broadcast(&worker_cond);
        //pthread_mutex_unlock(&worker_lock);
        scan_hpn_stub();
        //wait worker thread to stop
        while(!__sync_bool_compare_and_swap(&running_workers, 0, 0)){};
        //while (running_workers!=0){};
    }else
    {
        scan_hot_bps_and_invalid_single(ptr);
    }
#else
    stub_ptr = ptr;
    for (int i=0;i<WORKERS_N;i++)
    {
        workers_arg[i].start = 1;
        //__sync_bool_compare_and_swap(&workers_arg[i].start,0,1);
    }
    __sync_bool_compare_and_swap(&running_workers,0,WORKERS_N);
    //while(!__sync_bool_compare_and_swap(&running_workers, 0, 0)){};
    while(running_workers!=0){}
#endif
    return used;
}

void* worker_func(void* v)
{
    struct worker_arg *arg = (struct worker_arg*)v;
    while(1)
    {
        /*pthread_mutex_lock(&worker_lock);
        if (running_workers==0)
        {
            pthread_cond_wait(&worker_cond, &worker_lock);
        }*/
        while(arg->start==0){}
        while(running_workers==0){}
        //while(__sync_bool_compare_and_swap(&arg->start, 0, 0)){};
        //while(__sync_bool_compare_and_swap(&running_workers, 0, 0)){};
        scan_hpn_stub();
        //__sync_bool_compare_and_swap(&(arg->start),1,0);
        arg->start = 0;
        __sync_fetch_and_sub(&running_workers,1);
        //pthread_mutex_unlock(&worker_lock);
    }
    return NULL;
}
#else
size_t scan_hot_bps_and_invalid(void* ptr)
{
    size_t pages_scanned = 0;
    
    for (unsigned short node = hp_nodes[hpn_head].next;
            node!=hpn_tail; node = hp_nodes[node].next)
    {
        void* addr = (void*)hp_nodes[node].page_addr;
        scan_bt_and_invalid_single((struct mpx_bt_entry*)addr,
                PAGE_SIZE/(unsigned int)sizeof(struct mpx_bt_entry), ptr);
        pages_scanned++;
    }
    return pages_scanned;
}

#endif
/*
 * scan single mpx bt page against the free_queue
 * we need to check the wl_kick, if it is smaller than free_queue.water_level
 * then we need to scan
 */
void scan_single_bt_page_and_invalid(void* bte)
{
    struct mpx_bt_page * bt_page = (struct mpx_bt_page*) bte;
    //dont scan if no free in between
    //but how come water_level is smaller?!!!
    if (free_queue.water_level<=bt_page->wl_kick)
    {
        return;
    }
    scan_bt_and_invalid_bulk_way((struct mpx_bt_entry*)bte,
        PAGE_SIZE/(unsigned int)sizeof(struct mpx_bt_entry));
}

void scan_init()
{
    hpn_init();
    pmap_init();
    #if USE_THREAD
    running_workers = 0;
    //create worker thread
    for (int i=0;i<WORKERS_N;++i)
    {
        pthread_create(&worker_thread[i], NULL, worker_func, &(workers_arg[i]));
    }
    #endif
    return;
}

void scan_uninit()
{
    hpn_uninit();
    pmap_uninit();
    return;
}

