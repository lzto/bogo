/*
 * interceptor test library
 * 2015-2017 Tong Zhang<ztong@vt.edu>
 */

#include "mpx.h"
#include "scan.h"
#include "debugutil.h"
#include "pmap.h"

#include <dlfcn.h>//for dlsym

#include <stdio.h>
#include <stdint.h>

#include <assert.h>

//For gettid()
#include <sys/syscall.h>
//for mprotect
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>
#include <string.h>
#include <execinfo.h>

#define _INSTRUMENTOR_CPU_SIZE (8)

int __instrumentor_cpu = 0;
static bool memptx_initialized = false;

extern struct free_queue free_queue;

#if (DEBUG>10)
#define RECORD_SYNCOP(func) \
	fprintf(stderr,\
		"TID:%d TSC:%llu caught " #func ".\n",\
		gettid(),\
		rdtsc());\

#define RECORD_SYNCOP_WARG(func,p) \
	fprintf(stderr,\
		"TID:%d TSC:%llu caught " #func ": %p\n",\
		gettid(),\
		rdtsc(),\
		(void*)p);

#define RECORD_SYNCOP_WARG2(func,p,s) \
	fprintf(stderr,\
		"TID:%d TSC:%llu caught " #func ": %p 0x%x\n",\
		gettid(),\
		rdtsc(),\
		(void*)p,\
		(int)s);

#elif DUMP_MALLOC_AND_FREE

#define RECORD_SYNCOP(func) \
	fprintf(stderr,\
		"%llu " #func ".\n",\
		rdtsc());\

#define RECORD_SYNCOP_WARG(func,p) \
	fprintf(stderr,\
		"%llu " #func ": %p\n",\
		rdtsc(),\
		(void*)p);

#define RECORD_SYNCOP_WARG2(func,p,s) \
	fprintf(stderr,\
		"%llu " #func ": %p 0x%x\n",\
		rdtsc(),\
		(void*)p,\
		(int)s);

#else
#define RECORD_SYNCOP(func)
#define RECORD_SYNCOP_WARG(func,p)
#define RECORD_SYNCOP_WARG2(func,p,s)
#endif

namespace std {
struct nothrow_t {};
}  // namespace std

/*
 * buffer certain number of malloc since last free_queue flush or hot page evict
 * evict the oldest one when full, this only affect performance, and does not
 * affect correctness
 * cost measurement
 */
#if MALLOC_LOCALITY
struct bucket
{
    void* addrs[BUCKET_SIZE];
    unsigned victim;//next victim
    unsigned cnt;//how many items are there in the bucket
}__attribute__((aligned(64)));
static struct bucket last_malloc[MAL_SIZE];

void malc_reset()
{
    for(int i=0;i<BUCKET_SIZE;i++)
    {
        last_malloc[i].victim = 0;
        last_malloc[i].cnt = 0;
    }
}

void malc_insert(void* addr)
{
    unsigned lm_bucket_position
        = ((((unsigned long)addr)>>8) ^ ((unsigned long)addr))
                    % (MAL_SIZE);

    struct bucket *bucket = &last_malloc[lm_bucket_position];
    //last bucket element is the victim to be selected next time
    unsigned int victim = bucket->victim;
    bucket->addrs[victim] = addr;
    victim = (victim+1)%BUCKET_SIZE;
    bucket->victim = victim;
    if (bucket->cnt<BUCKET_SIZE)
    {
        bucket->cnt++;
    }
}

bool malc_exists(void* addr)
{
    bool ret = false;
    unsigned lm_bucket_position
        = ((((unsigned long)addr)>>8) ^ ((unsigned long)addr))
                    % (MAL_SIZE);
    unsigned victim = 0;
    struct bucket * bucket = &last_malloc[lm_bucket_position];
    for (int i=0; i<bucket->cnt; i++)
    {
        if (bucket->addrs[i]==addr)
        {
            bucket->addrs[i] = NULL;
            ret = true;
            if (bucket->cnt==BUCKET_SIZE)
                victim = i;
            break;
        }else if (bucket->addrs[i]==NULL)
        {
            if (bucket->cnt==BUCKET_SIZE)
                victim = i;
        }
    }
    bucket->victim = victim;
    return ret;
}

#else
void malc_reset(){};
void malc_insert(void* addr){};
bool malc_exists(void* addr)
{
    return false;
};
#endif

static inline void do_backtrace()
{
    fprintf(stderr,"\n====BACKTRACE====\n");
    int nptrs;
    void *buffer[100];
    nptrs = backtrace(buffer, 100);
    backtrace_symbols_fd(buffer,nptrs,STDERR_FILENO);
}

/*
 * cost measurement
 */
//accumulated page fault cost, slow path cost and fast path cost
static size_t acct_pf = 0;
static size_t acct_sp = 0;
static size_t acct_fp = 0;

//how many free we encountered
static size_t stat_free = 0;
static size_t stat_fast_free = 0;
static size_t stat_slow_free = 0;
//how many fp we encountered
static size_t stat_pf = 0;

//time spent in each of these handler in this epoch
//slow path
static size_t t_sf = 0;
//fast path
static size_t t_ff = 0;
//page fault
static size_t t_pf = 0;
//accumulated time
static size_t acc_t_sf = 0;
static size_t acc_t_ff = 0;
static size_t acc_t_pf = 0;

/*
 * need to serialize requests
 */
static bool call_me_when_ready = false;
static bool call_refresh_pmap_when_ready = false;
static bool in_handler = false;

////////////////////////////////////////////////////////////////////////////////
//functions to intercept

#define INTERCEPTOR_TYPE __attribute__((weak))

/*
 * for memory hooks
 */
typedef void* (*real_malloc)(size_t size);
typedef void* (*real_calloc)(size_t nmemb, size_t size);
typedef void* (*real_realloc)(void *ptr, size_t size);
typedef void (*real_free)(void *ptr);

/*
 * pre-initialize these function pointer
 */
static real_free free_fptr = NULL;
static real_malloc malloc_fptr = NULL;
static real_calloc calloc_fptr = NULL;

////////////////////////////////////////////////////////////////////////////////
//
//Helper function
//

inline void* GetRealFunctionAddress(const char * func_name)
{
	return (void*)dlsym(RTLD_NEXT, func_name);
}

inline void* GetRealVersionedFunctionAddress(const char* func_name, const char* ver)
{
	return (void*)dlvsym(RTLD_NEXT, func_name, ver);
}

inline void* GetRealFunctionAddressFromLibc(const char * func_name)
{
	static void* handle = NULL;
	if (handle==NULL)
	{
		handle = dlopen("libc.so.6",RTLD_LAZY);
		exit(-1);
	}
	// NOTE: libc.so.6 may *not* exist on Alpha and IA-64 architectures.
	return (void*)dlsym(handle, func_name);
}

static __inline__ int gettid()
{
	return syscall(SYS_gettid);
}

void dump_configuration()
{
#if MEMPTX_DUMP_CFG
    fprintf(stderr,
        "-----MEMPTX CFG--------\n");
#endif
}

void dump_stat()
{
#if MEMPTX_STAT
	fprintf(stderr,
		"-----MEMPTX STAT-------\n"
		"free %lu(fast %lu slow %lu)\n"
        "page fault %lu\n",
		stat_free, stat_fast_free, stat_slow_free,
        stat_pf
		);
    fprintf(stderr,
        "time consumption:\n"
        "t fast free:%lu\n"
        "t slow free:%lu\n"
        "t page fault:%lu\n",
        acc_t_ff, acc_t_sf, acc_t_pf); 
#endif//MEMPTX_STAT
}

static void dump_internal_state()
{
    fprintf(stderr,"\n======================\n");
    do_backtrace();
    fprintf(stderr,"\n======================\n");

    fprintf(stderr,"    === memptx state ===\n");
    extern bd_type bd;
    fprintf(stderr,"MPX bound directory: %p-%p\n",
                bd, ((uint8_t*)bd)+MPX_L1_SIZE);
    fprintf(stderr,"MPX bound table pages:...\n");
    pmap_dump();
    fprintf(stderr,"hot pages:\n");
    unsigned hpn_used = 0;
    for (unsigned short node = hp_nodes[hpn_head].next;
            node!=hpn_tail; node = hp_nodes[node].next)
    {
        hpn_used++;
        void* addr = (void*)hp_nodes[node].page_addr;
        fprintf(stderr,"\t%d:%p\n", node, addr);
    }
    fprintf(stderr,"hpn_quota:%d, used:%d\n", hpn_free_quota, hpn_used);
    fprintf(stderr,"free queue:");
    for (int i=0;i<free_queue.water_level;i++)
    {
        fprintf(stderr,"%p ",(void*)free_queue.items[i]);
    }
}

/*
 * catch ^C, dump stat and exit
 */
void signal_handler(int signo)
{
    switch(signo)
    {
        case (SIGUSR1):
        {
            dump_stat();
            break;
        }
        case (SIGINT):
        {
            dump_configuration();
            dump_stat();
            dump_internal_state();
            break;
        }
        case (SIGTRAP):
        {
            /*
             * we could possibly get here when destructor is called
             * prevent potential problem from happening
             */
            if (unlikely(memptx_initialized==false))
            {
                break;
            }
            /*
             * FIXME! asynchronize signal could be delivered
             * when dealing with page fault (SIGSEGV)
             * which could cause problem. see 403.gcc
             */
            if (in_handler)
            {
                call_refresh_pmap_when_ready = true;
                break;
            }
            call_refresh_pmap_when_ready = false;
            /*
             * mpx pages have been free'd
             * do pmap_get() and sanitize H here
             */
            pmap_get(pmap);
            unsigned short prev_node = hpn_head;
            for (unsigned short node = hp_nodes[hpn_head].next;
                    node!=hpn_tail; node = hp_nodes[node].next)
            {
                void* addr = (void*)hp_nodes[node].page_addr;
                bool is_mpx_page = pmap_mpx_btp_exists(addr);
                if (!is_mpx_page)
                {
                    //need to remove
                    //maintain reference to next node
                    unsigned short next_node = hp_nodes[node].next;
                    hp_nodes[prev_node].next = next_node;
                    hp_nodes[next_node].prev = prev_node;
                    //free current node
                    hpn_free(node);
                    //move indice back one
                    node = prev_node;
                }
                prev_node = node;
            }
            break;
        }
#if DYNAMIC_ADJUST_HQ
        case (SIGALRM):
        {
            if (in_handler)
            {
                call_me_when_ready = true;
                return;
            }
            call_me_when_ready = false;
            #if 1
            //evaluate collected stastics here and do adjustment
            float cost_pf = (float)(t_pf)/4000000.0;
            float cost_ff = (float)(t_ff)/4000000.0;
            float cost_total = cost_ff + cost_pf;

static size_t last_stat_fast_free = 0;
static size_t last_stat_slow_free = 0;
static size_t last_stat_pf = 0;
#if 0
                fprintf(stderr,
                        "pf(%.2f) + ff(%.2f) = %f sec in 100ms,"
                        " #calls pf(%lu) ff(%lu), "
                        " #cost per call:pf(%lu) ff(%lu)\n",
                        cost_pf,
                        cost_ff,
                        cost_total,
                        (stat_pf-last_stat_pf),
                        (stat_fast_free-last_stat_fast_free),
                        (unsigned long)(1000000 * cost_pf 
                                            / (stat_pf-last_stat_pf)),
                        (unsigned long)(1000000 * cost_ff 
                                            / (stat_fast_free-last_stat_fast_free)));
            //we have 0.1s duration
            if (cost_total<20)
            {
                goto reset;
            }

#endif

            #endif
            if (t_ff>t_pf)
            //if (acct_fp > acct_pf)
            {
                if (hpn_free_quota>MIN_HPN_FREE_QUOTA)
                {
                    unsigned short old_hpn_free_quota = hpn_free_quota;
                    //unsigned short hpn_adjustment = hpn_free_quota / 2;
                    unsigned short hpn_adjustment = hpn_free_quota / 4;
                    if (hpn_adjustment < MIN_HPN_FREE_QUOTA)
                    {
                        hpn_adjustment = MIN_HPN_FREE_QUOTA;
                    }
                    hpn_set_quota(hpn_adjustment);
                    if (old_hpn_free_quota!=hpn_free_quota)
                    {
                        malc_reset();
                    }
                    //fprintf(stderr,
                    //    KBLU "- hot page list size :%d\n" KNRM,
                    //    hpn_free_quota);
                }
            }else if (t_ff<t_pf)
            //}else if (acct_fp < acct_pf)
            {
                //pf introduce lots of sys time!!! 
                if (hpn_free_quota<HOT_PAGES_MAX)
                {
                    unsigned short old_hpn_free_quota = hpn_free_quota;
                    unsigned short hpn_adjustment = hpn_free_quota + 4;
                    //unsigned short hpn_adjustment = hpn_free_quota*2;
                    
                    hpn_set_quota(hpn_adjustment);
                    /*
                     * since we are growing free quota, malc should be valid
                     */
                    //if (old_hpn_free_quota!=hpn_free_quota)
                    //{
                    //    malc_reset();
                    //}
                }
                //fprintf(stderr,
                //    KRED "+ hot page list size :%d\n" KNRM,
                //    hpn_free_quota);
            }
reset:
            //fprintf(stderr,"hpn_free_quota=%d\n", hpn_free_quota);
            t_pf = 0;
            t_ff = 0;
            t_sf = 0;

            acct_pf = 0;
            acct_sp = 0;
            acct_fp = 0;

            last_stat_fast_free = stat_fast_free;
            last_stat_slow_free = stat_slow_free;
            last_stat_pf = stat_pf;


            //0.1s
            ualarm(100000,0);
            break;
        }
#endif
        default:
        ;
    }
}
////////////////////////////////////////////////////////////////////////////////
enum MINI_OPCODE
{
    OPCODE_BNDLDX,
    OPCODE_BNDSTX,
    OPCODE_UNK
};

MINI_OPCODE find_out_opcode(uint8_t* ip)
{
decode_again:
    switch(*ip)
    {
        case 0x41:
        case 0x42:
        case 0x43:
        case 0x0f:
            ip++;
            goto decode_again;
        case 0x1a:
            return OPCODE_BNDLDX;
        case 0x1b:
            return OPCODE_BNDSTX;
        default:
            return OPCODE_UNK;
    }
    return OPCODE_UNK;
}
//Track BT page hotness
/*
 * callback function to maintain bt_hot_pages,
 * this function will be called from signal handler in mpx_rt
 * which handles SIGSEGV
 * return false if the target address does not belong to MPX bt pages
 */
#define PROFILE_PF_HANDLER 0
extern "C" bool maintain_hot_bt_pages(siginfo_t *si, uint8_t* ip)
{
    static int nested = 0;
    #if 0
    if (unlikely(memptx_initialized==false))
    {
        //any call into maintain_hot_bt_pages before initialization is a bug!
        fprintf(stderr,KRED "BUG! maintain hot page requested before initialization?!!!\n" KNRM );
        dump_internal_state();
        exit(-1);
    }
    #endif

    void* ptr = si->si_addr;
    if (unlikely(nested++>1))
    {
        fprintf(stderr, KRED "BUG! we fall into fault loop?!!! exit.\n" KNRM);
        exit(-1);
    }
    if (unlikely(si->si_code!=SEGV_ACCERR))
    {
        fprintf(stderr, KRED "si_code!=SEGV_ACCERR?\n" KNRM);
        goto errout;
    }
    if (unlikely(ptr==NULL))
    {
        fprintf(stderr, KRED "ptr == NULL?\n" KNRM);
        goto errout;
    }
    if (unlikely(find_out_opcode(ip)==OPCODE_UNK))
    {
        fprintf(stderr,KRED "OPCODE mismatch!\n" KNRM);
        goto errout;
    }
    goto good;

errout:
    fprintf(stderr,
        KRED "Going to crash, involved address:%p\n" KNRM
        KRED "SIGNAL CODE: %d\n" KNRM,
        ptr,
        si->si_code);
    dump_internal_state();
    nested--;
    return false;
    assert(call_refresh_pmap_when_ready==0);
    assert(call_me_when_ready==0);

good:
    void* page_addr = PAGE_PTR(ptr);
    //fprintf(stderr,"BT:%p\n", page_addr);
    in_handler = true;
    //statistics we need from pf handler
    stat_pf++;
    acct_pf += free_queue.water_level;
    size_t mt_pf_begin = rdtsc();
    #if PROFILE_PF_HANDLER
    size_t stop_watch_section[6] = {0,0,0,0,0,0};
    #endif
    /////////////////////////////////////////////////////////////////////////
    /*
     * HOT page stuff goes here
     */
    #if PROFILE_PF_HANDLER
    stop_watch_section[0] = mt_pf_begin;
    #endif
    struct mpx_bt_page* bt_page = (struct mpx_bt_page*)page_addr;
    int err;
    //mprotect(page_addr, PAGE_SIZE, PROT_READ|PROT_WRITE|PROT_POPULATE);
    if ((err=mprotect_mpx(page_addr, PROT_READ|PROT_WRITE))!=0)
    {
        perror("mprotect");
        exit(-1);
    }
    #if PROFILE_PF_HANDLER
    stop_watch_section[1] = rdtsc();
    stop_watch_section[2] = rdtsc();
    #endif
    bool is_pc = is_page_clean(page_addr);
    #if PROFILE_PF_HANDLER
    stop_watch_section[3] = rdtsc();
    #endif
    if (!is_pc)
    {
        scan_single_bt_page_and_invalid(page_addr);
    }
    /*
     * means that we are always up to date and don't need to be scanned at all
     * when there's full scan
     */
    bt_page->wl_kick = -1;//leave me alone
    //allocate new hp_node, if failed, evict one
    unsigned short new_hpn = hpn_alloc();
    if (new_hpn==HPN_NULL)
    {
        new_hpn = hpn_evict();
        struct mpx_bt_page* victim_bt_page
            = (struct mpx_bt_page*)hp_nodes[new_hpn].page_addr;
        /*
         * we need to start scan against free_queue from wl_kick
         */
        victim_bt_page->wl_kick = free_queue.water_level;//leave me alone
        #if PROFILE_PF_HANDLER
        stop_watch_section[4] = rdtsc();
        #endif
        //mprotect(victim_bt_page, PAGE_SIZE, PROT_NONE);
        mprotect_mpx(victim_bt_page, PROT_NONE);
        #if PROFILE_PF_HANDLER
        stop_watch_section[5] = rdtsc();
        #endif

        //we need to reset it because there could be pointers used in this kicked-out
        //page and the pointer could be free'd later and that will not be added to 
        //Q because MALLOC_LOCALITY will prevent it.
        malc_reset();
    }
    //construct the hp_node
    hp_nodes[new_hpn].page_addr = (unsigned long)page_addr;
#if USE_PFCNT
    bt_page->pf_cnt++;
    if (bt_page->pf_cnt==(unsigned short)(-1))
    {
        bt_page->pf_cnt = 65534;
    }
    hp_nodes[new_hpn].pf_cnt = bt_page->pf_cnt;
#endif
    //insert into the queue
    hpn_insert_ordered(new_hpn);
    /////////////////////////////////////////////////////////////////////////
    in_handler = false;
    size_t mt_pf_end = rdtsc();
    size_t epoch_t = (mt_pf_end - mt_pf_begin);
    t_pf += epoch_t;
    acc_t_pf += epoch_t;
    nested--;
    if (call_refresh_pmap_when_ready)
    {
        signal_handler(SIGTRAP);
    }
    if (call_me_when_ready)
    {
        signal_handler(SIGALRM);
    }
    #if PROFILE_PF_HANDLER
//profiler
    size_t s0 = stop_watch_section[1]-stop_watch_section[0];
    size_t s1 = stop_watch_section[3]-stop_watch_section[2];
    size_t s2 = stop_watch_section[5]-stop_watch_section[4];
    fprintf(stderr,
            "syscall0:%ld(%.2f %%)+syscall1:%ld(%.2f %%)+syscall2:%ld(%.2f %%)= %ld(%.2f %%) "
            "(total epoch:%ld)\n",
            s0, 100.0 * s0/epoch_t,
            s1, 100.0 * s1/epoch_t,
            s2, 100.0 * s2/epoch_t,
            s0+s1+s2, 100.0*(s0+s1+s2)/epoch_t,
            epoch_t);
    #endif
    return true;

}
////////////////////////////////////////////////////////////////////////////////
/*
 * should initialize after mpx
 */
__attribute__ ((constructor(1007))) void memptx_init(void) 
{
    /*
     * whatever the rest of the ctor does, we initialize the function
     * pointers first
     */
    free_fptr = (real_free)GetRealFunctionAddress("__libc_free");
	if((void*)free_fptr==(void*)free)
	{
		fprintf(stderr,"failed free\n");
		exit(-1);
	}

    malloc_fptr = (real_malloc)GetRealFunctionAddress("__libc_malloc");
	if((void*)malloc_fptr==(void*)malloc)
	{
		fprintf(stderr,"failed malloc\n");
		exit(-1);
	}
#if 0
    calloc_fptr = (real_calloc)GetRealFunctionAddress("__libc_calloc");
	if((void*)calloc_fptr==(void*)calloc)
	{
		fprintf(stderr,"failed calloc\n");
		exit(-1);
	}
#endif
    if (signal(SIGINT, signal_handler) == SIG_ERR)
    {
        fprintf(stderr, "unable to catch SIGINT!\n");
    }
    if (signal(SIGUSR1, signal_handler) == SIG_ERR)
    {
        fprintf(stderr, "unable to catsh SIGUSR1!\n");
    }
    if (signal(SIGTRAP, signal_handler) == SIG_ERR)
    {
        fprintf(stderr, "fatal! unable to catch SIGTRAP!\n");
        exit(-1);
    }
#if DYNAMIC_ADJUST_HQ
    if (signal(SIGALRM, signal_handler) == SIG_ERR)
    {
        fprintf(stderr, "unable to catch SIGALRM,"
                        "can not enable auto adjustment");
        exit(-1);
    }else
    {
        ualarm(100000,0);
    }
#endif
    scan_init();
    memptx_initialized = true;
}


__attribute__ ((destructor(1007))) void memptx_fini(void)
{
    memptx_initialized = false;
    scan_uninit();
    dump_configuration();
    dump_stat();
}

////////////////////////////////////////////////////////////////////////////////
INTERCEPTOR_TYPE void * malloc(size_t size)
{
	void * ret = NULL;

	if(unlikely(malloc_fptr==NULL))
	{
        return ret;
	}
	ret = malloc_fptr(size);

	RECORD_SYNCOP_WARG2(malloc, ret, size);
    malc_insert(ret);
	return ret;
}
#if 0
INTERCEPTOR_TYPE void * calloc(size_t nmemb, size_t size)
{
    void* ret = NULL;

	if(unlikely(calloc_fptr==NULL))
	{
        return ret;
	}
	ret = calloc_fptr(nmemb, size);
    return ret;
}
#endif
#if 0
INTERCEPTOR_TYPE void * realloc(void *ptr, size_t size)
{
	static real_realloc fptr = NULL;
	if(fptr==NULL)
	{
		fptr = (real_realloc)GetRealFunctionAddress("__libc_realloc");
		if((void*)fptr==(void*)realloc)
		{
			printf("failed realloc\n");
			exit(-1);
		}
	}
	return fptr(ptr, size);
}
#endif

/*
 * only allow one free at a time
 */
INTERCEPTOR_TYPE void free(void *ptr)
{
    /*
     * we want this to be per-thread
     */
    static volatile int free_cnt = 0;
	RECORD_SYNCOP_WARG(free,ptr);

    if (unlikely(free_fptr==NULL))
        return;

    if (unlikely(memptx_initialized==false))
    {
        free_fptr(ptr);
    }

    if(likely(__sync_bool_compare_and_swap(&free_cnt,0,1)))
    {
        //free_cnt = 1;
        in_handler = true;
        /*
         * don't put into free queue if its just been malloced and free'd
         * and that has nothing to do with cold pages
         */
        free_fptr(ptr);

        bool recent_update = malc_exists(ptr);

		if (!recent_update)
		{
        	free_queue.items[free_queue.water_level] = (unsigned long)ptr;
	        free_queue.water_level++;
		}

        if(unlikely(free_queue.water_level>=bfq_allowed_max))
        {
slow_path:
            size_t mt_sf_begin = rdtsc();
            stat_slow_free++;
        	acct_sp += scan_all_bps_and_invalid();
            free_queue.water_level = 0;
            malc_reset();
            size_t mt_sf_end = rdtsc();
            size_t epoch_t = (mt_sf_end - mt_sf_begin);
            t_sf += epoch_t;
            acc_t_sf += epoch_t;
        }else
        {
            size_t mt_ff_begin = rdtsc();
            stat_fast_free++;
            //fast path, scan only hot bt pages
    	    acct_fp += scan_hot_bps_and_invalid(ptr);
            size_t mt_ff_end = rdtsc();
            size_t epoch_t = (mt_ff_end - mt_ff_begin);
            t_ff += epoch_t;
            acc_t_ff += epoch_t;
            //but if there are lots of work to do...
            //if ( (mt_ff_end-mt_ff_begin)>2000)
            //{
            //    acct_fp = 0;
            //    goto slow_path;
            //    unsigned short q = hpn_free_quota;
            //    hpn_set_quota(0);
            //    hpn_set_quota(q);
            //    malc_reset();
            //}
        }
        __sync_bool_compare_and_swap(&free_cnt,1,0);
    }else
    {
        fprintf(stderr,KRED "BUG! nested free()?!!\n" KNRM);
        free_fptr(ptr);
    }
	stat_free++;
    in_handler = false;
    if (call_refresh_pmap_when_ready)
    {
        signal_handler(SIGTRAP);
    }
    if (call_me_when_ready)
    {
        signal_handler(SIGALRM);
    }
}

#if 0
////////////////////////////////////////////////////////////////////////////////
//for C++
#define OPERATOR_NEW_BODY(mangled_name) \
	void * ret; \
	static real_malloc fptr = NULL;\
	if(fptr==NULL) \
	{ \
		fptr = (real_malloc)GetRealFunctionAddress("__libc_malloc"); \
		if((void*)fptr==(void*)malloc) \
		{ \
			printf("failed malloc\n"); \
			exit(-1); \
		} \
	} \
	ret = fptr(size); \
	RECORD_SYNCOP_WARG2(new, ret, size); \
	return ret;

INTERCEPTOR_TYPE void *operator new(size_t size)
{
	OPERATOR_NEW_BODY(_Znwm);
}

INTERCEPTOR_TYPE void *operator new[](size_t size)
{
	OPERATOR_NEW_BODY(_Znam);
}

INTERCEPTOR_TYPE void *operator new(size_t size, std::nothrow_t const&)
{
	OPERATOR_NEW_BODY(_ZnwmRKSt9nothrow_t);
}

INTERCEPTOR_TYPE void *operator new[](size_t size, std::nothrow_t const&)
{
	OPERATOR_NEW_BODY(_ZnamRKSt9nothrow_t);
}

#define OPERATOR_DELETE_BODY(mangled_name) \
 	static real_free fptr = NULL; \
	if(fptr==NULL) \
	{ \
		fptr = (real_free)GetRealFunctionAddress("__libc_free"); \
		if((void*)fptr==(void*)free) \
		{ \
			printf("failed free\n"); \
			exit(-1); \
		} \
	} \
	RECORD_SYNCOP_WARG(delete, ptr);\
	return fptr(ptr);

INTERCEPTOR_TYPE void operator delete(void *ptr) throw()
{
	OPERATOR_DELETE_BODY(_ZdlPv);
}

INTERCEPTOR_TYPE void operator delete[](void *ptr) throw()
{
	OPERATOR_DELETE_BODY(_ZdaPv);
}

INTERCEPTOR_TYPE void operator delete(void *ptr, std::nothrow_t const&)
{
	OPERATOR_DELETE_BODY(_ZdlPvRKSt9nothrow_t);
}

INTERCEPTOR_TYPE void operator delete[](void *ptr, std::nothrow_t const&)
{
	OPERATOR_DELETE_BODY(_ZdaPvRKSt9nothrow_t);
}
#endif

