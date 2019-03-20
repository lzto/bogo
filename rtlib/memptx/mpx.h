/*
 * MPX Definitions
 * 2016-2017 Tong Zhang<ztong@vt.edu>
 */
#ifndef _MPX_
#define _MPX_

/*
 * mprotect extension
 * prefault page to avoid minor fault
 */
#define PROT_POPULATE 0x10

#include <stddef.h>

#define MEMPTX_STAT 1
#define DEBUG 0
#define MEMPTX_DUMP_CFG 0

/*
 * adjust H and Q size on the fly
 */
#define DYNAMIC_ADJUST_HQ 1

/*
 * dump malloc/free information
 */
#define MALLOC_LOCALITY 1
//hash bucket configuration
//how many buckets are there
#define MAL_SIZE 1024
//bucket size
#define BUCKET_SIZE 15

/*
 * dump malloc/free information
 */
#define DUMP_MALLOC_AND_FREE 0

////////////////////////////////////////////////////////////////////////////////
//Basic PAGE Calculation

#define PAGE_BITS (12)
#define PAGE_SIZE (1<<PAGE_BITS)
#define PAGE_NUMBER(ptr) (((unsigned long long)ptr)>>PAGE_BITS)
#define PAGEMAP_ENTRY (8)
#define PAGE_MASK 0xFFFFFFFFFFFFF000
#define PAGE_PTR(x) (void*)((unsigned long)x & PAGE_MASK)


////////////////////////////////////////////////////////////////////////////////
//definitions copied from libmpx:mpx_wrappers.c and mpxrt.h

/* x86_64 directory size is 2GB.  */
#define NUM_L1_BITS 28
#define NUM_L2_BITS 17
#define NUM_IGN_BITS 3
#define MPX_L1_ADDR_MASK  0xfffffffffffff000ULL
#define MPX_L2_ADDR_MASK  0xfffffffffffffff8ULL
#define MPX_L2_VALID_MASK 0x0000000000000001ULL

#define REG_IP_IDX    REG_RIP
#define REX_PREFIX    "0x48, "

#define XSAVE_OFFSET_IN_FPMEM 0

#define MPX_L1_SIZE ((1UL << NUM_L1_BITS) * sizeof (void *))
#define MPX_L1_ENTRIES ((1UL << NUM_L1_BITS))

#define MPX_L2_SIZE ((1UL << NUM_L2_BITS) * sizeof (void *) * 4 )
#define MPX_L2_ENTRIES ((1UL << NUM_L2_BITS))

/* The mpx_pointer type is used for getting bits
   for bt_index (index in bounds table) and
   bd_index (index in bounds directory).  */
typedef union
{
  struct
  {
    unsigned long ignored:NUM_IGN_BITS;
    unsigned long l2entry:NUM_L2_BITS;
    unsigned long l1index:NUM_L1_BITS;
  };
  void *pointer;
} mpx_pointer;

/* The mpx_bt_entry struct represents a cell in bounds table.
   lb is the lower bound, ub is the upper bound,
   p is the stored pointer.  */
struct mpx_bt_entry
{
  void *lb;
  void *ub;
  void *p;
  void *reserved;
};

static_assert(sizeof(struct mpx_bt_entry)==(8*4), "sizeof(mpx_bt_entry)!=(8*4)");

/*
 * serve our special need
 */
struct mpx_bt_page
{
    //dont touch
    unsigned long obscure0[3];
    //pf times
    unsigned short pf_cnt;
    //free_queue water level when being kicked out
    unsigned short wl_kick;
    //padding
    unsigned short __padding[2];
    //the above use 32 bytes, left 127 entries
    unsigned long obscure1[508];
};

static_assert(sizeof(struct mpx_bt_page)==(4096), "sizeof(struct mpx_bt_page)!=(4096)");

/* A special type for bd is needed because bt addresses can be modified.  */
typedef struct mpx_bt_entry * volatile * bd_type;
////////////////////////////////////////////////////////////////////////////////

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

//////////////////////////////////////////////////////////////////////////////
//lazy scan core
/*
 * Total size of the queue will be
 *     ( 4 + BFQ_SIZE*8 ) + padding = n*4096 Byte
 */
#define BFQ_SIZE 8190
#define BFQ_ALLOWED_INITIAL (BFQ_SIZE)
extern size_t bfq_allowed_max;

struct free_queue {
    unsigned long items[BFQ_SIZE];
    unsigned short water_level;
}__attribute__((aligned(PAGE_SIZE)));

/*
 * track at most HOT_PAGES_MAX number of hot pages, if it reached max
 * we flush hot pages and start over again, this prevent fast path from
 * scanning lots of pages
 */
#define HOT_PAGES_MAX 1024
//#define HOT_PAGES_QUOTA_INITIAL (HOT_PAGES_MAX-2)
#define HOT_PAGES_QUOTA_INITIAL (16)
#define MIN_HPN_FREE_QUOTA 8
//#define MIN_HPN_FREE_QUOTA 6


/*
 * misc functions
 */
static __inline__ unsigned long long rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
/*
 * use page fault time to determine eviction sequence
 */
#define USE_PFCNT 0
/*
 * scan hot pages use multi-thread
 */
#define USE_THREAD 0
#define WORKERS_N 2

#endif

