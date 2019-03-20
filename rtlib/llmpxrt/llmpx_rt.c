/*
 * llvm mpx run time library
 * 2016-2017 Tong Zhang <ztong@vt.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

/*
 * debug flag
 */
#define LIBRT_DEBUG 0
#define USE_CACHE_STATISTIC 0
#define USE_DUMP_MPX_STATISTICS 0

#if USE_CACHE_STATISTIC
extern size_t bnd_cache_request;
extern size_t bnd_cache_hit;
extern size_t bnd_cache_evict;
extern size_t bnd_cache_miss;
#endif

#if LIBRT_DEBUG
#define DBG(level, ...) \
if (LIBRT_DEBUG<=level) \
{\
    fprintf(stderr, __VA_ARGS__); \
}
#else
#define DBG(level, ...)
#endif

#define LLMPX_FUNC(ret_type, func, ...) \
    ret_type \
    _llmpx_##func(__VA_ARGS__)

#ifdef __cplusplus
extern "C"
{
#endif


#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

/*
 * test stub
 */
LLMPX_FUNC(void, test)
{
    printf("this is a test\n");
}
////////////////////////////////////////////////////////////////////////////////
//lousy MPX stuff
#define NUM_L1_BITS 28
#define NUM_L2_BITS 17
#define NUM_IGN_BITS 3

#define MPX_BDE_INDEX_MASK  0x0000fffffff00000ULL
#define MPX_BTE_INDEX_MASK  0x00000000000ffff8ULL
#define MPX_L1_ADDR_MASK  0xfffffffffffff000ULL
#define MPX_L2_ADDR_MASK  0xfffffffffffffff8ULL
#define MPX_L2_VALID_MASK 0x0000000000000001ULL

#define MPX_L1_SIZE ((1UL << NUM_L1_BITS) * sizeof (void *))
#define MPX_L1_ENTRIES (size_t)((1UL << NUM_L1_BITS))

#define MPX_L2_SIZE ((1UL << NUM_L2_BITS) * sizeof (void *) * 4 )
#define MPX_L2_ENTRIES (size_t)((1UL << NUM_L2_BITS))

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
  unsigned long lb;
  unsigned long ub;
  unsigned long p;
  unsigned long reserved;
};

/* A special type for bd is needed because bt addresses can be modified.  */
typedef struct mpx_bt_entry * volatile * bd_type;

///////////////////////////////////////////////////////////////////////////////
static void* l1base = NULL;
bd_type bd = NULL;
extern void* get_bd();//libmpx.a
extern void mpxrt_prepare();//libmpx.a

struct mpx_bt_entry any_valid_entry = { 0, 0, 0, 0};


void dump_mpx_statistics();


/*
 * dummy function to bring in constructor/destructor
 */
void llmpx_rt_dummy_for_bring_in_ctor_dependency()
{
}

/*
 * should run after mpxrt constructor
 */
__attribute__ ((constructor (1006))) void llmpx_rt_init(void) 
{
    if (l1base!=NULL)
    {
        fprintf(stderr,"FUCK, llmpx_rt already initialized!\n");
        return;
    }
	l1base = get_bd();
	if (!l1base)
	{
        fprintf(stderr,"MPX not initialized!\n");
        fflush(stderr);
        exit(-1);
	}
    bd = (bd_type)l1base;
}

__attribute__ ((destructor(1006))) void llmpx_rt_fini(void)
{
#if USE_CACHE_STATISTIC
    fprintf(stderr,"=bound cache statistics=\n"
                    "requests: %lu\n"
                    "hit: %lu (%.2f %%)\n"
                    "miss: %lu\n"
                    "evict: %lu\n",
                    bnd_cache_request,
                    bnd_cache_hit, (100.0*bnd_cache_hit/bnd_cache_request),
                    bnd_cache_miss,
                    bnd_cache_evict);
#endif

#if USE_DUMP_MPX_STATISTICS
    dump_mpx_statistics();
#endif
}

struct mpx_bt_entry* walk_directory(void* ptr)
{
    unsigned long addr = (unsigned long)(ptr);
    unsigned int bde_index
        = (addr)>>(NUM_L2_BITS+NUM_IGN_BITS);
    unsigned int bte_index
        = (addr & MPX_BTE_INDEX_MASK)>>(NUM_IGN_BITS);
    /*
     * This should not happen
     * but some applications call this even before MPX is setup
     */
    if (unlikely(!bd))
        goto fail;

    struct mpx_bt_entry* bt = bd[bde_index];
    if (((unsigned long)bt)&MPX_L2_VALID_MASK)
    {
        bt = (struct mpx_bt_entry*)((unsigned long)bt&(~MPX_L2_VALID_MASK));
        return &(bt[bte_index]);
    }
fail:
    return &any_valid_entry;
}

///////////////////////////////////////////////////////////////////////////////
//Helper functions for Temporal Memory Safety
/*
 * find the lock belongs the ptr
 */
static unsigned long find_lock(void* ptr)
{
    return 0;
}

static unsigned long* find_key(void* ptr)
{
    struct mpx_bt_entry* entry = walk_directory(ptr);
    return &(entry->reserved);
}
/*
 * helper functions for temporal memory safety 
 */

static unsigned long global_lock_count = 0;

/*
 * allocate lock for given memory trunk
 */
LLMPX_FUNC(unsigned long, temporal_lock_alloca, void* ptr, unsigned long size)
{
    unsigned long lock = global_lock_count++;
    DBG(1, "alloca lock for ptr:%p, +0x%lx , (lock=0x%lx)\n",
        ptr, size, lock);
    //add lock, (ptr,size) pair to database
    return lock;
}

/*
 * store/load key of ptr to bound table?
 * this should be inserted after the ptr
 */
LLMPX_FUNC(void, temporal_key_store, void** ptr, unsigned long key)
{
    DBG(1, "BD store key 0x%lx for ptr %p\n", key, ptr);
    unsigned long* kaddr = find_key(ptr);
    DBG(1, "   kaddr=%p\n", kaddr);
    *kaddr = key;
}

LLMPX_FUNC(unsigned long, temporal_key_load, void** ptr)
{
    DBG(1, "BD load key for ptr %p \n", ptr);
    unsigned long key = 0;
    unsigned long* kaddr = find_key(ptr);
    DBG(1, "   kaddr=%p\n", kaddr);
    if (kaddr)
        key = *kaddr;
    DBG(1, "   key = %ld \n", key);
    return key;
}

/*
 * temporal check, the key belongs to the current ptr value
 * we need to search its lock and compare the lock-key
 */
LLMPX_FUNC(void, temporal_chk, void* ptr, unsigned long key)
{
    unsigned long lock = find_lock(ptr);
    DBG(1, "chk ptr=%p, (key,lock)=(0x%lx,0x%lx)\n", ptr, key, lock);
    if (lock == key)
    {
        return;
    }
    //fprintf(stderr,"K L mismatch! %p (%ld,%ld)\n", ptr, key, lock);
    return;
}
////////////////////////////////////////////////////////////////////////////////
/*
 * This function helps to dump bndldx bndstx info
 */
LLMPX_FUNC(void, dbg_dump_bndldstx, void*ptr, bool is_load)
{
    char* x = "ldx";
    if (is_load)
        x = "stx";
    fprintf(stderr,"bnd%s %p\n", x , ptr);
}

//////////////////////////////////////////////////////////////////////////////
/*
 * To make life easier, I made this helper stub for gdb
 * pass the address of pointer to it to figure out the bound entry in MPX BT
 */
void find_bte(void* ptr)
{
    struct mpx_bt_entry* entry = walk_directory(ptr);
    if (entry!=&any_valid_entry)
    {
        fprintf(stderr,"entry %p [0x%lx,0x%lx] ptr=0x%lx, rev=0x%lx\n",
            entry,
            entry->lb,
            ~(entry->ub),
            entry->p,
            entry->reserved);
    }
}

/*
 * dump MPX usage/utilization
 * traverse MPX BD to find any valid entry
 * then walk all BT to collect utilization info
 */
__attribute__ ((bnd_legacy)) static inline struct mpx_bt_entry *
guess_valid_bt(size_t bd_index)
{
	struct mpx_bt_entry *bt = (struct mpx_bt_entry *)
		((uintptr_t) bd[bd_index] & MPX_L2_ADDR_MASK);
	if(!(bt) || !((uintptr_t) bd[bd_index] & MPX_L2_VALID_MASK))
	{
		return NULL;
	}
	return bt;
}

void dump_mpx_statistics()
{
    size_t bd_index;
    struct mpx_bt_entry* bt;
    size_t total_bt_pages = 0;
    size_t total_bt_entry_used = 0;
    fprintf(stderr,"== MPX UTILIZATION ==\n");
    //walk BE
    for (bd_index = 0; bd_index<MPX_L1_ENTRIES; bd_index++)
    {
        if ((bt=guess_valid_bt(bd_index))==NULL)
        {
            continue;
        }
        /*
         * walk BT, for each page, DO NOT trigger PF here
         * need first consult OS for allocated BT page list
         * then search through it and collect statistics
         */
        //there are 1024/(8*4)=128 entries per BT page
        short used_entry = 0;
        char buf[256];
        short bufp = 0;
        for (int i=0; i<MPX_L2_ENTRIES; i++)
        {
            if (i%128==0)
            {
                used_entry = 0;
                bufp = 0;
                buf[bufp] = '[';
            }
            bufp++;
            if (bt[i].p==0)
            {
                buf[bufp] = '_';
            }else
            {
                buf[bufp] = 'x';
                used_entry++;
            }
            if (i%128 == 127)
            {
                if(used_entry==0)
                {
                    continue;
                }
                buf[129]=']';
                buf[130]='\0';
                fprintf(stderr,"%s %d/128\n",buf,used_entry);
                total_bt_entry_used+=used_entry;
                total_bt_pages++;
            }
        }
    }
    fprintf(stderr,"Total MPX BT Pages Used: %ld\n", total_bt_pages);
    fprintf(stderr,"Total MPX BT Entry Used: %ld\n", total_bt_entry_used);
    fprintf(stderr,"MPX BT Utilization: %.2f %%\n",
            100.0*total_bt_entry_used*(8*4)/(total_bt_pages*4096));
    fprintf(stderr,"=====================\n");
}

////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
}
#endif

