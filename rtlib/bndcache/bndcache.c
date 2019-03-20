/*
 * bound cache for MPX
 * 2017 Tong Zhang <ztong@vt.edu>
 * Better compiled this file with llvm to .bc file
 * and with all functions being inlined
 */

/*
 * This is bound cache for MPX
 * This bound cache gathers scattered bound load/bound store requests together
 * to one MPX page, hoping that the TLB miss and cache miss could be minimized 
 * for some application who access large amount of different MPX BT pages within
 * a very short period of time
 * Current, the MPX bound cache is implemented as a write back cache with LRU 
 * replacement policy
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * debug flag and lousy macro
 */
#define LIBRT_DEBUG 0

#define USE_CACHE_STATISTIC 0

#if USE_CACHE_STATISTIC
size_t bnd_cache_request = 0;
size_t bnd_cache_hit = 0;
size_t bnd_cache_evict = 0;
size_t bnd_cache_miss = 0;
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

#define LLMPX_FUNC(func) _llmpx_##func

#define LLMPX_BND_CACHE_FUNC(ret_type, func, ...) \
    ret_type \
    __attribute__((always_inline)) \
    _llmpx_##func(__VA_ARGS__)


#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * how many sets do we have in the cache
 */
#define CACHE_SIZE (1024u)

struct cache_line
{
    unsigned long tag:63;
    bool dirty:1;
}__attribute__((__aligned__(8)));

#define CACHE_WAY (0x4u)

struct cache_set
{
    struct cache_line cl[CACHE_WAY];//4-way
    unsigned char lru_bits:4;
    unsigned char valid:4;
}__attribute__((__aligned__(8)));

//align to 4k boundary
static struct cache_set cache_tags[CACHE_SIZE]
    __attribute__((__aligned__(4096)))
    = {{{{0,false}},0,0}};

/*
 * pseudo lru state machine
 */
struct lru_cog_replace
{
    unsigned char replace:2;
};
static struct lru_cog_replace pseudo_lru_replace[] =
{
    {0x0u}, {0x0u}, {0x1u}, {0x1u}, {0x2u}, {0x3u}, {0x2u}, {0x3u}
};
struct lru_cog_state
{
    unsigned char next_state:3;
};
//[line][state]
static struct lru_cog_state pseudo_lru_refer[4][8] = 
{
    //0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111
    //line 0->11_
    {{0b110u},{0b111u},{0b110u},{0b111u},{0b110u},{0b111u},{0b110u},{0b111u}},
    //line 1->10_u
    {{0b100u},{0b101u},{0b100u},{0b101u},{0b100u},{0b101u},{0b100u},{0b101u}},
    //line 2->0_1u
    {{0b001u},{0b001u},{0b011u},{0b011u},{0b001u},{0b001u},{0b011u},{0b011u}},
    //line 3->0_0
    {{0b000u},{0b010u},{0b000u},{0b010u},{0b000u},{0b010u},{0b000u},{0b010u}}
};

static unsigned char __attribute__((noinline))
LLMPX_FUNC(bnd_cache_miss)(struct cache_set* cache_set, void* ptr, bool is_load)
{
    unsigned char line;
    unsigned char lru_bits = cache_set->lru_bits;
    struct cache_line* result;
    void* old_ptr;
#if USE_CACHE_STATISTIC
    bnd_cache_miss++;
#endif
    if (cache_set->valid<CACHE_WAY)
    {
        line = cache_set->valid;
        result = &(cache_set->cl[line]);
        old_ptr = (void*)(result->tag);
        cache_set->valid++;
        goto bring_into_cache_line;
    }
//evict_cache_line:
    /*
     * desired entry not found
     * find a victim and evict if needed
     */
    line = pseudo_lru_replace[lru_bits].replace;
    result = &(cache_set->cl[line]);
    old_ptr = (void*)(result->tag);

    //if the cache line is dirty we need to write it back before evict
    if (result->dirty)
    {
        __asm__ __volatile__
        ("bndldx (%0,%1), %%bnd0\n"
         "bndstx %%bnd0, (%2,%1)"
            ://nothing modified
            :"r" (result), "r" (*(void**)old_ptr), "r"(old_ptr):);
        result->dirty = false;
    }
#if USE_CACHE_STATISTIC
    bnd_cache_evict++;
#endif

bring_into_cache_line:
    if (is_load)
    {
        //only do this if it is load
        //issue bndldx(ptr,*ptr) and bndstx(result, *ptr) to bring bnd into cache
        __asm__ __volatile__
            ("bndldx (%0,%1), %%bnd0\n"//load bound into bnd0
             "bndstx %%bnd0, (%2,%1)"//store bnd0 into cache
                ://nothing modified
                :"r" (ptr), "r" (*(void**)ptr), "r" (result):);
    }
    result->tag = (unsigned long)ptr;
    return line;
}

////////////////////////////////////////////////////////////////////////////////
/*
 * call this to get cache line address
 * seperate fast path and slow path
 * inline fast path and noinline slow path
 */
LLMPX_BND_CACHE_FUNC(void*, bnd_cache_demand, void*ptr, bool is_load)
{
    unsigned cache_set_no = (unsigned long)ptr;
    cache_set_no = ((cache_set_no>>16) + 
                    cache_set_no) % CACHE_SIZE;
    struct cache_set* cache_set = &cache_tags[cache_set_no];
    unsigned char lru_bits = cache_set->lru_bits;
    unsigned char line;
    struct cache_line* result;
    void* old_ptr;
#if USE_CACHE_STATISTIC
    bnd_cache_request++;
#endif
    //find cache entry, return if found a match
    for(line=0;line<CACHE_WAY;++line)
    {
        result = &(cache_set->cl[line]);
        old_ptr = (void*)(result->tag);
        if (old_ptr==ptr)
        {
#if USE_CACHE_STATISTIC
    bnd_cache_hit++;
#endif
            goto out;
        }
    }
    line = LLMPX_FUNC(bnd_cache_miss)(cache_set, ptr, is_load);//slow path
    result = &(cache_set->cl[line]);
out:
    result->dirty |=(!is_load);
    cache_set->lru_bits
        = pseudo_lru_refer[line][lru_bits].next_state;
    return (void*)result;
}

////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

