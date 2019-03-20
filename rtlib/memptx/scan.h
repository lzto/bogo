/*
 * MPX helper
 * 2016-2017 Tong Zhang <ztong@vt.edu>
 */
#ifndef _MPX_SCAN_
#define _MPX_SCAN_
#include "mpx.h"

#define HPN_NULL ((unsigned short)(-1))

//////////////////////////////////////////////////////////////////////////////
/*
 * hot page linked list definitions goes here
 */
struct hp_node
{
    //page address
    unsigned long page_addr;
    //page fault counter, cache mpx_bt_page->pf_cnt here for better locality
    unsigned short pf_cnt;
    unsigned short permission;//read or write or r+w
    unsigned short next;
    unsigned short prev;
    //padding to 16 bytes
    unsigned short padding[0];
};

static_assert(sizeof(struct hp_node)==16, "sizeof(struct hp_node)!=16");

unsigned short hpn_alloc();
void hpn_free(unsigned short node);
void hpn_insert_ordered(unsigned short node);
unsigned short hpn_evict();
void hpn_set_quota(int new_quota);
unsigned short hpn_find_page(void* page_addr);//return hp_node index
unsigned short hpn_get_page_permission(void* page_addr);

//extern struct hp_node hp_nodes[HOT_PAGES_MAX];
extern struct hp_node *hp_nodes;
extern int hpn_head;
extern int hpn_tail;
extern int hpn_free_cnt;//actual number of free nodes
extern int hpn_free_quota;//allowed hot pages, at most

//////////////////////////////////////////////////////////////////////////////
//scan interface
size_t scan_all_bps_and_invalid();
size_t scan_hot_bps_and_invalid(void *p);
void scan_single_bt_page_and_invalid(void *bte);

void scan_init();
void scan_uninit();

#endif

