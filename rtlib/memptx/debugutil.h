/*
 * Debug utils
 * 2016 Tong Zhang <ztong@vt.edu>
 */
#ifndef _DEBUG_UTIL_
#define _DEBUG_UTIL_
////////////////////////////////////////
//Colorful concole
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
////////////////////////////////////////
//debug output
#define WARN_ONCE(a) \
{\
    static bool _warn_once = true;\
    if (_warn_once)\
    {\
        _warn_once = false;\
        a;\
    }\
}

#endif//_DEBUG_UTIL_

