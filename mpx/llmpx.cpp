/*
 * llvm mpx support
 * 2016-2018 Tong Zhang <ztong@vt.edu>
 * -----------------------------
 *  IR level bound generation, propogation and check
 *
 *  1. for spatial safety and temporal safety.
 *
 *  2. for temporal safety, we use key-lock match method
 *  keys are stored in bound table
 *      `- we need to write key info after making the bndstx
 *  the lock is stored somewhereelse
 *      `- this need to be stored for range search?
 *  
 *  refer to AMD64 ABI Draft 0.3 for calling convention
 */


#include "llvm/Transforms/Instrumentation.h"

#include "llvm/ADT/Statistic.h"

#include "llvm/Pass.h"

#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/Dominators.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#include "llvm-c/Core.h"

#include "llvm/ADT/SetVector.h"

#include <list>
#include <map>
#include <stack>
#include <queue>
#include <set>
#include <algorithm>

#include "color.h"


#include "stopwatch.h"
STOP_WATCH;

using namespace llvm;

#define DEBUG_TYPE "llmpx"

/*
 * debug mpxpass
 */
#define DEBUG_MPX_PASS_0_1 0
#define DEBUG_MPX_PASS_1 0
#define DEBUG_MPX_PASS_1_5 0
#define DEBUG_MPX_PASS_2 0
#define DEBUG_MPX_PASS_3 0

#define DEBUG_AA_PASS 1
#define DEBUG_DEAD_BNDSTX_ELIM 1
#define DEBUG_DEAD_BNDLDX_ELIM 1

#define DEBUG_PROCESS_INST 0

#define DEBUG_WRAPPER_PROCESSOR 0

/*
 * bb level bound check consolidation
 */
#define BND_CHK_CONSOLIDATION 1
/////////////////////////////////////////////////////////////////
//The following two optimizations may result in false negatives
/*
 * AA optimization switch
 */
#define AA_OPTIMIZATION 0

/*
 * safe access elimination switch
 */
#define SAFE_ACCESS_ELIMINATION 1
/////////////////////////////////////////////////////////////////
/*
 * dead bound elimination
 */
#define DEAD_BOUND_ELIMINATION 1

/*
 * debug handler function
 */
#define DEBUG_HANDLE_CALL 0
#define DEBUG_HANDLE_CALL_CAST 0
#define DEBUG_HANDLE_INVOKE 0
#define DEBUG_HANDLE_INVOKE_CAST 0
#define DEBUG_HANDLE_EXTRACTVALUE 0
#define DEBUG_HANDLE_GEP 0
#define DEBUG_HANDLE_PHINODE 0
#define DEBUG_HANDLE_RET 0
#define DEBUG_HANDLE_STORE 0
#define DEBUG_HANDLE_LOAD 0
#define DEBUG_HANDLE_ALLOCA 0
#define DEBUG_HANDLE_BINARY_OP 0
#define DEBUG_HANDLE_BITCAST 0
#define DEBUG_HANDLE_SELECT 0

#define DEBUG_INSERT_BOUND_CHECK 0

#define DEBUG_KEY_STORE 0

#define DEBUG_FIND_ACTUAL_TYPE 0

#define DEBUG_IS_SAFE_ACCESS 0

#define USE_MPX_TESTER 0

#define _LLMPX_MAX_FIND_DEPTH_ 4

/*
 * prefix for help library functions and symbols
 */
#define LLMPX_SYMBOL_PREFIX "_llmpx_"
#define LLMPX_WRAPPER_PREFIX "__mpx_wrapper_"

/*
 * define statistics if not enabled in LLVM
 */

#if (!LLVM_ENABLE_STATS)

#undef STATISTIC
#define CUSTOM_STATISTICS 1
#define STATISTIC(X,Y) \
unsigned long X;\
const char* X##_desc = Y;

#define STATISTICS_DUMP(X) \
    errs()<<"    "<<X<<" : "<<X##_desc<<"\n";

#endif

STATISTIC(FuncCounter, "Functions greeted");
STATISTIC(ExternalFuncCounter, "External function");
//STATISTIC(ValueTypeCacheHitRatio, "Count Value Type Cache Hit Ratio");
STATISTIC(ElimSafeAccess, "Total safe access check eliminated");
STATISTIC(ElimBound, "Total unused bound eliminated");
STATISTIC(TotalChecksAdded, "Total checks added");
STATISTIC(TotalBNDLDXAdded, "Total bndldx added");
STATISTIC(TotalBNDSTXAdded, "Total bndstx added");
STATISTIC(TotalBNDMKAdded, "Total bndmk added");
STATISTIC(TotalStaticBNDAdded, "Total Static Bound Added");
STATISTIC(TotalAliasSetsFound, "Total Alias Set Found");
STATISTIC(TotalAliasSetsUsedForMPX, "Alias Set used for MPX Partationing");
STATISTIC(DeadBNDSTXEliminated, "Total dead bndstx eliminated");
STATISTIC(DeadBNDLDXEliminated, "Total dead bndldx eliminated");
STATISTIC(ConsolidatedBNDCHK, "Total bound checks consolidated");

class llmpx : public ModulePass
{
    private:
        bool runOnModule(Module &);
        bool mpxPass(Module &);

        void harden_cfi(Module &);

        #ifdef CUSTOM_STATISTICS
        void dump_statistics();
        #endif
        void create_global_constants(Module&);
        void collect_safe_access(Module&);
        /*
         * the safe access list, which we don't need to check
         * map from the instruction to the pointer it want to access
         */
        std::map<Value*, Value*> safe_access_list;

        void transform_functions(Module&);
        void transform_global(Module&);
        void process_each_function(Module&);
        void cleanup(Module&);
        void verify(Module&);

        int dead_bndstx_elimination(Module&);
        int dead_bndldx_elimination(Module&);

        /*
         * Helper function, try to find the real value which has its id 
         * associated with it 
         */
        Value* find_true_val_has_aa_id(Value*);

        /*
         * remove any used bound, return the number of bound removed
         */
        int remove_dead_bound(Module&);

#if USE_MPX_TESTER
        bool mpxTester(Module &);
#endif
        
        /*
         * creat symbols for helper library
         */
        void create_llmpx_symbols(Module&);

        /*
         * a list for llvm mpx symbols
         */
        Function* _llmpx_test;
        Function* _llmpx_temporal_chk;
        Function* _llmpx_temporal_lock_alloca;
        Function* _llmpx_temporal_key_load;
        Function* _llmpx_temporal_key_store;

        /*
         * use this to collect bound table usage statistics
         */
        Function* _llmpx_dbg_dump_bndldstx;
        std::list<Value*>
            insert_dbg_dump_bndldstx(Instruction *I, Value* ptr, bool is_load);

        /*
         * create mpx intrinsic symbols
         */
        void create_mpx_intr_symbols(Module&);
        /*
         * MPX Intrinsic Functions
         */
        Function* mpx_bndmk;
        Function* mpx_bndldx;
        Function* mpx_bndstx;
        Function* mpx_bndclrr;
        Function* mpx_bndclrm;
        Function* mpx_bndcurr;
        Function* mpx_bndcurm;
        Function* mpx_bndcn;

        /*
         * Bound Cache
         */
        void create_llmpx_bnd_cache_symbols(Module&);
        Function* _llmpx_bnd_cache_demand;

        /*
         * create symbols for wrapper functions in mpxwrap library
         */
        void create_llmpx_wrapper_symbols(Module&);
        //contains mapping from orig function to casted wrapper function
        std::map<Function*, Value*> orig_to_cw_flist;

        /*
         * context for current module
         */
        LLVMContext *ctx;
        Module* module;
        Function* cfunc;
        ObjectSizeOffsetVisitor* obj_size_vis;

        /*
         * tr_flist: transformed function list
         * functions transformed with bound information added
         */
        //contains mapping from old to new function
        std::map<Function*, Function*> tr_flist;
        //contains mapping from new to old function
        std::map<Function*, Function*> revtr_flist;
        //contains mapping from old_func to
        //the number of bound need to be added for return
        std::map<Function*, unsigned int> tr_flist_ret;
        //contains number of bound returned for certain type
        std::map<Type*, unsigned int> tr_bndinfo_for_rettype;

        //contains old function
        std::list<Function*> flist_orig;
        //contains new function
        std::list<Function*> flist_new;
        //contains the index of the first bound argument
        std::map<Function*, int> tr_bnd_list;
        //contains the index of pointer argument with byval attribute
        std::map<Function*, std::list<int>*> tr_ptrbyval_list;

        //contains mapping from old type to new type
        std::map<Type*, Type*> tr_typelist;
        //same as above, but for mpx wrapper
        //will eventually be merged with tr_typelist
        std::map<Type*, Type*> tr_typelist_for_wrap;

        /*
         * this stores the bound instruction need to be inserted
         * for each instruction
         * --------------------
         * original instruction => bndmk/bndldx instructions
         * original instruction => bndmov/bndstx instructions
         * original instruction => bndcl/bndcu instructions
         */
        std::map<Value*, std::list<Value*>*> bound_checklist;
        /*
         * similar to bound_checklist
         * this stores the key instruction need to be inserted
         */
        std::map<Value*, std::list<Value*>*> key_checklist;

        /*
         * global variable bound list, initialize only once 
         * for each application
         */
        std::map<Value*, std::list<Value*>*> gv_bound_checklist;
        /*
         * global variable key list initialize only once for each
         * application
         */
        std::map<Value*, std::list<Value*>*> gv_key_checklist;

        /*
         * gv bound/key load cache, for value,Instruction pair
         */
        std::map<std::pair<Value*, Instruction*>,
                std::list<Value*>*> gv_bound_checklist_cache;
        std::map<std::pair<Value*, Instruction*>,
                std::list<Value*>*> gv_key_checklist_cache;
        

        /*
         * delete marker for key/bound/instruction
         */
        std::list<Value*> delete_ii;
        
        /*
         * transformed type
         * This is a cache, if we already transformed such type
         * we don't have to do it again
         */
        bool has_transformed_type(Type*);
        Type* get_transformed_type(Type*);

        /*
         * orig type, transformed type
         */
        void add_transform_type_pair(Type*, Type*);

        //save thing as above,
        //eventually this will be merged with functions above
        bool has_transformed_type_for_wrap(Type*);
        Type* get_transformed_type_for_wrap(Type*);
        void add_transform_type_pair_for_wrap(Type*, Type*);

        /*
         * examine each function to see
         * whether the function need to be transformed
         */
        bool function_need_to_be_transformed(Function*);
        /*
         * examine phi node, if this phi node is for function pointer
         * and the function pointer type contains any pointer
         * either in return value or argument, we need to transform
         * this phi node
         */
        bool this_phi_node_need_transform(PHINode *);
        /*
         * transform function type into instrumented function type
         * bound information will be added to 
         * return - if pointer type or aggregated type with pointer in 
         *          it is returned
         * arguments - if any of the argument requires bound information
         */
        Type* transform_function_type(Type*);
        /*
         * generate bound_checklist
         */
        void gen_bound_checklist(Instruction*);
        /*
         * process bound check list we just gathered
         */
        void process_bound_checklist();
        /*
         * process each instruction
         * returns value not equal to orig value
         * if changed
         */
        Value* process_each_instruction(Value*);
        /*
         * helper function
         */
        void add_instruction_to_bcl(Value*);
        /*
         * find and return transformed function if possible,
         * otherwise return itself
         */
        Function* find_transformed_function(Value*);
        /*
         * find and return original function if possible,
         * otherwise return itself
         */
        Function* find_nontransformed_function(Value*);
        /*
         * is the function in original function need tranformation?
         */
        bool is_function_orig(Value*);
        /*
         * is the function a transformed function?
         */
        bool is_function_trans(Value*);
        /*
         * get bound for value, given that 
         * there's a bound
         */
        Value* get_bound(Value*, Instruction*);
        /*
         * get bound for value, given that 
         * there's a bound
         */
        Value* get_key(Value*, Instruction*);

        /*
         * associate metadata which belongs to Value*
         * to Instruction*
         */
        std::list<Value*>
            associate_meta(Instruction*, Value*);

        /*
         * insert spatial and temporal check
         */
        std::list<Value*>
            insert_check(Instruction *I, Value* ptr, bool is_load);
        /*
         * insert spatial/temporal checks for instruction I,
         * which want to access memroy pointed by ptr,
         * and the bound information is hold by bnd
         */
        std::list<Value*>
            insert_bound_check(Instruction *I, Value* ptr, bool is_load);
        std::list<Value*>
            insert_keylock_check(Instruction *I, Value* ptr, bool is_load);

        /*
         * insert bndldx for ptr before Instruction I
         */
        std::list<Value*>
            insert_bound_load(Instruction *I, Value* ptr, Value* ptrval);
        /*
         * all inserted bndldx is recorded here
         */
        std::list<Value*> bndldxlist;
        /*
         * insert key load for ptr before Instruction I
         */
        std::list<Value*>
            insert_key_load(Instruction *I, Value* ptr);


        /*
         * insert bndstx for ptr and bnd after Instruction I
         */
        std::list<Value*>
            insert_bound_store(Instruction*, Value *, Value*, Value*);
        /*
         * all inserted bndstx is recorded here
         */
        std::list<Value*> bndstxlist;
        /*
         * insert key store for ptr and key after Instruction I
         */
        std::list<Value*>
            insert_key_store(Instruction*, Value*, Value*);

        /*
         * rip off tail call attribute for call instruction
         */
        void ripoff_tail_call(Value*);
        /*
         * a set of global constant bound/key which might be useful
         */
        GlobalVariable* bnd_infinite;
        GlobalVariable* bnd_invalid;
        std::map<Function*, Value*> infinite_bnd_for_function;
        std::map<Function*, Value*> invalid_bnd_for_function;

        GlobalVariable* key_anyvalid;
        GlobalVariable* key_anyinvalid;
        std::map<Function*, Value*> anyvalid_key_for_function;
        std::map<Function*, Value*> anyinvalid_key_for_function;

        /*
         * helper function to load infinite/invalid bound/key after 
         * Instruction
         */
        Value* get_infinite_bound(Function*);
        Value* get_infinite_bound(Instruction*);
        Value* get_invalid_bound(Function*);
        Value* get_invalid_bound(Instruction*);

        Value* get_anyvalid_key(Function*);
        Value* get_anyvalid_key(Instruction*);
        Value* get_anyinvalid_key(Function*);
        Value* get_anyinvalid_key(Instruction*);

        /*
         * instruction handler
         */
        Value* handleAlloca(Value*);
        Value* handleBitCast(Value*);
        Value* handleCall(Value*);
        Value* handleInvoke(Value*);
        Value* handleInsertElement(Value*);
        Value* handleExtractElement(Value*);
        Value* handleExtractValue(Value*);
        Value* handleInsertValue(Value*);
        Value* handleGetElementPtr(Value*);
        Value* handleIntToPtr(Value*);
        Value* handleLoad(Value*);
        Value* handlePHINode(Value*);
        Value* handleSelect(Value*);
        Value* handleRet(Value*);
        Value* handleStore(Value*);
        //for int/pointer cast
        Value* handlePtrToInt(Value*);
        Value* handleBinaryOperator(Value*);

        bool is_safe_access(Value* addr, uint64_t type_size);
        bool is_safe_access_cache(Value* inst);

        /*
         * find actual type, for ugly load/store
         */
        Type* find_actual_type(Value*, bool);
        Type* find_actual_type(Value*, bool, int);
        std::map<Value*,Type*> at_cache0;
        std::map<Value*,Type*> at_cache1;

        /*
         * for debug purpose
         */
        std::stack<Value*> dbgstk;
        void dump_dbgstk();

        /*
         * should invoke aapass() first to get ipaa set,
         * invoke this function to get which aa group/alias
         * set this value belongs to
         */
        int get_aa_set_id(Value*);
        /*
         * map from set number to how many times this set is used
         */
        std::map<unsigned, int> stat_set_used;

    public:
        static char ID;
        llmpx() : ModulePass(ID)
        {
        }
        const char* getPassName()
        {
            return "llmpx";
        }
        void getAnalysisUsage(AnalysisUsage &au) const override
        {
            au.setPreservesAll();
            au.addRequired<AAResultsWrapperPass>();
            au.addPreserved<GlobalsAAWrapperPass>();
            au.addRequired<TargetLibraryInfoWrapperPass>();
            au.addRequired<ScalarEvolutionWrapperPass>();
        }
};

#ifdef CUSTOM_STATISTICS
void llmpx::dump_statistics()
{

    errs()<<"------------STATISTICS---------------\n";
    STATISTICS_DUMP(FuncCounter);
    STATISTICS_DUMP(ExternalFuncCounter);
    STATISTICS_DUMP(ElimSafeAccess);
    STATISTICS_DUMP(ElimBound);
    STATISTICS_DUMP(TotalChecksAdded);
    STATISTICS_DUMP(TotalBNDLDXAdded);
    STATISTICS_DUMP(TotalBNDSTXAdded);
    STATISTICS_DUMP(TotalBNDMKAdded);
    STATISTICS_DUMP(TotalStaticBNDAdded);
    STATISTICS_DUMP(TotalAliasSetsFound);
    STATISTICS_DUMP(TotalAliasSetsUsedForMPX);
    STATISTICS_DUMP(DeadBNDSTXEliminated);
    STATISTICS_DUMP(DeadBNDLDXEliminated);
    STATISTICS_DUMP(ConsolidatedBNDCHK);
    errs()<<"\n\n\n";
}
#endif

char llmpx::ID;

/*
 * command line options
 */
cl::opt<bool> llmpx_no_check("llmpx_no_check",
                cl::desc("no checks at all, only bound propogation - disabled by default"),
                cl::init(false));

cl::opt<bool> llmpx_store_only_check("llmpx_store_only_check",
                cl::desc("only check store"),
                cl::init(false));

cl::opt<bool> llmpx_enable_temporal_safety("llmpx_enable_temporal_safety",
                cl::desc("enable temporal safety check - like CETS\n"
                        "key stored in MPX table reserved area - disabled by default"),
                cl::init(false));

cl::opt<bool> llmpx_dump_bndldstx("llmpx_dump_bndldstx",
                cl::desc("dump bndstx/bndldx info - disabled by default"),
                cl::init(false));

cl::opt<bool> llmpx_bound_cache("llmpx_bound_cache",
                cl::desc("enable bound cache - search bound from bound cache\n"
                        "instead of load it from MPX table - disabled by default"),
                cl::init(false));

cl::opt<bool> llmpx_use_ppa("llmpx_use_ppa",
                cl::desc("store different pointer set into different MPX pool\n"
                        "partition pointer set for better performance\n"
                        "- disabled by default"),
                cl::init(false));

cl::opt<bool> llmpx_harden_cfi("llmpx_harden_cfi",
                cl::desc("harden data used for control flow for CFI\n"
                        "- disabled by default"),
                cl::init(false));

////////////////////////////////////////////////////////////////////////////////
/*
 * This is for alias analysis
 */
/*
 * Helper function
 * find/return the set which ptr belongs, create one if non-exists
 */
static std::set<Value*>& find_or_create_new_set(Value* ptr,
    std::list<std::set<Value*>> &list)
{
    for (auto& the_set : list)
    {
        if (the_set.find(ptr)!=the_set.end())
        {
            return the_set;
        }
    }
    std::set<Value*> new_list;
    new_list.insert(ptr);
    list.push_back(new_list);
    return list.back();
}

/*
 * HACK: is there any more elegent way to do this??
 */
static Value* get_arg(Function* func, unsigned argno)
{
    Function::arg_iterator ait = func->arg_begin();
    while(argno!=0)
    {
        ait++;
        argno--;
    }
    return dyn_cast<Value>(ait);
}

/*
 * get the alias set id for this ptr
 * Expensive O(n) search,
 * may be build a hash map for quick search
 * return -1 if not found in any set
 *
 * CAUTION: FIXME: call to get_aa_set_id should use value(i.e. *ptr) as parameter,
 * as we only know that which set of value alias to each other
 */
int llmpx::get_aa_set_id(Value* ptr)
{
#if AA_OPTIMIZATION
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////

/*
 * do not touch llmpx helper library
 */

bool is_llmpx_library_function(const std::string& str)
{
    if (str.find(LLMPX_SYMBOL_PREFIX) == 0)
    {
        return true;
    }
    return false;
}

/*
 * mpx wrapper for stdlib functions
 * transformed symbol should have LLMPX_WRAPPER_PREFIX
 * Call to these functions are now only have bound information,
 * key/lock information is not modeled.
 *      i.e. This only has support for spatial safety,
 *           eventually, temporal safety wrapper will be added.
 * For all these functions, if it is external symbol(and shoule be),
 * we rewrite CALLINST to use wrapper fuction instead
 * NOTE: The wrapper function prototypes are constructed at
 *       ::transform_functions
 */
static const char* llmpx_func_wrapper_list[] =
{
    "malloc",
    "mmap",
    "realloc",
    "calloc",
    "memset",
    "bzero",
    "memmove",
    "memcpy",
    "mempcpy",
    "strncat",
    "strcat",
    "stpcpy",
    "stpncpy",
    "strcpy",
    "strncpy",
    "strlen"
};

bool is_in_wrapper_list(const std::string& str)
{
    if (std::find(std::begin(llmpx_func_wrapper_list),
            std::end(llmpx_func_wrapper_list),
            str) != std::end(llmpx_func_wrapper_list))
    {
        return true;
    }
    return false;
}


/*
 * do not transform these functions
 */
static const char* llmpx_func_skip_list[] =
{
    //skip our ctor and dtor
    "llmpx_ctor",
    "llmpx_dtor",
    "main",
    "malloc",
    "free",
    //C++
    "_Znwj",//operator new(unsigned int)
    "_Znaj",//operator new[](unsigned int)
    "_Znwm",//operator new(unsigned long)
    "_Znam",//operator new[](unsigned long)
    "_ZdlPv",//operator delete(void*)
    "_ZdaPv",//operator delete[](void*)
    "_ZdlPvRKSt9nothrow_t",//operator delete(void*, std::nothrow_t const&)
    "_ZdaPvRKSt9nothrow_t"//operator delete[](void*, std::nothrow_t const&)
};

bool is_in_skip_list(const std::string& str)
{
    if (is_llmpx_library_function(str))
    {
        return true;
    }
    if (std::find(std::begin(llmpx_func_skip_list),
            std::end(llmpx_func_skip_list),
            str) != std::end(llmpx_func_skip_list))
    {
        return true;
    }
    return false;
}

/*
 * for C++, do not transform these basic blocks
 * most of these are for exception handling
 */
static const char* llmpx_bb_skip_list[] = 
{
#if 0
    "lpad",
    "catch",
    "catch.dispatch",
    "catch.fallthrough",
    "ehcleanup",
    "eh.resume",
    "ehspec.unexpected",
    "__except",
    "__except.ret",
    "filter.dispatch",
    "terminate.handler",
    "throw",
    "throw.cont",
    "cleanup.action",
    "cleanup.cont",
    "cleanup.done"
#else
    ""
#endif
};

bool is_in_bb_skip_list(const std::string& str)
{
    if (std::find(std::begin(llmpx_bb_skip_list),
            std::end(llmpx_bb_skip_list),
            str) != std::end(llmpx_bb_skip_list))
    {
        return true;
    }
    return false;
}


/*
 * do not instrument these functions
 */
static const char* llmpx_do_not_instrument_list[] =
{
    //skip ctor and dtor
    "llmpx_ctor",
    "llmpx_dtor"
};

bool is_in_do_not_instrument_list(const std::string& str)
{
    if (is_llmpx_library_function(str))
    {
        return true;
    }
    if (std::find(std::begin(llmpx_do_not_instrument_list),
            std::end(llmpx_do_not_instrument_list),
            str) != std::end(llmpx_do_not_instrument_list))
    {
        return true;
    }
    return false;
}

/*
 * helper function
 */
Instruction* GetNextInstruction(Instruction* I)
{
    if (isa<TerminatorInst>(I))
    {
        return I;
    }
    BasicBlock::iterator BBI(I);
    return dyn_cast<Instruction>(++BBI);
}

Instruction* GetNextNonPHIInstruction(Instruction* I)
{
    if (isa<TerminatorInst>(I))
    {
        return I;
    }
    BasicBlock::iterator BBI(I);
    while(isa<PHINode>(BBI))
    {
        ++BBI;
    }
    return dyn_cast<Instruction>(BBI);
}

/*
 * debug function
 */
void llmpx::dump_dbgstk()
{
    errs()<<ANSI_COLOR_GREEN<<"Process Stack:"<<ANSI_COLOR_RESET<<"\n";
    while(dbgstk.size())
    {
        errs()<<(dbgstk.size()-1)<<" : ";
        dbgstk.top()->dump();
        dbgstk.pop();
    }
    errs()<<ANSI_COLOR_GREEN<<"-------------"<<ANSI_COLOR_RESET<<"\n";
}


/*
 * ripoff tail call attribute
 */
void llmpx::ripoff_tail_call(Value* val)
{
    /*
    if(!isa<CallInst>(val))
    {
        return;
    }
    CallInst* callinst = dyn_cast<CallInst>(val);
    callinst->setTailCall(false);
    */
}


/*
 * helper function, load infinite/invalid bound for this function
 */
Value* llmpx::get_infinite_bound(Function* f)
{
    /*
     * if already have one, return it directly...
     */
    if (infinite_bnd_for_function.count(f)!=0)
    {
        return infinite_bnd_for_function[f];
    }

    Instruction* insertpoint 
        = dyn_cast<Instruction>(f->getEntryBlock().getFirstInsertionPt());

    std::list<Value*> ilist;
    Type* X86_BNDTy = Type::getX86_BNDTy(*ctx);
    PointerType* X86_BNDPtrTy = Type::getX86_BNDPtrTy(*ctx);

    IRBuilder<> builder(insertpoint);
    Value* bndptr = ConstantExpr::getPointerCast(bnd_infinite, X86_BNDPtrTy);

    Value* bnd = builder.CreateLoad(X86_BNDTy, bndptr, "inf_bnd");
    infinite_bnd_for_function[f] = bnd;
    return bnd;
}

Value* llmpx::get_infinite_bound(Instruction* i)
{
    Function* function = i->getParent()->getParent();
    
    return get_infinite_bound(function);
}
Value* llmpx::get_invalid_bound(Function* f)
{
    /*
     * if already have one, return it directly...
     */
    if (invalid_bnd_for_function.count(f)!=0)
    {
        return invalid_bnd_for_function[f];
    }

    Instruction* insertpoint 
        = dyn_cast<Instruction>(f->getEntryBlock().getFirstInsertionPt());

    std::list<Value*> ilist;
    Type* X86_BNDTy = Type::getX86_BNDTy(*ctx);
    PointerType* X86_BNDPtrTy = Type::getX86_BNDPtrTy(*ctx);

    IRBuilder<> builder(insertpoint);
    Value* bndptr = ConstantExpr::getPointerCast(bnd_invalid, X86_BNDPtrTy);

    Value* bnd = builder.CreateLoad(X86_BNDTy, bndptr, "inv_bnd");
    invalid_bnd_for_function[f] = bnd;
    return bnd;
}

Value* llmpx::get_invalid_bound(Instruction* i)
{
    Function* function = i->getParent()->getParent();
    
    return get_invalid_bound(function);
}

/*
 * helper function, load valid/invalid key for this function
 */
Value* llmpx::get_anyvalid_key(Function* f)
{
    /*
     * if already have one, return it directly...
     */
    if (anyvalid_key_for_function.count(f)!=0)
    {
        return anyvalid_key_for_function[f];
    }

    Instruction* insertpoint 
        = dyn_cast<Instruction>(f->getEntryBlock().getFirstInsertionPt());

    IRBuilder<> builder(insertpoint);

    Value* key = builder.CreateLoad(Type::getInt64Ty(*ctx), key_anyvalid, "key_anyvalid");
    anyvalid_key_for_function[f] = key;
    return key;
}

Value* llmpx::get_anyvalid_key(Instruction* i)
{
    Function* function = i->getParent()->getParent();
    return get_anyvalid_key(function);
}

Value* llmpx::get_anyinvalid_key(Function* f)
{
    /*
     * if already have one, return it directly...
     */
    if (anyinvalid_key_for_function.count(f)!=0)
    {
        return anyinvalid_key_for_function[f];
    }

    Instruction* insertpoint 
        = dyn_cast<Instruction>(f->getEntryBlock().getFirstInsertionPt());

    IRBuilder<> builder(insertpoint);

    Value* key = builder.CreateLoad(Type::getInt64Ty(*ctx), key_anyinvalid, "key_anyinvalid");
    anyinvalid_key_for_function[f] = key;
    return key;
}

Value* llmpx::get_anyinvalid_key(Instruction* i)
{
    Function* function = i->getParent()->getParent();
    return get_anyinvalid_key(function);
}


/*
 * create llmpx helper library symbols
 */
void llmpx::create_llmpx_symbols(Module& module)
{
    Type* Int8Ty = Type::getInt8Ty(*ctx);
    Type* Int64Ty = Type::getInt64Ty(*ctx);
    Type* VoidTy = Type::getVoidTy(*ctx);
    Type* VoidPtrTy = PointerType::getUnqual(Int8Ty);

    //func void test()
    {
    FunctionType* _llmpx_test_fntype
        = FunctionType::get(VoidTy,false);
    module.getOrInsertFunction(
            LLMPX_SYMBOL_PREFIX "test",
            _llmpx_test_fntype);
    _llmpx_test = module.getFunction(LLMPX_SYMBOL_PREFIX "test");
    assert(_llmpx_test
        && "wtf, _llmpx_test is null, this is impossible!");
    }

    //func void temporal_chk(VoidPtrTy, Int64Ty)
    {
    std::vector<Type*> plist;
    plist.push_back(VoidPtrTy);//ptr
    plist.push_back(Int64Ty);//key
    FunctionType* _llmpx_temporal_chk_fntype
        = FunctionType::get(VoidTy, plist, false);
    module.getOrInsertFunction(
        LLMPX_SYMBOL_PREFIX "temporal_chk",
        _llmpx_temporal_chk_fntype);
    _llmpx_temporal_chk
        = module.getFunction(LLMPX_SYMBOL_PREFIX "temporal_chk");
    assert(_llmpx_temporal_chk
        && "wtf, _llmpx_temporal_chk is null, this is impossible!");
    }
    //func Int64Ty temporal_lock_alloca(VoidPtrTy, Int64Ty)
    {
    std::vector<Type*> plist;
    plist.push_back(VoidPtrTy);//ptr
    plist.push_back(Int64Ty);//size
    FunctionType* _llmpx_temporal_lock_alloca_fntype
        = FunctionType::get(Int64Ty, plist, false);
    module.getOrInsertFunction(
        LLMPX_SYMBOL_PREFIX "temporal_lock_alloca",
        _llmpx_temporal_lock_alloca_fntype);
    _llmpx_temporal_lock_alloca
        = module.getFunction(LLMPX_SYMBOL_PREFIX "temporal_lock_alloca");
    assert(_llmpx_temporal_lock_alloca
        && "wtf, _llmpx_temporal_lock_alloca is null, this is impossible!");
    }
    //func void temporal_key_store(VoidPtrTy, Int64Ty)
    {
    std::vector<Type*> plist;
    plist.push_back(VoidPtrTy);//ptr
    plist.push_back(Int64Ty);//key
    FunctionType* _llmpx_temporal_key_store_fntype
        = FunctionType::get(VoidTy, plist, false);
    module.getOrInsertFunction(
        LLMPX_SYMBOL_PREFIX "temporal_key_store",
        _llmpx_temporal_key_store_fntype);
    _llmpx_temporal_key_store
        = module.getFunction(LLMPX_SYMBOL_PREFIX "temporal_key_store");
    assert(_llmpx_temporal_key_store
        && "wtf, _llmpx_temporal_key_store is null, this is impossible!");
    }
    //func Int64Ty temporal_key_load(VoidPtrTy)
    {
    std::vector<Type*> plist;
    plist.push_back(VoidPtrTy);//ptr
    FunctionType* _llmpx_temporal_key_load_fntype
        = FunctionType::get(Int64Ty, plist, false);
    module.getOrInsertFunction(
        LLMPX_SYMBOL_PREFIX "temporal_key_load",
        _llmpx_temporal_key_load_fntype);
    _llmpx_temporal_key_load
        = module.getFunction(LLMPX_SYMBOL_PREFIX "temporal_key_load");
    assert(_llmpx_temporal_key_load
        && "wtf, _llmpx_temporal_key_load is null, this is impossible!");
    }  
    //func void _llmpx_dbg_dump_bndldstx(VoidPtrTy, Int1Ty)
    {
    std::vector<Type*> plist;
    plist.push_back(VoidPtrTy);//ptr
    plist.push_back(Type::getInt1Ty(*ctx));
    FunctionType* _llmpx_dbg_dump_bndldstx_fntype
        = FunctionType::get(VoidTy, plist, false);
    module.getOrInsertFunction(
        LLMPX_SYMBOL_PREFIX "dbg_dump_bndldstx",
        _llmpx_dbg_dump_bndldstx_fntype);
    _llmpx_dbg_dump_bndldstx
        = module.getFunction(LLMPX_SYMBOL_PREFIX "dbg_dump_bndldstx");
    assert(_llmpx_dbg_dump_bndldstx
        && "wtf, _llmpx_dbg_dump_bndldstx is null, this is impossible!");
    }
}

/*
 * create mpx intrinsic symbols
 */
void llmpx::create_mpx_intr_symbols(Module& module)
{
    mpx_bndmk = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndmk);
    mpx_bndldx = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndldx);
    mpx_bndstx = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndstx);
    mpx_bndclrr = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndclrr);
    mpx_bndclrm = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndclrm);
    mpx_bndcurr = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndcurr);
    mpx_bndcurm = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndcurm);
    mpx_bndcn = Intrinsic::getDeclaration(&module,
                    Intrinsic::x86_bndcn);
}

/*
 * for bound cache symbols
 */
void llmpx::create_llmpx_bnd_cache_symbols(Module& module)
{
    Type* Int8Ty = Type::getInt8Ty(*ctx);
    Type* VoidPtrTy = PointerType::getUnqual(Int8Ty);
    /*
     * func VoidPtrTy _llmpx_bnd_cache_demand(VoidPtrTy, Int1Ty)
     * this function require two arguments:
     *  - address of pointer
     *  - load or store
     * and returns one result
     *  - cached address of pointer
     */
    {
    std::vector<Type*> plist;
    plist.push_back(VoidPtrTy);//address of pointer
    plist.push_back(Type::getInt1Ty(*ctx));//store or load
    FunctionType* _llmpx_bnd_cache_demand_fntype
        = FunctionType::get(VoidPtrTy, plist, false);
    module.getOrInsertFunction(
        LLMPX_SYMBOL_PREFIX "bnd_cache_demand",
        _llmpx_bnd_cache_demand_fntype);
    _llmpx_bnd_cache_demand
        = module.getFunction(LLMPX_SYMBOL_PREFIX "bnd_cache_demand");
    assert(_llmpx_bnd_cache_demand
        && "wtf, _llmpx_bnd_cache_demand is null, this is impossible!");
    }
}

/*
 * create symbols for wrapper function in mpxwrap
 * NOTE: mpx_wrapper only support spatial check for now,
 * so its simpler than ::transform_functions()
 */
void llmpx::create_llmpx_wrapper_symbols(Module& module)
{
#if DEBUG_WRAPPER_PROCESSOR
    errs()<<ANSI_COLOR_YELLOW
        <<"====create_llmpx_wrapper_symbols===="
        <<ANSI_COLOR_RESET
        <<"\n";
#endif
    for (auto it = std::begin(llmpx_func_wrapper_list),
              end = std::end(llmpx_func_wrapper_list);
              it!=end; ++it)
    {
        Function* func_ptr = module.getFunction(*it);
        if(!func_ptr)
            continue;

#if DEBUG_WRAPPER_PROCESSOR
        errs()<<ANSI_COLOR_RED<<" * ";
        func_ptr->dump();
        errs()<<ANSI_COLOR_RESET;
#endif
        FunctionType* func_type = func_ptr->getFunctionType();
        assert(func_type && "function type is not available!?");
        /*
         * create wrapper function and insert it into function list
         */
        Function* wrapper_func
            = Function::Create(func_type,
                                func_ptr->getLinkage(),
                                std::string(LLMPX_WRAPPER_PREFIX) + *it);
        wrapper_func->setAttributes(func_ptr->getAttributes());
        func_ptr->getParent()
                ->getFunctionList()
                    .insert(Module::iterator(func_ptr), wrapper_func);
#if DEBUG_WRAPPER_PROCESSOR
        errs()<<ANSI_COLOR_GREEN<<" -> ";
        wrapper_func->dump();
        errs()<<ANSI_COLOR_RESET;
#endif
        /*
         * construct its wrapper's actual type
         */
        //first, the return type
        Type* orig_ret_type = func_ptr->getReturnType();
        Type* ret_type = func_type->getReturnType();
        if (orig_ret_type->isPointerTy())
        {
            tr_flist_ret[func_ptr] = 1;
            if (has_transformed_type_for_wrap(orig_ret_type))
            {
                ret_type = get_transformed_type_for_wrap(orig_ret_type);
            }else
            {
                std::vector<Type*> assemble_new_type;
                assemble_new_type.push_back(orig_ret_type);
                Type* x86bndty = Type::getX86_BNDTy(*ctx);
                for (int bi = 0; bi < tr_flist_ret[func_ptr]; bi++)
                {
                    assemble_new_type.push_back(x86bndty);
                }
                Type* new_ret_type = StructType::create(*ctx, assemble_new_type);
                ret_type = new_ret_type;
                add_transform_type_pair_for_wrap(orig_ret_type, new_ret_type);
            }
        }
        //arguments
        int arg_index = 1;
        int ptr_arg_cnt = 0;
        std::vector<Type*> params;
        for (Function::arg_iterator i = func_ptr->arg_begin(),
                    e = func_ptr->arg_end();
                    i != e; ++i, ++arg_index)
        {
            params.push_back(i->getType());
            bool arg_is_pointer = false;
            if(isa<PointerType>(i->getType()))
            {
                arg_is_pointer = true;
                ptr_arg_cnt++;
            }
        }
        unsigned orig_param_cnt = params.size();
        /*
         * append bound to argument list
         */
        //the number of bound/key need to be added
        for (int bndi = 0; bndi<ptr_arg_cnt; bndi++)
        {
            params.push_back(Type::getX86_BNDTy(*ctx));
        }

        /*
         * create the new function,
         * which should be its actual proto
         * and will be casted from wrapper function type
         */
        FunctionType* new_func_type 
            = FunctionType::get(ret_type,
                                params,
                                func_type->isVarArg());
        //Create cast from wrapper_func to function of new_func_type
        Value* new_func
            = ConstantExpr::getPointerCast(wrapper_func, new_func_type->getPointerTo());
#if DEBUG_WRAPPER_PROCESSOR
        errs()<<ANSI_COLOR_BLUE<<"F: ";
        new_func->dump();
        errs()<<ANSI_COLOR_RESET;
#endif
        orig_to_cw_flist[func_ptr] = new_func;
    }
#if DEBUG_WRAPPER_PROCESSOR
    errs()<<ANSI_COLOR_YELLOW
        <<"====done adding wrapper functions===="
        <<ANSI_COLOR_RESET
        <<"\n";
#endif
}

/*
 * create global constants: constant bound and key/lock
 */
void llmpx::create_global_constants(Module& module)
{
    /*
     * constant bound
     */
    auto* ArrayTy = ArrayType::get(IntegerType::get(*ctx, 64), 2);

    bnd_infinite 
        = new GlobalVariable(module,
                    ArrayTy,
                    false,
                    GlobalValue::LinkOnceAnyLinkage,
                    0,"llmpx_bnd_infinite");
    bnd_infinite->setAlignment(16);
    std::vector<uint64_t> infdata;
    infdata.push_back(0);
    infdata.push_back(0);
    Constant* bnd_inf_data = ConstantDataArray::get(*ctx,infdata);
    bnd_infinite->setInitializer(bnd_inf_data);
    
    bnd_invalid
        = new GlobalVariable(module,
                    ArrayTy,
                    false,
                    GlobalValue::LinkOnceAnyLinkage,
                    0,"llmpx_bnd_invalid");
    bnd_invalid->setAlignment(16);
    std::vector<uint64_t> invdata;
    invdata.push_back(~0ULL);
    invdata.push_back(~0ULL);
    Constant* bnd_inv_data = ConstantDataArray::get(*ctx, invdata);
    bnd_invalid->setInitializer(bnd_inv_data);

    /*
     * constant key
     */
    auto* Int64Ty = Type::getInt64Ty(*ctx);
    key_anyvalid
        = new GlobalVariable(module, Int64Ty,
                    false, GlobalValue::LinkOnceAnyLinkage,
                    0, "llmpx_key_anyvalid");
    key_anyvalid->setAlignment(8);
    key_anyvalid->setInitializer(ConstantInt::get(Int64Ty, 0));

    key_anyinvalid
        = new GlobalVariable(module, Int64Ty,
                    false, GlobalVariable::LinkOnceAnyLinkage,
                    0, "llmpx_key_anyinvalid");
    key_anyinvalid->setAlignment(8);
    key_anyinvalid->setInitializer(ConstantInt::get(Int64Ty, ~0));
}

/*
 * stub function
 */
bool llmpx::runOnModule(Module &module)
{
    this->module = &module;
    ctx = &module.getContext();
    //prepare global constant bound
    create_global_constants(module);
    /*
     * create mpx intrinsic symbols
     */
    create_mpx_intr_symbols(module);
    /*
     * create symbols for external help library
     */
    create_llmpx_symbols(module);
    /*
     * create symbols for wrapper functions in mpxwrap
     */
    create_llmpx_wrapper_symbols(module);
    /*
     * create symbol for bound cache functions
     */
    create_llmpx_bnd_cache_symbols(module);
#if USE_MPX_TESTER
    return mpxTester(module);
#else
    return mpxPass(module);;
#endif
}

void llmpx::add_instruction_to_bcl(Value* I)
{
    if (!I)
    {
        llvm_unreachable("adding NULL to bound_checklist??");
    }
    if (bound_checklist.find(I)==bound_checklist.end())
        bound_checklist[I] = new std::list<Value*>();

    if (llmpx_enable_temporal_safety)
    {
        if(key_checklist.find(I)==key_checklist.end())
        {
            key_checklist[I] = new std::list<Value*>();
        }
    }
}

/*
 * helper function
 */

bool llmpx::has_transformed_type(Type* orig_type)
{
    if (tr_typelist.find(orig_type)!=tr_typelist.end())
    {
        return true;
    }
    return false;
}

/*
 * get transformed type, for return type
 * This function operand on tr_typelist
 */
Type* llmpx::get_transformed_type(Type* orig_type)
{
    if (tr_typelist.find(orig_type)!=tr_typelist.end())
    {
        return tr_typelist[orig_type];
    }
    return orig_type;
}

void llmpx::add_transform_type_pair(Type* orig_type, Type* transformed_type)
{
    if (has_transformed_type(orig_type))
        return;
    tr_typelist[orig_type] = transformed_type;
}

//same thing for mpx wrap
bool llmpx::has_transformed_type_for_wrap(Type* orig_type)
{
    if (tr_typelist_for_wrap.find(orig_type)!=tr_typelist_for_wrap.end())
    {
        return true;
    }
    return false;
}

Type* llmpx::get_transformed_type_for_wrap(Type* orig_type)
{
    if (tr_typelist_for_wrap.find(orig_type)!=tr_typelist_for_wrap.end())
    {
        return tr_typelist_for_wrap[orig_type];
    }
    return orig_type;
}

void llmpx::add_transform_type_pair_for_wrap(Type* orig_type, Type* transformed_type)
{
    if (has_transformed_type_for_wrap(orig_type))
        return;
    tr_typelist_for_wrap[orig_type] = transformed_type;
}
/*
 * is the function in original function need tranformation?
 */
bool llmpx::is_function_orig(Value* ptr)
{
    if (ptr==NULL)
        return false;
    if (!isa<Function>(ptr))
        return false;

    Function* func_ptr = dyn_cast<Function>(ptr);

    if (tr_flist.find(func_ptr)!=tr_flist.end())
    {
        return true;
    }
    return false;
}

/*
 * is the function a transformed function?
 */
bool llmpx::is_function_trans(Value* ptr)
{
    if (ptr==NULL)
        return false;
    if (!isa<Function>(ptr))
        return false;

    Function* func_ptr = dyn_cast<Function>(ptr);
    if (revtr_flist.find(func_ptr)!=revtr_flist.end())
    {
        return true;
    }
    return false;
}

/*
 * find and return transformed function,
 * otherwise return itself
 */
Function* llmpx::find_transformed_function(Value* func)
{
    Function* func_ptr = dyn_cast<Function>(func);

    //errs()<<"want to find transformed function using : "
    //    <<func_ptr->getName()<<"\n";
    Function* new_func_ptr = func_ptr;
    if (is_function_orig(func_ptr))
    {
        new_func_ptr = tr_flist[func_ptr];
    }
    if (is_function_trans(func_ptr))
    {
        new_func_ptr = func_ptr;
        Function* old_func_ptr = revtr_flist[func_ptr];
        func_ptr = old_func_ptr;
    }
    assert((func_ptr!=NULL)&& "no way the result function is null #1");
    //errs()<<"The transformed function is : "
    //    <<new_func_ptr->getName()
    //    <<"\n";
    return new_func_ptr;
}

/*
 * find and return original function,
 * otherwise return itself
 */
Function* llmpx::find_nontransformed_function(Value* func)
{
    Function* func_ptr = dyn_cast<Function>(func);
    //errs()<<"want to find orig function using : "
    //        <<func_ptr->getName()<<"\n";
    Function* new_func_ptr = func_ptr;
    if (is_function_orig(func_ptr))
    {
        new_func_ptr = tr_flist[func_ptr];
    }
    if (is_function_trans(func_ptr))
    {
        new_func_ptr = func_ptr;
        Function* old_func_ptr = revtr_flist[func_ptr];
        func_ptr = old_func_ptr;
    }
    assert((func_ptr!=NULL)&& "no way the result function is null #2");
    //errs()<<"The orig function is : "
    //    <<func_ptr->getName()
    //    <<"\n";
    return func_ptr;
}

/*
 * associate meta of V to I
 * bound is located at 0
 * key is located at 1
 */
std::list<Value*>
llmpx::associate_meta(Instruction* I, Value* V)
{
    std::list<Value*> ilist;
    Value* bnd;
    Value* key;
    if(isa<GlobalVariable>(V))
    {
        if (gv_bound_checklist.find(V)==gv_bound_checklist.end())
        {
            if(dyn_cast<GlobalVariable>(V)->getLinkage()
                        !=GlobalValue::ExternalLinkage)
            {
                V->dump();
                llvm_unreachable("non-external global not found"
                                    " in gv_bound_checklist?");
            }
            ilist.push_back(get_infinite_bound(I));
            ilist.push_back(get_anyvalid_key(I));
            goto end;
        }
        bnd = gv_bound_checklist[V]->back();
        if(llmpx_enable_temporal_safety)
        {
            key = gv_key_checklist[V]->back();
        }
    }else
    {
        V = process_each_instruction(V);
        bnd = get_bound(V, I);
        if(llmpx_enable_temporal_safety)
        {
            key = get_key(V, I);
        }
    }
    if (!bnd)
    {
        bnd = get_infinite_bound(I);
    }
    ilist.push_back(bnd);

    if (llmpx_enable_temporal_safety)
    {
        if (!key)
        {
            key = get_anyvalid_key(I);
        }
        ilist.push_back(key);
    }
end:
    return ilist;
}

/*
 * insert bndldx ptr before Instruction I
 */
std::list<Value*>
llmpx::insert_bound_load(Instruction *I, Value* ptr, Value* ptrval)
{
    TotalBNDLDXAdded++;

    std::list<Value*> ilist;
    
    Value* addr = ptr;

    if (!ptrval->getType()->isPointerTy())
    {
        ptrval->dump();
        dump_dbgstk();
        llvm_unreachable("insert_bound_load: This is nasty, ptrval is not ptr?");
    }
    /*
     * need to cast ptr to desired type?
     */
    PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
    IRBuilder<> builder(I);
    if (ptr->getType()!=Int8PtrTy)
    {
        std::string addr_name = "castptr";
        if(ptr->hasName())
        {
            addr_name += ".";
            addr_name += ptr->getName().data();
        }
        addr = builder.CreateBitCast(ptr,Int8PtrTy,addr_name);
        ilist.push_back(dyn_cast<Instruction>(addr));
    }
    if (ptrval->getType()!=Int8PtrTy)
    {
        std::string addr_name = "castptrval";
        if(ptrval->hasName())
        {
            addr_name += ".";
            addr_name += ptrval->getName().data();
        }
        ptrval = builder.CreateBitCast(ptrval,Int8PtrTy,addr_name);
        ilist.push_back(dyn_cast<Instruction>(ptrval));
    }
    /*
     * if bound cache is enabled, get the cache line
     */
    if (llmpx_bound_cache)
    {
        std::vector<Value*> bnd_cache_args;
        bnd_cache_args.push_back(addr);
        bnd_cache_args.push_back(ConstantInt::getTrue(*ctx));
        Instruction* bnd_cache_addr
            = CallInst::Create(_llmpx_bnd_cache_demand, bnd_cache_args, "", I);
        bnd_cache_addr->setDebugLoc(I->getDebugLoc());
        addr = bnd_cache_addr;
    }
    /*
     * insert bndldx
     */
    insert_dbg_dump_bndldstx(I, addr, true);
    std::vector<Value *> args;

    args.push_back(addr);
    args.push_back(ptrval);
    if (llmpx_use_ppa)
    {
        int setid = get_aa_set_id(addr);
        /*
         * otherwise need to shift address to somewhere else,
         * FIXME: make sure we don't have collision
         */
        args.push_back(
            ConstantInt::get(Type::getInt32Ty(*ctx), (1<<27)*setid));
    }else
    {
        args.push_back(ConstantInt::get(Type::getInt32Ty(*ctx), 0));
    }

    Instruction* bndldx = CallInst::Create(mpx_bndldx, args, "", I);
    ilist.push_back(bndldx);

    bndldxlist.push_back(bndldx);

    return ilist;
}
/*
 * insert key load for ptr before Instruction I
 */
std::list<Value*>
llmpx::insert_key_load(Instruction *I, Value* ptr)
{
#if DEBUG_KEY_STORE
    errs()<<" insert_key_load:\n";
    errs()<<"   I - ";
    I->print(errs());
    errs()<<"\n   ptr - ";
    ptr->print(errs());
    errs()<<"\n";
#endif

    std::list<Value*> ilist;

    Type* Int8Ty = Type::getInt8Ty(*ctx);
    Type* VoidPtrTy = PointerType::getUnqual(Int8Ty);
    IRBuilder<> builder0(I);
    Value* addr = builder0.CreateBitCast(ptr, VoidPtrTy);

    /*
     * insert key load
     */
    std::vector<Value *> args;
    args.push_back(addr);
    Instruction* keyldx
        = CallInst::Create(_llmpx_temporal_key_load, args,
                            "keyldx", I);
    ilist.push_back(keyldx);
    return ilist;
}

/*
 * insert bndstx using ptr and it's bnd after Instruction I
 */
std::list<Value*>
llmpx::insert_bound_store(Instruction* I, Value *ptr, Value* ptrval, Value* bnd)
{

#if 0
    errs()<<" insert_bound_store:\n";
    errs()<<"   I - ";
    I->print(errs());
    errs()<<"\n   ptr - ";
    ptr->print(errs());
    errs()<<"\n   bnd - ";
    bnd->print(errs());
    errs()<<"\n";
#endif
    std::list<Value*> ilist;
    Instruction* insertPoint = GetNextInstruction(I);
    /*
     * need to cast ptr to desired type?
     */
    IRBuilder<> builder(insertPoint);
    PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
    Value* addr = ptr;
    if (ptr->getType()!=Int8PtrTy)
    {
        IRBuilder<> builder(insertPoint);
        addr = builder.CreateBitCast(ptr,Int8PtrTy,"");
        Instruction* addr_inst = dyn_cast<Instruction>(addr);
        if (addr_inst!=NULL)
            insertPoint = GetNextInstruction(addr_inst);
        ilist.push_back(addr);
    }
    if (ptrval->getType()!=Int8PtrTy)
    {
        if(ptrval->getType()->isPointerTy())
        {
            ptrval = builder.CreateBitCast(ptrval,Int8PtrTy,"");
        }else
        {
            ptrval = builder.CreateIntToPtr(ptrval, Int8PtrTy);
        }
        Instruction* ptrval_inst = dyn_cast<Instruction>(ptrval);
        if (ptrval_inst!=NULL)
            insertPoint = GetNextInstruction(ptrval_inst);
        ilist.push_back(ptrval);
    }
    //errs()<<"      - Create bndstx\n";
    /*
     * if bound cache is enabled, get the cache line
     */
    if (llmpx_bound_cache)
    {
        std::vector<Value*> bnd_cache_args;
        bnd_cache_args.push_back(addr);
        bnd_cache_args.push_back(ConstantInt::getFalse(*ctx));
        Instruction* bnd_cache_addr
            = CallInst::Create(_llmpx_bnd_cache_demand,
                                bnd_cache_args, "", insertPoint);
        bnd_cache_addr->setDebugLoc(I->getDebugLoc());
        addr = bnd_cache_addr;
    }
    /*
     * skip infinite bound store
     */
    if (bnd->hasName() && bnd->getName()=="inf_bnd")
    {
        return ilist;
    }
    /*
     * insert bndstx
     */
    TotalBNDSTXAdded++;
    std::vector<Value *> args;
    args.push_back(addr);
    args.push_back(ptrval);
    if (llmpx_use_ppa)
    {
        int setid = get_aa_set_id(addr);
        /*
         * otherwise need to shift address to somewhere else,
         * FIXME: make sure we don't have collision
         */
        args.push_back(
            ConstantInt::get(Type::getInt32Ty(*ctx), (1<<27)*setid));
    }else
    {
        args.push_back(ConstantInt::get(Type::getInt32Ty(*ctx), 0));
    }
    args.push_back(bnd);
   
    Instruction* bndstx 
            = CallInst::Create(mpx_bndstx, args, "", insertPoint);
    ilist.push_back(bndstx);

    insert_dbg_dump_bndldstx(bndstx, addr, false);

    bndstxlist.push_back(bndstx);

    return ilist;
}

/*
 * insert key store for ptr and it's key after Instruction I
 */
std::list<Value*>
llmpx::insert_key_store(Instruction* I, Value *ptr, Value* key)
{
#if DEBUG_KEY_STORE
    errs()<<" insert_key_store:\n";
    errs()<<"   I - ";
    I->print(errs());
    errs()<<"\n   ptr - ";
    ptr->print(errs());
    errs()<<"\n   key - ";
    key->print(errs());
    errs()<<"\n";
#endif
    std::list<Value*> ilist;
    Instruction* insertPoint = GetNextInstruction(I);

    //cast to void*
    Type* Int8Ty = Type::getInt8Ty(*ctx);
    Type* VoidPtrTy = PointerType::getUnqual(Int8Ty);
    IRBuilder<> builder0(I);
    Value* addr = builder0.CreateBitCast(ptr, VoidPtrTy);

    /*
     * insert key store
     */
    std::vector<Value *> args;
    args.push_back(addr);
    args.push_back(key);
   
    Instruction* keystx
            = CallInst::Create(_llmpx_temporal_key_store, args,
                            "", insertPoint);
    ilist.push_back(keystx);
    return ilist;
}

/*
 * insert debug function to dump bndldx/bndstx info
 */
std::list<Value*>
llmpx::insert_dbg_dump_bndldstx(Instruction *I, Value* ptr, bool is_load)
{
    std::list<Value*> ilist;
    if(!llmpx_dump_bndldstx)
    {
        return ilist;
    }

    std::vector<Value *> args;
    args.push_back(ptr);
    args.push_back(is_load?
                    ConstantInt::getTrue(*ctx)
                    :ConstantInt::getFalse(*ctx));
   
    Instruction* dbg_dump_bndldstx_call
            = CallInst::Create(_llmpx_dbg_dump_bndldstx, args, "", I);
    ilist.push_back(dbg_dump_bndldstx_call);
    return ilist;
}

/*
 * insert bound checks for instruction I,
 * which want to access memroy pointed by ptr,
 * and the bound information is hold by bnd
 * all instructions are inserted before I
 * NOTE: new ptr is returned at index 0 incase that it might be
 *       changed by process_each_instruction()
 * The check should be like this:
 *      bndcl ptr, bnd0
 *      bndcu ptr+size-1, bnd0
 *
 * FIXME: Do we need to retrive the up-to-date bound?
 *        it's better to do some liveness analysis, to see whether the bound
 *        have been re-written, so that we don't need to load the bound
 *        from bt again? or at least we need to reload the ptrval to get
 *        the right bound
 */
std::list<Value*>
llmpx::insert_bound_check(Instruction *I, Value* ptr, bool is_load)
{
    TotalChecksAdded++;
#if DEBUG_INSERT_BOUND_CHECK
    errs()<<"  = insert_bound_check\n";
    errs()<<"     + ptr:";
    ptr->print(errs());
    errs()<<"\n";
#endif
    std::list<Value*> ilist;
    //get its bound
    Value* addr = ptr;
    Value* bnd;
    if (isa<ConstantExpr>(addr))
    {
        ConstantExpr* constant
            = dyn_cast<ConstantExpr>(ptr);
        Instruction* inst
            = constant->getAsInstruction();
        /*if (isa<GetElementPtrInst>(inst))
        {
            addr = dyn_cast<GetElementPtrInst>(inst)->getPointerOperand();
        }*/
        delete inst;
    }
    if(isa<GlobalVariable>(addr))
    {
#if 1
        //errs()<<"global bound:";
        std::list<Value*>* blist = gv_bound_checklist[addr];
        if (blist==NULL)
        {
            errs()<<"llmpx::insert_bound_check(), global bound is null?\n";
            addr->print(errs());
            errs()<<"\n";
            return ilist;
        }
        Value* raw_gvbnd = blist->back();
        //need to load global bound
        Type* X86_BNDTy = Type::getX86_BNDTy(*ctx);
        PointerType* X86_BNDPtrTy = Type::getX86_BNDPtrTy(*ctx);
        IRBuilder<> builder(I);
        Value* gvbnd_ptr = builder.CreateBitCast(raw_gvbnd, X86_BNDPtrTy);
        bnd = builder.CreateLoad(X86_BNDTy, gvbnd_ptr, "");
        //recover original pointer
        addr = ptr;
#else
        //no check for global variable?
        return ilist;
#endif
    }else
    {
        //errs()<<"bound:";
        addr = process_each_instruction(addr);
        bnd = get_bound(addr, I);
    }
    //new ptr need to be returned?
    ilist.push_back(addr);

    /*
     * no bound ?
     * or infinite bound, don't bother check
     */
    if ((!bnd)
        || ((bnd->hasName() &&
            (bnd->getName()=="inf_bnd"))))
    {
#if 0
        //can not find bnd, insert check against inf bnd anyway
        bnd = get_infinite_bound(I);
        ilist.push_back(bnd);
#else
        //can not find bnd won't do the check
        TotalChecksAdded--;
        return ilist;
#endif
    }
    //bnd->print(errs());
    //errs()<<"\n";
    /*
     * respect command line options
     */
    if (is_load && llmpx_store_only_check)
    {
        TotalChecksAdded--;
        return ilist;
    }
    Value* base;
    Value* index;
    Value* scale;
    int disp;
    /*
     * need to cast ptr to desired type?
     */
    IRBuilder<> builder(I);
    PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
    std::string addr_basename = "";
    if (addr->hasName())
    {
        addr_basename = addr->getName().data();
    }
    /*
     * insert bndcl
     * use base+displace if possible
     */
    if (isa<GetElementPtrInst>(addr) || isa<ConstantExpr>(addr))
    {
        GetElementPtrInst* gep;
        if (isa<GetElementPtrInst>(addr))
        {
            gep = dyn_cast<GetElementPtrInst>(addr);
        }else
        {
            ConstantExpr* cexpr = dyn_cast<ConstantExpr>(addr);
            Instruction* cexpr_inst = cexpr->getAsInstruction();
            gep = dyn_cast<GetElementPtrInst>(cexpr_inst);
            if (gep==NULL)
            {
                delete cexpr_inst;
                goto use_lea;
            }
        }
        const DataLayout &dl = cfunc->getParent()->getDataLayout();
        APInt offset(dl.getTypeStoreSizeInBits(gep->getType()),0,false);
        if (!gep->accumulateConstantOffset(dl , offset))
        {
            goto use_lea;
        }
        base = gep->getPointerOperand();
        if (base->getType()!=Int8PtrTy)
        {
            std::string addr_name = "base."+ addr_basename;
            base = builder.CreateBitCast(base, Int8PtrTy, addr_name);
            ilist.push_back(dyn_cast<Instruction>(base));
        }
        index = ConstantInt::get(Type::getInt64Ty(*ctx),0);
        scale = ConstantInt::get(Type::getInt8Ty(*ctx),1);
        disp = offset.getSExtValue();

        std::vector<Value *> args4;
        args4.push_back(bnd);
        args4.push_back(base);
        args4.push_back(index);//index
        args4.push_back(scale);//scale
        args4.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),disp));//disp

        Instruction* bndcl = builder.CreateCall(mpx_bndclrm, args4);
        ilist.push_back(bndcl);
        if (gep!=addr)
        {
            delete gep;
        }
    }else
    {
use_lea:
        if (addr->getType()!=Int8PtrTy)
        {
            std::string addr_name = "base." + addr_basename;
            addr = builder.CreateBitCast(addr,Int8PtrTy,addr_name);
            ilist.push_back(dyn_cast<Instruction>(addr));
        }
        std::vector<Value *> args4;
        args4.push_back(bnd);
        args4.push_back(addr);
        Instruction* bndcl = builder.CreateCall(mpx_bndclrr, args4);
        ilist.push_back(bndcl);
        base = addr;
        index = ConstantInt::get(Type::getInt64Ty(*ctx),0);
        scale = ConstantInt::get(Type::getInt8Ty(*ctx),1);
        disp = 0;
    }
    /*
     * insert bndcu
     */
    std::vector<Value *> args5;
    args5.push_back(bnd);
    uint64_t rwsize
        = module->getDataLayout()
             .getTypeStoreSizeInBits(ptr->getType()->getContainedType(0))/8;
    args5.push_back(base);//base
    args5.push_back(index);//index
    args5.push_back(scale);//scale
    args5.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),disp + rwsize -1));//disp

    Instruction* bndcu = builder.CreateCall(mpx_bndcurm, args5);
    ilist.push_back(bndcu);

    return ilist;
}

/*
 * insert key lock check for temporal safety
 */
std::list<Value*>
llmpx::insert_keylock_check(Instruction *I, Value* ptr, bool is_load)
{
    std::list<Value*> ilist;
    if (!llmpx_enable_temporal_safety)
    {
        return ilist;
    }
    if (is_load && llmpx_store_only_check)
    {
        return ilist;
    }

    Value* key = get_key(ptr, I);
    if (!key)
    {
        key = get_anyvalid_key(I);
    }

    Type* Int8Ty = Type::getInt8Ty(*ctx);
    Type* Int64Ty = Type::getInt64Ty(*ctx);
    Type* VoidPtrTy = PointerType::getUnqual(Int8Ty);

    IRBuilder<> builder0(I);
 
    std::vector<Value*> args;
    Value* voidptr = builder0.CreateBitCast(ptr, VoidPtrTy);
    args.push_back(voidptr);
    args.push_back(key);

    Value* func_call = builder0.CreateCall(_llmpx_temporal_chk, args);
    ilist.push_back(func_call);

    return ilist;
}

/*
 * insert check, a wrapper
 */
std::list<Value*>
llmpx::insert_check(Instruction *I, Value* ptr, bool is_load)
{
    std::list<Value*> ilist;
    if (llmpx_no_check)
        return ilist;

    ilist = insert_bound_check(I, ptr, is_load);
    ptr = ilist.front();//just in case that ptr might be changed
    if (llmpx_enable_temporal_safety)
    {
        std::list<Value*> ilist2
            = insert_keylock_check(I, ptr, is_load);
        ilist.splice(ilist.end(), ilist2);
    }
    return ilist;
}


/*
 * get bound for v, given that there's a bound
 * if any instruction need to be inserted, should be inserted before I
 */
Value* llmpx::get_bound(Value* v, Instruction* I)
{
    if (v==NULL)
    {
        I->print(errs());
        errs()<<"\n";
        llvm_unreachable("no way for get_bound(NULL)");
    }

    if (isa<ConstantExpr>(v))
    {
        std::list<Value*>* cblist;
        if (gv_bound_checklist_cache.find(std::make_pair(v,I))
            ==gv_bound_checklist_cache.end())
        {
            cblist = new std::list<Value*>();
            gv_bound_checklist_cache[std::make_pair(v,I)] = cblist;
        }else
        {
            cblist
                = gv_bound_checklist_cache[std::make_pair(v,I)];
            return cblist->back();
        }

        ConstantExpr* constant
            = dyn_cast<ConstantExpr>(v);
        Instruction* inst
            = constant->getAsInstruction();
        Value* addr;
        if(isa<GetElementPtrInst>(inst))
        {
            addr = dyn_cast<GetElementPtrInst>(inst)->getPointerOperand();
        }else if(isa<BitCastInst>(inst))
        {
            addr = dyn_cast<BitCastInst>(inst)->getOperand(0);
        }else if(isa<IntToPtrInst>(inst))
        {
            delete inst;
            return NULL;
        }else if(isa<PtrToIntInst>(inst))
        {
            addr = dyn_cast<PtrToIntInst>(inst)->getPointerOperand();
        }else
        {
            inst->print(errs());
            errs()<<"\n";
            llvm_unreachable("unhandled inst in get_bound for constant\n");
        }
        if (!isa<GlobalVariable>(addr))
        {
            Value* bnd = get_bound(addr, I);
            delete inst;
            return bnd;
            //llvm_unreachable("unhandled get_bound for non-global constant\n");
        }
        if (gv_bound_checklist.find(addr)==gv_bound_checklist.end())
        {
            if(dyn_cast<GlobalVariable>(addr)->getLinkage()
                != GlobalValue::ExternalLinkage)
            {
                addr->print(errs());
                llvm_unreachable("non-external global not found"
                                    " in gv_bound_checklist?");
            }
            delete inst;
            return NULL;
        }
        std::list<Value*>* blist = gv_bound_checklist[addr];
        Value* raw_gvbnd = blist->back();
        Type* X86_BNDTy = Type::getX86_BNDTy(*ctx);
        PointerType* X86_BNDPtrTy = Type::getX86_BNDPtrTy(*ctx);
        IRBuilder<> builder(I);
        Value* gvbnd_ptr = builder.CreateBitCast(raw_gvbnd, X86_BNDPtrTy);
        Value* bnd = builder.CreateLoad(X86_BNDTy, gvbnd_ptr, "");

        cblist->push_back(bnd);
        delete inst;
        return bnd;
    }

    //not ConstantExpr...
    if (bound_checklist.find(v)==bound_checklist.end())
    {
        return NULL;
    }
    std::list<Value*>* blist = bound_checklist[v];
    if (!blist)
    {
        return NULL;
    }
    if(blist==&delete_ii)
    {
        return NULL;
    }
    if(blist->size()==0)
    {
        return NULL;
    }
    Value* val = blist->back();
    //the bound is already tied with it
    if (!val)
    {
        return NULL;
    }
    if (val->getType() && val->getType()->isX86_BNDTy())
    {
        return val;
    }
    //otherwise need to find bound in its parent
    return NULL;
}

/*
 * get key for v, given that there's a key
 * if any instruction need to be inserted, should be inserted before I
 */
Value* llmpx::get_key(Value* v, Instruction* I)
{
    if (v==NULL)
    {
        I->print(errs());
        errs()<<"\n";
        llvm_unreachable("no way for get_key(NULL)");
    }
    if (isa<ConstantExpr>(v))
    {
        std::list<Value*>* cklist;
        if (gv_key_checklist_cache.find(std::make_pair(v,I))
            == gv_key_checklist_cache.end())
        {
            cklist = new std::list<Value*>();
            gv_key_checklist_cache[std::make_pair(v,I)] = cklist;
        }else
        {
            cklist = gv_key_checklist_cache[std::make_pair(v,I)];
            return cklist->back();
        }

        ConstantExpr* constant
            = dyn_cast<ConstantExpr>(v);
        Instruction* inst
            = constant->getAsInstruction();
        Value* addr;
        if(isa<GetElementPtrInst>(inst))
        {
            addr = dyn_cast<GetElementPtrInst>(inst)->getPointerOperand();
        }else if(isa<BitCastInst>(inst))
        {
            addr = dyn_cast<BitCastInst>(inst)->getOperand(0);
        }else if(isa<IntToPtrInst>(inst))
        {
            delete inst;
            return NULL;
        }else if (isa<PtrToIntInst>(inst))
        {
            addr = dyn_cast<PtrToIntInst>(inst)->getOperand(0);
        }else
        {
            inst->print(errs());
            errs()<<"\n";
            llvm_unreachable("unhandled inst in get_key for constant\n");
        }
        if (!isa<GlobalVariable>(addr))
        {
            Value* key = get_key(addr, I);
            delete inst;
            return key;
        }
        if (gv_key_checklist.find(addr)==gv_key_checklist.end())
        {
            if(dyn_cast<GlobalVariable>(addr)->getLinkage()
                != GlobalValue::ExternalLinkage)
            {
                addr->print(errs());
                llvm_unreachable("non-external global not found"
                                    " in gv_key_checklist?");
            }
            delete inst;
            return NULL;
        }

        std::list<Value*>* klist = gv_key_checklist[addr];
        Value* raw_gvkey = klist->back();
        IRBuilder<> builder(I);
        Value* gvkey_ptr
            = builder.CreateBitCast(raw_gvkey, Type::getInt64PtrTy(*ctx));
        Value* key
            = builder.CreateLoad(Type::getInt64Ty(*ctx), gvkey_ptr, "");

        cklist->push_back(key);
        delete inst;
        return key;
    }

    //not ConstantExpr...
    if(key_checklist.find(v)==key_checklist.end())
    {
        return NULL;
    }
    std::list<Value*>* klist = key_checklist[v];
    if (!klist)
    {
        return NULL;
    }
    if(klist==&delete_ii)
    {
        return NULL;
    }
    if(klist->size()==0)
    {
        return NULL;
    }
    Value* val = klist->back();
    //the key is already tied with it
    if (!val)
    {
        return NULL;
    }
    if (val->getType() && val->getType()->isIntegerTy())
    {
        return val;
    }
    //otherwise need to find key in its parent
    return NULL;
}

/*
 * see if the function need to be transformed for passing bound
 * information
 *  - arguments
 *  - return value
 * The number of bound need to be added to return value is collected
 * in variable 
 */
bool llmpx::function_need_to_be_transformed(Function* func_ptr)
{
    if (is_in_skip_list(func_ptr->getName()))
    {
        errs()<<"Skip "<<func_ptr->getName()<<"\n";
        return false;
    }
    /*
     * examine return value type
     *   - pointer type
     *   - aggregate type who contains pointer 
     */
    //errs()<<" Function: "<<func_ptr->getName()<<"\n";
    Type *ret_type = func_ptr->getFunctionType()->getReturnType();
    //errs()<<"     - return type:";
    //ret_type->print(errs());
    //errs()<<"\n";
    //can not handle vaarg for now.
    if (func_ptr->getFunctionType()->isVarArg())
    {
        //errs()<<"Can not handle vaarg for now\n";
        return false;
    }
    unsigned int ret_bnd_count = 0;
    //if (ret_type->isAggregateType()){}
    if (ret_type->isStructTy())
    {
        //only care one level down
        for (int i = 0; i<ret_type->getStructNumElements(); i++)
        {
            Type* element_type = ret_type->getStructElementType(i);
            if (element_type->isPointerTy())
            {
                ret_bnd_count++;
            }
        }
    }
    if (ret_type->isPointerTy())
    {
        ret_bnd_count++;
    }
    if (ret_bnd_count>0)
    {
        tr_flist_ret[func_ptr] = ret_bnd_count;
        return true;
    }

    /*
     * examine parameters
     */
    for (Function::arg_iterator i = func_ptr->arg_begin(),
                    e = func_ptr->arg_end();
                    i!=e;++i)
    {
        if (isa<PointerType>(i->getType()))
        {
            tr_flist_ret[func_ptr] = ret_bnd_count;
            return true;
        }
    }

    return false;
}
/*
 * examine phi node, if this phi node is for function pointer
 * and the function pointer type contains any pointer
 * either in return value or argument, we need to transform
 * this phi node
 */
bool llmpx::this_phi_node_need_transform(PHINode* phinode)
{
    Type* phitype = phinode->getType();
    Type* containedType = phitype->getContainedType(0);
    if (!containedType->isFunctionTy())
    {
        return false;
    }
    FunctionType* ftype = dyn_cast<FunctionType>(containedType);
    if (ftype->getReturnType()->isPointerTy())
    {
        return true;
    }
    for (int i=0;i<ftype->getNumParams();i++)
    {
        if (ftype->getParamType(i)->isPointerTy())
        {
            return true;
        }
    }
    return false;
}
/*
 * transform function type into instrumented function type
 * bound information will be added to 
 * return - if pointer type or aggregated type with pointer in 
 *          it is returned
 * arguments - if any of the argument requires bound information
 * ----
 * the transformed type will be returned if the function type is ever
 * modified, otherwise the original type is returned
 * on error NULL is returned
 */
Type* llmpx::transform_function_type(Type* orig_type)
{
    if (!isa<FunctionType>(orig_type))
    {
        return NULL;
    }
    bool need_transform = false;
    FunctionType *orig_ftype = dyn_cast<FunctionType>(orig_type);

    //don't know what to do with vararg
    if (orig_ftype->isVarArg())
    {
        return NULL;
    }

    Type* x86bndty = Type::getX86_BNDTy(*ctx);
    Type* keyty = Type::getInt64Ty(*ctx);

    int ret_bnd_count = 0;
    //for return type
    Type* orig_ret_type = orig_ftype->getReturnType();
    if (orig_ret_type->isPointerTy())
    {
        ret_bnd_count++;
    }else if (orig_ret_type->isStructTy())
    {
        //only care one level down
        for (int i = 0; i<orig_ret_type->getStructNumElements(); i++)
        {
            Type* element_type = orig_ret_type->getStructElementType(i);
            if (element_type->isPointerTy())
            {
                ret_bnd_count++;
            }
        }
    }
    Type* new_ret_type = orig_ret_type;
    //create new aggregated type for return
    if(ret_bnd_count!=0)
    {
        need_transform = true;
        if (has_transformed_type(orig_ret_type))
        {
            new_ret_type = get_transformed_type(orig_ret_type);
        }else
        {
            std::vector<Type*> assemble_new_type;
            assemble_new_type.push_back(orig_ret_type);
            for (int bi = 0; bi < ret_bnd_count; bi++)
            {
                assemble_new_type.push_back(x86bndty);
            }
            /*
             * return key if temporal safety is enabled
             * the number of key == the number of bound
             */
            if (llmpx_enable_temporal_safety)
            {
                for (int bi=0; bi < ret_bnd_count; bi++)
                {
                    assemble_new_type.push_back(keyty);
                }
            }
            new_ret_type = StructType::create(*ctx, assemble_new_type);
            add_transform_type_pair(orig_ret_type, new_ret_type);
        }
    }
    //for parameters
    std::vector<Type*> params;
    int argbnd_cnt = 0;
    for (auto arg_type: orig_ftype->params())
    {
        params.push_back(arg_type);
        if(arg_type->isPointerTy())
        {
            argbnd_cnt++;
            need_transform = true;
        }
    }
    for (int i=0;i<argbnd_cnt;i++)
    {
        params.push_back(x86bndty);
    }
    if (llmpx_enable_temporal_safety)
    {
        for(int i=0; i<argbnd_cnt; i++)
        {
            params.push_back(keyty);
        }
    }
    if (!need_transform)
        return orig_type;
    FunctionType* new_ftype;
    if (params.size()!=0)
    {
        new_ftype = FunctionType::get(new_ret_type, params, orig_ftype->isVarArg());
    }else
    {
        new_ftype = FunctionType::get(new_ret_type, orig_ftype->isVarArg());
    }
    tr_bndinfo_for_rettype[new_ftype] = ret_bnd_count;
    return new_ftype;
}

void llmpx::collect_safe_access(Module& module)
{
    #if DEBUG_MPX_PASS_0_1 
    errs()<<"---Collect Safe Access---\n";
    #endif
    size_t total_dereference = 0;
    for (Module::iterator mi = module.begin(), me = module.end();
            mi != me; ++mi)
    {
        Function *func = dyn_cast<Function>(mi);
        if (func->isDeclaration())
        {
            continue;
        }
        cfunc = func;
        for(Function::iterator fi = func->begin(), fe = func->end();
            fi != fe; ++fi)
        {
            BasicBlock* blk = dyn_cast<BasicBlock>(fi);
            for (BasicBlock::iterator bi = blk->begin(), be = blk->end();
                bi != be; ++bi)
            {
                Value* ptr_operand;
                uint64_t rwsize;
                if(isa<LoadInst>(bi))
                {
                    LoadInst* load = dyn_cast<LoadInst>(bi);
                    ptr_operand = load->getPointerOperand();
                    rwsize = module.getDataLayout()
                             .getTypeStoreSizeInBits(load->getType());
                }else if(isa<StoreInst>(bi))
                {
                    StoreInst* store = dyn_cast<StoreInst>(bi);
                    ptr_operand = store->getPointerOperand();
                    rwsize = module.getDataLayout()
                             .getTypeStoreSizeInBits(store
                                    ->getValueOperand()
                                    ->getType());
                }else
                {
                    continue;
                }
                total_dereference++;
                if (is_safe_access(ptr_operand, rwsize))
                {
                    safe_access_list[dyn_cast<Instruction>(bi)] = ptr_operand;
                }
            }
        }
    }
    errs()<<" "
        <<safe_access_list.size()
        <<"/"
        <<total_dereference
        <<" safe access collected\n";
}

/*
 * Transform functions, add bound/key information
 * ----------------------------------------------
 * If the argument were passed byval, the bound will be constructed
 * at the beginning of the callee instead of being passed as another
 * bnd argument. 
 * FIXME: it is possible that the byval parameter contains pointer,
 * if that pointer was dereferenced, it is possible that there will be a bound
 * mismatch
 */
void llmpx::transform_functions(Module& module)
{
    #if DEBUG_MPX_PASS_1
    errs()<<"---Transform function---\n";
    #endif
    /*
     * transform function to use bound information
     */
    //gather funcs need to be transfomed into orig_flist
    std::list<Function*> orig_flist;
    for (Module::iterator f_begin = module.begin(), f_end = module.end();
            f_begin != f_end; ++f_begin)
    {
        Function *func_ptr = dyn_cast<Function>(f_begin);
        if (func_ptr->isDeclaration())
        {
            continue;
        }
        if (!function_need_to_be_transformed(func_ptr))
        {
            continue;
        }
        orig_flist.push_back(func_ptr);
        #if DEBUG_MPX_PASS_1
        errs()<<"   add "<<func_ptr->getName()<<"\n";
        #endif
    }
    //transform each function in orig_flist
    //scan all parameters for pointer type and add incoming bound
    //scan ret type for pointer type and add return bound
    for (std::list<Function*>::iterator it=orig_flist.begin();
            it != orig_flist.end(); ++it)
    {
        Function* func_ptr = *it;
        #if DEBUG_MPX_PASS_1
        errs()<<"  Transform Function : "
              <<func_ptr->getName()<<"\n";
        #endif

        Type* orig_ret_type = func_ptr->getReturnType();
        Type* ret_type = orig_ret_type;

        if (tr_flist_ret[func_ptr]!=0)
        {
            //means that we need to transform return type
            //to add bound
            if (has_transformed_type(orig_ret_type))
            {
                ret_type = get_transformed_type(orig_ret_type);
            }else
            {
                std::vector<Type*> assemble_new_type;
                assemble_new_type.push_back(orig_ret_type);
                Type* x86bndty = Type::getX86_BNDTy(*ctx);
                for (int bi = 0; bi < tr_flist_ret[func_ptr]; bi++)
                {
                    assemble_new_type.push_back(x86bndty);
                }
                if(llmpx_enable_temporal_safety)
                {
                    for (int bi = 0; bi < tr_flist_ret[func_ptr]; bi++)
                    {
                        assemble_new_type.push_back(Type::getInt64Ty(*ctx));
                    }
                }
                Type* new_ret_type = StructType::create(*ctx, assemble_new_type);
                #if DEBUG_MPX_PASS_1>2
                errs()<<"Assembled new ret type:";
                new_ret_type->dump();
                #endif
                ret_type = new_ret_type;
                add_transform_type_pair(orig_ret_type, new_ret_type);
            }
        }

        const FunctionType* func_fype = func_ptr->getFunctionType();
        std::vector<Type*> params;
        SmallVector<AttributeSet, 8> param_attrs_vec;
        const AttributeSet& orig_attr = func_ptr->getAttributes();
        //attributes of return value
        if (orig_attr.hasAttributes(AttributeSet::ReturnIndex))
        {
            #if DEBUG_MPX_PASS_1>2
            errs()<<" attribute for return value:";
            orig_attr.getRetAttributes().dump();
            #endif
            AttributeSet ret_attr 
                = AttributeSet::get(func_ptr->getContext(),
                                orig_attr.getRetAttributes());
            int j = 0;
            std::list<int> attr_to_remove_idx;
            #if 0
            for (int j=0;j<ret_attr.getNumSlots();j++)
            {
                if (ret_attr.getAttribute(j,Attribute::NoAlias)!=Attribute())
                {
                    attr_to_remove_idx.push_back(j);
                }else if(ret_attr.getAttribute(j, Attribute::NonNull)!=Attribute())
                {
                    attr_to_remove_idx.push_back(j);
                }else if(ret_attr.getAttribute(j, Attribute::Dereferenceable)!=Attribute())
                {
                    attr_to_remove_idx.push_back(j);
                }
            }
            for (auto j: attr_to_remove_idx)
            {
                ret_attr = ret_attr.removeAttribute(*ctx, j, Attribute::NoAlias);
                ret_attr = ret_attr.removeAttribute(*ctx, j, Attribute::NonNull);
                ret_attr = ret_attr.removeAttribute(*ctx, j, Attribute::Dereferenceable);
            }
            #endif
            #if DEBUG_MPX_PASS_1>2
            errs()<<"after strip:";
            ret_attr.dump();
            #endif
            param_attrs_vec.push_back(ret_attr);
        }
        //attributes of the arguments
        int arg_index = 1;
        int ptr_arg_cnt = 0;
        int ptr_arg_with_byval_attr = 0;
        //std::vector<int> ptr_arg_index_list;
        for (Function::arg_iterator i = func_ptr->arg_begin(),
                    e = func_ptr->arg_end();
                    i != e; ++i, ++arg_index)
        {
            params.push_back(i->getType());
            bool arg_is_pointer = false;
            if(isa<PointerType>(i->getType()))
            {
                //ptr_arg_index_list.push_back(arg_index);
                arg_is_pointer = true;
                ptr_arg_cnt++;
            }
            /*
             * FIXME: This remove Returned Attribute,
             * may reduce the chance being optimized
             * need to think it over
             */
            if(orig_attr.hasAttributes(arg_index))
            {
                AttributeSet attrs
                    = orig_attr.getParamAttributes(arg_index);
                AttrBuilder B(orig_attr, arg_index);
                AttributeSet one_arg_attr 
                    = AttributeSet::get(func_ptr->getContext(),
                                    params.size(), B);
                if (arg_is_pointer)
                {
                    if (one_arg_attr.getAttribute(arg_index,
                                            Attribute::Returned)
                                            !=Attribute())
                    {
                        one_arg_attr 
                            = one_arg_attr
                                .removeAttribute(*ctx,
                                                arg_index,
                                                Attribute::Returned);
                    }
                    if (one_arg_attr.getAttribute(arg_index,
                                            Attribute::ByVal)
                                            !=Attribute())
                    {
                        ptr_arg_with_byval_attr++;
                    }
                }
                param_attrs_vec.push_back(one_arg_attr);
            }
        }
        unsigned orig_param_cnt = params.size();
        /*
         * append bound
         */
        //the number of bound/key need to be added
        int bk_cnt = ptr_arg_cnt - ptr_arg_with_byval_attr;
        for (int bndi = 0; bndi<bk_cnt; bndi++)
        {
            params.push_back(Type::getX86_BNDTy(*ctx));
            #if 0
            int corresponding_ptr_arg_idx = ptr_arg_index_list[bndi];
            if (orig_attr.hasAttributes(corresponding_ptr_arg_idx))
            {
                AttributeSet attrs
                    = orig_attr.getParamAttributes(corresponding_ptr_arg_idx);
                attrs.dump();
                AttrBuilder B(orig_attr, corresponding_ptr_arg_idx);
                param_attrs_vec.push_back(
                    AttributeSet::get(func_ptr->getContext(),
                                    params.size(), B));
            }
            #endif
        }
        /*
         * append key
         */
        if (llmpx_enable_temporal_safety)
        {
            for (int keyi = 0; keyi<bk_cnt; ++keyi)
            {
                params.push_back(Type::getInt64Ty(*ctx));
            }
        }

        //create new function
        FunctionType* new_func_type 
            = FunctionType::get(ret_type,
                                params,
                                func_fype->isVarArg());
        Function* new_func = NULL;
        new_func = Function::Create(new_func_type, func_ptr->getLinkage(),
                                    func_ptr->getName()+"_wbnd");
        tr_bnd_list[new_func] = arg_index;
        //set the new function attributes
        new_func->copyAttributesFrom(func_ptr);
        new_func->setAttributes(
                AttributeSet::get(func_ptr->getContext(),
                param_attrs_vec));

        func_ptr->getParent()
                ->getFunctionList()
                    .insert(Module::iterator(func_ptr), new_func);

        // Splice the instructions from the old function into the new
        // function and set the arguments appropriately
        new_func->getBasicBlockList().splice(new_func->begin(), 
                                   func_ptr->getBasicBlockList());
        std::list<std::string> ptr_names;
        Function::arg_iterator arg_i2 = new_func->arg_begin();
        /*
         * return attribute is at index 0
         * argument attribute start from index 1
         */
        int attr_index = 1;
        const AttributeSet& newfunc_attr = new_func->getAttributes();
        /*
         * this stores the index of argument 
         * instead of its position in AttributeSet
         */
        std::list<int>* ptrbyval_list = new std::list<int>;
        tr_ptrbyval_list[new_func] = ptrbyval_list;
        for(Function::arg_iterator arg_i = func_ptr->arg_begin(),
                arg_e = func_ptr->arg_end();
                arg_i != arg_e; ++arg_i, ++attr_index)
        {
            Value* arg_orig = dyn_cast<Value>(arg_i);
            Value* arg_new = dyn_cast<Value>(arg_i2);

            if(isa<PointerType>(arg_orig->getType()))
            {
                bool ptr_is_byval = false;
                if (newfunc_attr.hasAttributes(attr_index))
                {
                    AttributeSet attrs
                        = newfunc_attr.getParamAttributes(attr_index);
                    if(attrs.getAttribute(attr_index,
                                            Attribute::ByVal)
                                            !=Attribute())
                    {
                        ptr_is_byval = true;
                        ptrbyval_list->push_back(attr_index-1);
                    }
                }
                if (!ptr_is_byval)
                {
                    ptr_names.push_back(arg_orig->getName());
                }
            }
            
            arg_orig->replaceAllUsesWith(arg_new);
            arg_new->takeName(arg_orig);
            ++arg_i2;
            arg_index++;
        } 
        for( auto pn: ptr_names )
        {
            arg_i2->setName(pn+".bnd");
            ++arg_i2;
        }

        if (llmpx_enable_temporal_safety)
        {
            for( auto pn: ptr_names )
            {
                arg_i2->setName(pn+".key");
                ++arg_i2;
            }
        }
        //errs()<<"function args replace all uses\n";
        //take care of debug info??
        //ValueToValueMapTy VMap;
        //CloneDebugInfoMetadata(new_func, func_ptr, VMap);
        /*
         * delete old function after all uses has been resolved
         */
        tr_flist[func_ptr] = new_func;
        revtr_flist[new_func] = func_ptr;
        flist_orig.push_back(func_ptr);
        flist_new.push_back(new_func);
    }
    #if DEBUG_MPX_PASS_1
    errs()<<"---------------------------\n";
    #endif
}

/*
 * transform all global variables
 * insert ctor and dtor function,
 * for each global variable, bndmk and bndstx its bound in ctor
 */
void llmpx::transform_global(Module& module)
{
    #if DEBUG_MPX_PASS_1_5
    errs()<<ANSI_COLOR_GREEN
        <<"Transform Global\n"
        <<ANSI_COLOR_RESET;
    #endif
    /*
     * create constructor hold inorder to linkagainst runtime library
     */
    #if 0
    //constructor for MPX
    Function *mpxrt_ctor = Function::Create(
        FunctionType::get(Type::getVoidTy(*ctx),false),
        GlobalValue::ExternalLinkage, "mpxrt_prepare", &module);
    //llmpx runtime library
    Function *llmpx_rt_ctor = Function::Create(
        FunctionType::get(Type::getVoidTy(*ctx),false),
        GlobalValue::ExternalLinkage, "llmpx_rt_init", &module);
    #endif
    //FIXME, need this place holder to bring in ctor and dtor
    Function *llmpx_rt_dummy = Function::Create(
        FunctionType::get(Type::getVoidTy(*ctx), false),
        GlobalValue::ExternalLinkage,
        "llmpx_rt_dummy_for_bring_in_ctor_dependency", &module);

    /*
     * insert ctor
     */
    Function *llmpx_ctor = Function::Create(
        FunctionType::get(Type::getVoidTy(*ctx),false),
        GlobalValue::InternalLinkage, "llmpx_ctor", &module);
    /*
     * append to global ctor does not work?!!
     * insert call at the beginning of main instead.. need fix
     */

#if 0
    appendToGlobalCtors(module, llmpx_ctor, 100);
#else
    Function* mainfunc = module.getFunction("main");
    BasicBlock& mbb = mainfunc->getEntryBlock();
    IRBuilder<> builder0(dyn_cast<Instruction>(mbb.getFirstInsertionPt()));
    builder0.CreateCall(llmpx_ctor);
#endif

    /*
     * test llmpx library
     */
    //builder0.CreateCall(_llmpx_test);
    /*
     * insert dtor??
     */
    /*
     * init global variable bound and store in bound table
     * all instructions are inserted in ctor
     */
    BasicBlock *ctorbb = BasicBlock::Create(*ctx, "llmpx_ctor_bb", llmpx_ctor);
    IRBuilder<> builder(ReturnInst::Create(*ctx, ctorbb));
#if 0
    builder.CreateCall(mpxrt_ctor);
    builder.CreateCall(llmpx_rt_ctor);
#endif
    builder.CreateCall(llmpx_rt_dummy);

#if 1
    Module::GlobalListType &globals = module.getGlobalList();
    for(GlobalVariable &gvi: module.globals())
    {
        GlobalVariable* gi = &gvi;
        if (gi->isDeclaration())
            continue;
        if (!isa<Value>(gi))
            continue;
        Value* gv = dyn_cast<Value>(gi);
        StringRef gvname = gv->getName();
        if (gvname.startswith("llvm.") || 
            gvname.startswith("llmpx_"))
            continue;
        bool gv_use_func = false;
        #if (DEBUG_MPX_PASS_1_5>2)
        errs()<<"  - ";
            gv->print(errs());
        errs()<<":\n";
        #endif
        if (!gi->hasInitializer())
        {
            continue;
        }
        Constant* initializer = gi->getInitializer();
        Type* itype = initializer->getType();
        TotalStaticBNDAdded++;
        /*
         * make bound
         * initialization of constant bound has been changed from using bndmk
         * instruction to series of store instruction in .init_array section,
         * so that it can possibly be further optimized (i.e. make it as 
         * constant bound)
         */
        PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
        unsigned allocated_size = module.getDataLayout()
                        .getTypeAllocSize(itype);

        #if (DEBUG_MPX_PASS_1_5>2)
        errs()<<"bnd parm size:"<<allocated_size<<"\n";
        #endif
        /*
         * create global constant bound for it 
         */
        Type* ArrayTy = ArrayType::get(IntegerType::get(*ctx, 64), 2);
        GlobalVariable* gvbnd
            = new GlobalVariable(module,
                ArrayTy,
                false,
                gi->getLinkage(),
                0, "llmpx_bnd_"+gvname);
        gvbnd->setAlignment(16);
        gvbnd->setInitializer(Constant::getNullValue(ArrayTy));

        Type* int64ty = IntegerType::get(*ctx,64);
        Type* int64ptrty = Type::getInt64PtrTy(*ctx);
        /*
         * FIXME: the following instruction should be inserted into .init_array
         */
        Constant* lb = ConstantExpr::getPtrToInt(gi, int64ty);
        std::vector<Constant*> indic_lb;
        indic_lb.push_back(ConstantInt::get(int64ty,0));
        indic_lb.push_back(ConstantInt::get(int64ty,0));
        Constant* bnd_lb
            = ConstantExpr::getGetElementPtr(NULL, gvbnd, indic_lb);

        Constant* ub
            = ConstantExpr::getNeg(
                ConstantExpr::getAdd(
                    ConstantExpr::getPtrToInt(gi, int64ty),
                    ConstantInt::get(int64ty, (allocated_size))));
        std::vector<Constant*> indic_ub;
        indic_ub.push_back(ConstantInt::get(int64ty,0));
        indic_ub.push_back(ConstantInt::get(int64ty,1));
        Constant* bnd_ub
            = ConstantExpr::getGetElementPtr(
                    NULL,
                    gvbnd,
                    indic_ub);

        Instruction* inslb = builder.CreateStore(lb, bnd_lb);
        Instruction* insub = builder.CreateStore(ub, bnd_ub);
        gv_bound_checklist[gv] = new std::list<Value*>;
        gv_bound_checklist[gv]->push_back(inslb);
        gv_bound_checklist[gv]->push_back(insub);
        gv_bound_checklist[gv]->push_back(gvbnd);

        /*
         * create global key and lock
         */
        if (llmpx_enable_temporal_safety)
        {
            std::vector<Value*> args;
            args.push_back(ConstantExpr::getBitCast(gi,Int8PtrTy));
            args.push_back(ConstantInt::get(Type::getInt64Ty(*ctx), (allocated_size)));
            Value* key
                = builder.CreateCall(_llmpx_temporal_lock_alloca,
                            args, "llmpx_key."+gvname);

            auto* Int64Ty = Type::getInt64Ty(*ctx);

            GlobalVariable* gvkey
                 = new GlobalVariable(module, Int64Ty,
                        false, gi->getLinkage(),
                        0, "llmpx_key_"+gvname);
            gvkey->setInitializer(ConstantInt::get(Int64Ty, 0));

            Instruction* keystore = builder.CreateStore(key,
                ConstantExpr::getPointerCast(gvkey, Type::getInt64PtrTy(*ctx)));

            gv_key_checklist[gv] = new std::list<Value*>;
            gv_key_checklist[gv]->push_back(keystore);
            gv_key_checklist[gv]->push_back(gvkey);
        }
#if 0
        /*
         * do we really need this?
         * global struct member will be stored later,
         * this means new bound will be generated and stored.
         */
        //handle global struct type
        if (itype->isStructTy())
        {
            #if (DEBUG_MPX_PASS_1_5<2)
            errs()<<"Found StructType\n";
            #endif
            bool changed = false;
            std::vector<Constant*> new_data;
            StructType* istype = dyn_cast<StructType>(itype);
            for( int j=0;j<istype->getNumElements();j++)
            {
                Constant* element
                    = initializer->getAggregateElement(j);
                Type* etype = istype->getElementType(j);
                if(!etype->isPointerTy())
                {
                    new_data.push_back(element);
                    continue;
                }
                Type* cetype 
                    = dyn_cast<PointerType>(etype)->getContainedType(0);
                if(isa<FunctionType>(cetype))
                {
                    #if (DEBUG_MPX_PASS_1_5<2)
                    errs()<<"  found function_type : ";
                    etype->dump();
                    #endif
                    Constant* func = element->stripPointerCasts();
                    if (!is_function_orig(func))
                    {
                        new_data.push_back(element);
                        continue;
                    }
                    Function* newfunc = find_transformed_function(func);
                    element 
                        = ConstantExpr::getBitCast(newfunc, etype);
                    new_data.push_back(element);
                    changed = true;
                }else
                {
                    new_data.push_back(element);
                }
            }
            if (changed)
            {
                Constant* new_initializer
                    = ConstantStruct::get(istype, new_data);
                gi->setInitializer(new_initializer);
                #if (DEBUG_MPX_PASS_1_5<3)
                errs()<<"Assembled new initializer:";
                new_initializer->dump();
                #endif
            }
        }
#endif
    }
#endif
}

/*
 * prcess all functions in module
 */
void llmpx::process_each_function(Module& module)
{
    std::list<Function*> processed_flist;
    //for each function
    for (Module::iterator f_begin = module.begin(), f_end = module.end();
            f_begin != f_end; ++f_begin)
    {
        bound_checklist.clear();
        key_checklist.clear();
        gv_bound_checklist_cache.clear();
        gv_key_checklist_cache.clear();

        Function *func_ptr = dyn_cast<Function>(f_begin);

        bool found = (std::find(std::begin(processed_flist),
                                 std::end(processed_flist), func_ptr) 
                            != std::end(processed_flist));
        if (found)
        {
            continue;
        }
        if (func_ptr->isDeclaration())
        {
            ExternalFuncCounter++;
            continue;
        }
        if (is_in_do_not_instrument_list(func_ptr->getName()))
        {
            continue;
        }
        FuncCounter++;
        #if DEBUG_MPX_PASS_2
        errs()<<ANSI_COLOR_MAGENTA
            <<"Process Function : "
            <<func_ptr->getName()
            <<ANSI_COLOR_RESET
            <<"\n";
        #endif
        /*
         * take care of incoming bound and key
         */
        //iterate through parameters
        //and associate bound/key infomation to parameters
#if 1
        /*
         * findout real function to instrument
         */
        processed_flist.push_back(func_ptr);
        Function* new_func_ptr = find_transformed_function(func_ptr);
        func_ptr = find_nontransformed_function(func_ptr);
        assert((new_func_ptr!=NULL) && "the new function is null?");
        assert((func_ptr!=NULL) && "the orig function is null?");

        /*
         * NOTE: Associate bound/key in parameters with its arguments
         * For argument passed byval, we construct its bound/key
         */
        if (new_func_ptr!=func_ptr)
        {
            processed_flist.push_back(new_func_ptr);
            int bnd_idx = tr_bnd_list[new_func_ptr];
            std::list<int>* ptrbyval_list = tr_ptrbyval_list[new_func_ptr];
            func_ptr = new_func_ptr;
            Function::arg_iterator bndi = func_ptr->arg_begin();
            for (int bi=1;bi<bnd_idx;bi++)
            {
                ++bndi;
            }
            int idx = 0;
            /*
             * Create builder to insert instruction at the beginning
             * of the function
             */
            IRBuilder<> init_builder(
                    dyn_cast<Instruction>(func_ptr
                                        ->getEntryBlock()
                                        .getFirstInsertionPt()));
            for (Function::arg_iterator i = func_ptr->arg_begin(),
                            e = func_ptr->arg_end();
                            (i!=e)&&(idx<bnd_idx); ++i, ++idx)
            {
                if (isa<PointerType>(i->getType()))
                {
                    Value* bound;
                    Value* ptr_arg = dyn_cast<Value>(i);
                    if (!ptr_arg)
                        llvm_unreachable("argument is bad!\n");

                    std::list<Value*>* blist = new std::list<Value*>();

                    auto ptrbyval_it
                        = std::find(ptrbyval_list->begin(),
                                    ptrbyval_list->end(),
                                    idx);
                    if (ptrbyval_it != ptrbyval_list->end())
                    {
                        /*
                         * ptr has pass byval attribute, we need to construct
                         * its bound here
                         */

                        std::vector<Value*> args_for_bndmk;

                        PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
                        Value* ptr8 = init_builder.CreateBitCast(ptr_arg, Int8PtrTy, "");
                        args_for_bndmk.push_back(ptr8);

                        /*
                         * get the actual type of this pointer
                         */
                        Type *itype = ptr_arg->getType()->getPointerElementType();

                        unsigned allocated_size = 0;
                        allocated_size = module.getDataLayout()
                                        .getTypeAllocSize(itype);

                        Constant* dist_arg_for_bndmk
                            = ConstantInt::get(Type::getInt64Ty(*ctx), (allocated_size-1));
                        args_for_bndmk.push_back(dist_arg_for_bndmk);

                        Instruction* bndmkcall
                            = init_builder.CreateCall(mpx_bndmk, args_for_bndmk,
                                            ptr_arg->getName()+".bv.bnd");
                        bound = bndmkcall;
                        TotalBNDMKAdded++;
                    }else
                    {
                        /*
                         * bound is passed as argument, associate with 
                         * corresponding ptr argument
                         */
                        bound = dyn_cast<Value>(bndi);
                        ++bndi;
                    }
                    blist->push_back(bound);
                    bound_checklist[ptr_arg] = blist;
                    #if (DEBUG_MPX_PASS_2>2)
                    errs()<<ANSI_COLOR_YELLOW;
                    errs()<<" - associate ";
                    ptr_arg->print(errs());
                    errs()<<" with bnd ";
                    bound->print(errs());
                    errs()<<ANSI_COLOR_RESET;
                    errs()<<"\n";
                    #endif
                }
            }
            /*
             * for keys, do the same as bounds
             * deal with ptr with byval attribute
             */
            if (llmpx_enable_temporal_safety)
            {
                Function::arg_iterator keyi = bndi;
                int idx = 0;
                for (Function::arg_iterator i = func_ptr->arg_begin(),
                                e = func_ptr->arg_end();
                                (i!=e)&&(idx<bnd_idx); ++i, ++idx)
                {
                    if (isa<PointerType>(i->getType()))
                    {
                        Value* key;
                        Value* ptr_arg = dyn_cast<Value>(i);
                        if (!ptr_arg)
                            llvm_unreachable("argument is bad!\n");
                        std::list<Value*>* klist = new std::list<Value*>();

                        //associate key of i to 
                        auto ptrbyval_it
                            = std::find(ptrbyval_list->begin(),
                                        ptrbyval_list->end(),
                                        idx);
                        if (ptrbyval_it != ptrbyval_list->end())
                        {
                            /*
                             * ptr has pass byval attribute, we need to construct
                             * its key here
                             */
                            std::vector<Value*> args_for_kl_alloca;

                            PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
                            Value* ptr8 = init_builder.CreateBitCast(ptr_arg, Int8PtrTy, "");
                            args_for_kl_alloca.push_back(ptr8);

                            /*
                             * get the actual type of this pointer
                             */
                            Type *itype = ptr_arg->getType()->getPointerElementType();

                            unsigned allocated_size = 0;
                            allocated_size = module.getDataLayout()
                                            .getTypeAllocSize(itype);

                            Constant* dist_arg_for_kl_alloca
                                = ConstantInt::get(Type::getInt64Ty(*ctx), (allocated_size-1));
                            args_for_kl_alloca.push_back(dist_arg_for_kl_alloca);

                            Instruction* lock_alloca_call
                                = init_builder.CreateCall(_llmpx_temporal_lock_alloca,
                                                args_for_kl_alloca,
                                                ptr_arg->getName()+".bv.key");

                            key = lock_alloca_call;
                        }else
                        {
                            /*
                             * key is passed as argument, associate with 
                             * corresponding ptr argument
                             */
                            key = dyn_cast<Value>(keyi);
                            ++keyi;
                        }
                        klist->push_back(key);
                        key_checklist[ptr_arg] = klist;
                        #if (DEBUG_MPX_PASS_2>2)
                        errs()<<ANSI_COLOR_YELLOW;
                        errs()<<" - associate ";
                        ptr_arg->print(errs());
                        errs()<<" with key ";
                        key->print(errs());
                        errs()<<ANSI_COLOR_RESET;
                        errs()<<"\n";
                        #endif
                    }
                }
            }
        }
#endif
        /*
         * this is worklist algorithm
         */
        std::set<BasicBlock*> bb_visited;
        std::queue<BasicBlock*> bb_work_list;
        bb_work_list.push(&func_ptr->getEntryBlock());
        while(bb_work_list.size())
        {
            /*
             * pick the first item in the worklist
             */
            BasicBlock* bb = bb_work_list.front();
            bb_work_list.pop();
            if(bb_visited.count(bb))
            {
                continue;
            }
            bb_visited.insert(bb);
            /*
             * for each basic block, we scan through instructions
             *  - gather bound information when new pointer is allocated
             *  - insert bound check when pointer is dereferenced
             */
            
            for (BasicBlock::iterator ii = bb->begin(),
                    ie = bb->end();
                    ii!=ie; ++ii)
            {
                Instruction *I = dyn_cast<Instruction>(ii);
                gen_bound_checklist(I);
            }
            /*
             * insert all successor of current basic block to work list
             */
            for (succ_iterator si = succ_begin(bb),
                    se = succ_end(bb);
                    si!=se; ++si)
            {
                BasicBlock* succ_bb = cast<BasicBlock>(*si);
                bb_work_list.push(succ_bb);
            }
        }
        cfunc = func_ptr;

        /*
         * there may be cases that block has no predecessor
         * we need to handle this specially
         */
        Function* func = func_ptr;
        for(Function::iterator i = func->begin(), e = func->end(); i != e; ++i)
        {
            BasicBlock* blk = dyn_cast<BasicBlock>(i);
            if (bb_visited.count(blk)==0)
            {
                #if (DEBUG_MPX_PASS_2>1)
                errs()<<" return bb has no predecessor, scan it anyway\n";
                #endif
                bb_visited.insert(blk);
                for (BasicBlock::iterator ii = blk->begin(),
                        ie = blk->end();
                        ii!=ie; ++ii)
                {
                    Instruction *I = dyn_cast<Instruction>(ii);
                    gen_bound_checklist(I);
                }
            }
        }
        process_bound_checklist();
    }
}

Value* llmpx::find_true_val_has_aa_id(Value* the_ptr)
{
    Value* ret = the_ptr;
    while (get_aa_set_id(ret)==-1)
    {
        if (isa<BitCastInst>(ret))
        {
            BitCastInst* bi = dyn_cast<BitCastInst>(ret);
            ret = bi->getOperand(0);
        }else if (isa<GetElementPtrInst>(ret))
        {
            GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(ret);
            ret = gep->getOperand(0);
        }else
        {
            //errs()<<"AA: next is not bitcast:";
            //ret->dump();
            break;
        }
    }
    return ret;
}

/*
 * if the stored bound is never used/loaded by bndldx, remove it
 */
int llmpx::dead_bndstx_elimination(Module& module)
{
#if DEBUG_DEAD_BNDSTX_ELIM
    errs()<<ANSI_COLOR_YELLOW
        <<"Dead BNDSTX Elimination"
        <<ANSI_COLOR_RESET<<"\n";
#endif
    int cnt = 0;
    std::list<Value*> bndldx_ptr_list;
    for (auto &i: bndldxlist)
    {
        CallInst* ci = dyn_cast<CallInst>(i);
        Value* the_ptr = find_true_val_has_aa_id(ci->getArgOperand(0));
        bndldx_ptr_list.push_back(the_ptr);
    }
    std::list<Value*> delete_list;
    for (auto &i: bndstxlist)
    {
        CallInst* ci = dyn_cast<CallInst>(i);
        Value* the_ptr = find_true_val_has_aa_id(ci->getArgOperand(0));
        int set_no = get_aa_set_id(the_ptr);
        bool found_alias = false;
        if (set_no==-1)
        {
            errs()<<ANSI_COLOR_RED
                <<"aa set number not found:"
                <<ANSI_COLOR_RESET<<"\n";
            ci->dump();
            the_ptr->dump();
            found_alias = true;
            //goto remove_bndstx;
            //exit(-1);
        }
        for (auto &j: bndldx_ptr_list)
        {
            if (get_aa_set_id(j)==set_no)
            {
                found_alias = true;
                goto remove_bndstx;
            }
        }
remove_bndstx:
        if (!found_alias)
        {
            //dead, remove it
            delete_list.push_back(ci);
        }
    }
    cnt = delete_list.size();
    DeadBNDSTXEliminated += cnt;
    for (auto&i: delete_list)
    {
        CallInst* ci = dyn_cast<CallInst>(i);
        #if (DEBUG_DEAD_BNDSTX_ELIM>2)
        ci->dump();
        #endif
        ci->eraseFromParent();
        bndstxlist.remove(ci);
    }
#if DEBUG_DEAD_BNDSTX_ELIM
    errs()<<cnt<<" bndstx removed\n";
#endif
    TotalBNDSTXAdded -= cnt;
    return cnt;
}

/*
 * remove bndldx if the address is not aliased with address in bndstx,
 * means that it try to load null bound
 */
int llmpx::dead_bndldx_elimination(Module &module)
{
#if DEBUG_DEAD_BNDLDX_ELIM
    errs()<<ANSI_COLOR_YELLOW
        <<"Dead BNDLDX Elimination"
        <<ANSI_COLOR_RESET<<"\n";
#endif
    int cnt = 0;
    std::list<Value*> bndstx_ptr_list;
    for (auto &i: bndstxlist)
    {
        CallInst* ci = dyn_cast<CallInst>(i);
        Value* the_ptr = find_true_val_has_aa_id(ci->getArgOperand(0));
        bndstx_ptr_list.push_back(the_ptr);
    }
    std::list<Value*> delete_list;
    for (auto &i: bndldxlist)
    {
        CallInst* ci = dyn_cast<CallInst>(i);
        Value* the_ptr = find_true_val_has_aa_id(ci->getArgOperand(0));
        int set_no = get_aa_set_id(the_ptr);
        bool found_alias = false;
        if (set_no==-1)
        {
            errs()<<ANSI_COLOR_RED
                <<"aa set number not found:"
                <<ANSI_COLOR_RESET<<"\n";
            ci->dump();
            the_ptr->dump();
            found_alias = true;
            //goto remove_bndldx;
            //exit(-1);
        }
        for (auto &j: bndstx_ptr_list)
        {
            if (get_aa_set_id(j)==set_no)
            {
                found_alias = true;
                goto remove_bndldx;
            }
        }
remove_bndldx:
        if (!found_alias)
        {
            //dead, remove it
            delete_list.push_back(ci);
        }
    }
    cnt = delete_list.size();
    DeadBNDLDXEliminated += cnt;
    for (auto&i: delete_list)
    {
        CallInst* ci = dyn_cast<CallInst>(i);
        //remove all use of this bound
        //only targeting simple case for now
        bool simple = true;
        std::list<User*> user_list;
        for (auto use = ci->use_begin(), use_end = ci->use_end();
                use!=use_end; ++use)
        {
            User* user = use->getUser();
            if (!isa<CallInst>(user))
            {
                simple = false;
                break;
            }
            CallInst* ci = dyn_cast<CallInst>(user);
            Function* func = ci->getCalledFunction();
            if (!func)
            {
                simple = false;
                break;
            }
            StringRef func_name = func->getName();
            if (!func_name.startswith("llvm.x86.bndc"))
            {
                simple = false;
                break;
            }
            user_list.push_back(user);
        }
        if (!simple)
        {
            cnt--;
            DeadBNDLDXEliminated--;
            continue;
        }
        #if (DEBUG_DEAD_BNDLDX_ELIM>2)
        ci->dump();
        #endif
        int chks_removed = user_list.size();
        TotalChecksAdded -= chks_removed/2;
        for (auto* user: user_list)
        {
        #if (DEBUG_DEAD_BNDLDX_ELIM>2)
            errs()<<" - ";
            user->dump();
        #endif
            Instruction* i = dyn_cast<Instruction>(user);
            i->eraseFromParent();
        }
        ci->eraseFromParent();
        bndldxlist.remove(ci);
    }
    TotalBNDLDXAdded -= cnt;
    #if DEBUG_DEAD_BNDLDX_ELIM
    errs()<<cnt<<" bndldx removed\n";
    #endif
    return cnt;
}

#if BND_CHK_CONSOLIDATION
static const char* mpx_chk_intrinsics[] =
{
    "llvm.x86.bndclrr",
    "llvm.x86.bndclrm",
    "llvm.x86.bndcurr",
    "llvm.x86.bndcurm"
};
#endif

/*
 * cleanup
 * remove all function that has been transformed
 * rename function back to its original name
 * remove all unused bound
 */
void llmpx::cleanup(Module& module)
{
    //inspect all functions
    for (auto I: flist_orig)
    {
        Function* old_func_ptr = I;
        Function* new_func_ptr = find_transformed_function(old_func_ptr);

        //Don't have to replace, because the signature is already different
        //old_func_ptr->replaceAllUsesWith(new_func_ptr);
        // Remove the old function from the module

        //preserve old function name 
        std::string old_func_name
            = std::string(old_func_ptr->getName().data());

        if (old_func_ptr->getNumUses()==0)
        {
            old_func_ptr->eraseFromParent();
        }else
        {
            #if DEBUG_MPX_PASS_3
            errs()<<ANSI_COLOR_RED
                <<"WARNNING: Old function still have uses!\n"
                <<ANSI_COLOR_RESET;
            errs()<<ANSI_COLOR_YELLOW
                <<"=> "
                <<old_func_ptr->getName()<<"(";
            old_func_ptr->getType()->print(errs());
            errs()<<" "<<old_func_ptr->getNumUses()<<" uses ";
            errs()<<") \n=> "<<new_func_ptr->getName()<<"(";
            new_func_ptr->getType()->print(errs());
            errs()<<" "<<new_func_ptr->getNumUses()<<" uses ";
            errs()<<")\n"<<ANSI_COLOR_RESET;
            #endif
            #if DEBUG_MPX_PASS_3
            for (auto *U: old_func_ptr->users())
            {
                errs()<<"    - ";
                U->dump();
            }
            #endif
            /*
             * force replace all uses?
             */
            #if DEBUG_MPX_PASS_3
            errs()<<"Force replace all uses ... ";
            #endif
            old_func_ptr->replaceAllUsesWith(
                ConstantExpr::getBitCast(new_func_ptr,
                    old_func_ptr->getFunctionType()->getPointerTo()));
            #if DEBUG_MPX_PASS_3
            errs()<<"Erase anyway.\n";
            #endif
            old_func_ptr->eraseFromParent();
        }
        //use its original name
        new_func_ptr->setName(old_func_name);
    }
#if DEBUG_MPX_PASS_3
    errs()<<" - optimization...\n";
#endif

opt_again:
    int again;
#if DEAD_BOUND_ELIMINATION
    while(remove_dead_bound(module)!=0){};
#endif
#if AA_OPTIMIZATION
    again = dead_bndstx_elimination(module);
    again = dead_bndldx_elimination(module);
    if (again!=0)
    {
        goto opt_again;
    }
#endif
//for basic block
#if BND_CHK_CONSOLIDATION
    int bb_cnt = 0;
    errs()<<"BB level bound check consolidation\n";
    for (Module::iterator mi = module.begin(), me = module.end();
        mi != me; ++mi)
    {
        Function* func = dyn_cast<Function>(mi);
        if (func->isDeclaration())
            continue;
        cfunc = func;
        for (Function::iterator fi = func->begin(), fe = func->end();
            fi != fe; ++fi)
        {
            BasicBlock* blk = dyn_cast<BasicBlock>(fi);
            if (!blk->hasName())
                continue;
            StringRef blkname = blk->getName();
            //if (!blkname.startswith("vector.body"))
            //    continue;
            //errs()<<"Process bb\n";
            bb_cnt++;
            //collect things need to be reordered
            std::list<Instruction*> reorderlist;
            for(BasicBlock::iterator bi = blk->begin(), be = blk->end();
                bi != be; ++bi)
            {
                if(!isa<CallInst>(bi))
                    continue;
                CallInst* ci = dyn_cast<CallInst>(bi);
                Function* f = ci->getCalledFunction();
                if (!f)
                    continue;
                StringRef fname = f->getName();
                if (std::find(std::begin(mpx_chk_intrinsics),
                            std::end(mpx_chk_intrinsics), fname)
                            == std::end(mpx_chk_intrinsics))
                {
                    continue;
                }
                reorderlist.push_back(ci);
            }
            //test scev
            auto *se = &getAnalysis<ScalarEvolutionWrapperPass>(*func).getSE();
            struct _brange {
                Value* lobase;
                Value* hibase;
                Value* hioffset;
                Instruction* lbchk;
                Instruction* ubchk;
            };
            std::map<Value*, struct _brange> bnd_to_range;
            for (auto it: reorderlist)
            {
                CallInst* ci = dyn_cast<CallInst>(it);
                Function* xfunc = ci->getCalledFunction();

                Value* bnd = ci->getArgOperand(0);
                struct _brange& bndrange = bnd_to_range[bnd];
                Value* ptr = ci->getArgOperand(1)->stripPointerCasts();
                Value* offset = NULL;
                StringRef fname = xfunc->getName();
                if(fname.endswith("rm"))
                {
                    offset = ci->getArgOperand(4);
                }
                if (bndrange.lobase==NULL)
                {
                    bndrange.lobase = ptr;
                    bndrange.hibase = ptr;
                    bndrange.hioffset = offset;
                    bndrange.lbchk = ci;
                    bndrange.ubchk = ci;
                }else
                {
                    const SCEV* s = NULL;
                    s = se->getSCEV(ptr);
                    const SCEV* ls = se->getSCEV(bndrange.lobase);
                    const SCEV* hs = NULL;
                    hs = se->getSCEV(bndrange.hibase);
                    const SCEV* dls = se->getMinusSCEV(ls, s);
                    const SCEV* dhs = se->getMinusSCEV(s, hs);
                    if (se->isKnownNonNegative(dls))
                    {
                        bndrange.lobase = ptr;
                        bndrange.lbchk = ci;
                    }
                    if (se->isKnownNonNegative(dhs))
                    {
                        bndrange.hibase = ptr;
                        bndrange.hioffset = offset;
                        bndrange.ubchk = ci;
                    }
                }
            }
            //worth optimize?
            if ((bnd_to_range.size()*2)==reorderlist.size())
            {
                continue;
            }
#if 0
            size_t optimize_effect = reorderlist.size() - bnd_to_range.size()*2;
            errs()<<ANSI_COLOR_CYAN
                <<"reduce bndchk by "
                <<optimize_effect
                <<ANSI_COLOR_RESET
                <<"\n";
            errs()<<ANSI_COLOR_MAGENTA<<"Result:\n"<<ANSI_COLOR_RESET;
#endif
            std::set<Instruction*> reserve_chk;
            for (auto it: bnd_to_range)
            {
#if 0
                errs()<<">";
                (it.first)->dump();
                (it.second).lobase->dump();
                (it.second).hibase->dump();
#endif
                reserve_chk.insert(it.second.lbchk);
                reserve_chk.insert(it.second.ubchk);
#if 0
                if(it.second.hioffset)
                {
                    errs()<<"offset:";
                    it.second.hioffset->dump();
                }
#endif
            }
            //add all uses to the list
            std::list<Instruction*> fix_reorderlist;
collect_all_def_use:
            for(BasicBlock::iterator bi = blk->begin(), be = blk->end();
                bi != be; ++bi)
            {
                Instruction* cins = dyn_cast<Instruction>(bi);
                if (std::find(reorderlist.begin(), reorderlist.end(),cins)
                    !=reorderlist.end())
                {
                    fix_reorderlist.push_back(cins);
                    continue;
                }
                bool found_use = false;
                for (auto use = cins->use_begin(), use_end = cins->use_end();
                    use!=use_end; ++use)
                {
                    User* user = use->getUser();
                    Instruction*uins = dyn_cast<Instruction>(user);
                    if (uins->getParent()!=blk)
                        continue;
                    if (std::find(reorderlist.begin(),reorderlist.end(),uins)==reorderlist.end())
                        continue;
                    found_use = true;
                    break;
                }
                if (found_use)
                    fix_reorderlist.push_back(cins);
            }
            if (fix_reorderlist.size()!=reorderlist.size())
            {
                reorderlist.clear();
                reorderlist.splice(reorderlist.begin(),fix_reorderlist);
                goto collect_all_def_use;
            }
            //re-order them
            if (fix_reorderlist.size()<2)
            {
                continue;
            }
            Instruction* inspt;
            //skip phinode
            while (isa<PHINode>(fix_reorderlist.front()))
            {
                fix_reorderlist.pop_front();
            }
            inspt = fix_reorderlist.front();
            fix_reorderlist.pop_front();
            
            for (auto it: fix_reorderlist)
            {
                if (isa<CallInst>(it))
                {
                    Function* f = dyn_cast<CallInst>(it)->getCalledFunction();
                    if (!f)
                    {
                        goto moveinst;
                    }
                    StringRef fname = f->getName();
                    if (std::find(std::begin(mpx_chk_intrinsics),
                                std::end(mpx_chk_intrinsics), fname)
                                == std::end(mpx_chk_intrinsics))
                    {
                        goto moveinst;
                    }
                    if (reserve_chk.find(it)!=reserve_chk.end())
                    {
                        it->eraseFromParent();
                        ConsolidatedBNDCHK++;
                        continue;
                    }
                }
moveinst:
                //it->removeFromParent();
                //it->insertAfter(inspt);
                inspt = it;
            }
            //Instruction* inspt = GetNextInstruction(inspt);
        }
    }
    errs()<<" number of bb visited : "<<bb_cnt<<"\n";
#endif
}

int llmpx::remove_dead_bound(Module& module)
{
    int bnd_cnt = 0;
    /*
     * remove unused bound as well as all its use if possible
     */
    std::list<Instruction*> dead_bound;
    for (Module::iterator fi = module.begin(), fe = module.end();
            fi != fe; ++fi)
    {
        Function* func = dyn_cast<Function>(fi);
        for(Function::iterator i = func->begin(), e = func->end(); i != e; ++i)
        {
            BasicBlock* blk = dyn_cast<BasicBlock>(i);
            for (BasicBlock::iterator ins = blk->begin(), inse = blk->end(); ins != inse; ++ins)
            {
                Instruction* iii = dyn_cast<Instruction>(ins);
                if (iii && iii->getType()->isX86_BNDTy())
                {
                    if (iii->hasNUsesOrMore(1))
                    {
                        continue;
                    }
                    dead_bound.push_back(iii);
                    ElimBound++;
                    bnd_cnt++;
                }
            }
        }
    }
    for (auto di: dead_bound)
    {
        std::list<Instruction*> alluse;
        for(auto u = di->use_begin(), ue =di->use_end();
            u!=ue; ++u)
        {
            Use &use = *u;
            if (!isa<Instruction>(use.getUser()))
                continue;
            Instruction* ins = dyn_cast<Instruction>(use.getUser());
            if (ins->hasOneUse())
            {
                alluse.push_back(ins);
            }
        }
        //statistics
        switch (di->getOpcode())
        {
            case (Instruction::Call):
            {
                CallInst* call_inst = dyn_cast<CallInst>(di);
                Function* f = call_inst->getCalledFunction();
                if(f==mpx_bndmk)
                {
                    TotalBNDMKAdded--;
                }else if(f==mpx_bndldx)
                {
                    bndldxlist.remove(call_inst);
                    TotalBNDLDXAdded--;
                }
                break;
            }
            default:
                break;
        }
        /////////////////////////
        di->eraseFromParent();
        for(auto use: alluse)
        {
            use->eraseFromParent();
        }
    }
    return bnd_cnt;
}

void llmpx::verify(Module& module)
{
    errs()<<"  check bogus instruction parent ";
    for (Module::iterator fi = module.begin(), fe = module.end();
            fi != fe; ++fi)
    {
        Function* func = dyn_cast<Function>(fi);
        for(Function::iterator i = func->begin(), e = func->end(); i != e; ++i)
        {
            BasicBlock* blk = dyn_cast<BasicBlock>(i);
            for (BasicBlock::iterator ins = blk->begin(), inse = blk->end(); ins != inse; ++ins)
            {
                Instruction* iii = dyn_cast<Instruction>(ins);
                if (iii->getParent()!=blk)
                {
                    errs()<<"["<<ANSI_COLOR_RED<<"BAD"<<ANSI_COLOR_RESET<<"]\n";
                    iii->print(errs());
                    errs()<<"\n";
                    llvm_unreachable("Instruction has bogus parent pointer!");
                }
            }
        }
    }
    errs()<<"["<<ANSI_COLOR_GREEN<<"OK"<<ANSI_COLOR_RESET<<"]\n";
    //dominance relation
    #if 1
    errs()<<"  check dominance relation ";
    DominatorTree DT;
    for (Module::iterator fi = module.begin(), fe = module.end();
            fi != fe; ++fi)
    {
        Function* func = dyn_cast<Function>(fi);
        DT.recalculate(*func);
        for(Function::iterator i = func->begin(), e = func->end(); i != e; ++i)
        {
            BasicBlock* blk = dyn_cast<BasicBlock>(i);
            for (BasicBlock::iterator ins = blk->begin(), inse = blk->end(); ins != inse; ++ins)
            {
                Instruction* iii = dyn_cast<Instruction>(ins);
                for (unsigned i =0, e = iii->getNumOperands(); i!=e; ++i)
                {
                    if (!isa<Instruction>(iii->getOperand(i)))
                    {
                        continue;
                    }
                    Instruction *Op = cast<Instruction>(iii->getOperand(i));
                    if (InvokeInst *II = dyn_cast<InvokeInst>(Op))
                    {
                        if (II->getNormalDest() == II->getUnwindDest())
                        {
                            continue;
                        }
                    }
                    const Use &U = iii->getOperandUse(i);
                    if (!DT.dominates(Op, U))
                    {
                        errs()<<"["<<ANSI_COLOR_RED<<"BAD"<<ANSI_COLOR_RESET<<"]\n";
                        errs()<<ANSI_COLOR_RED<<"Use:"<<ANSI_COLOR_RESET;
                        iii->dump();
                        BasicBlock* usebb = iii->getParent();
                        errs()<<"   in BB:"<<usebb->getName()<<"\n";
                        errs()<<ANSI_COLOR_GREEN<<"Def:"<<ANSI_COLOR_RESET;
                        Op->dump();
                        BasicBlock* defbb = Op->getParent();
                        errs()<<"   in BB:"<<defbb->getName()<<"\n";
                        for (succ_iterator si = succ_begin(defbb),
                            se = succ_end(defbb);
                            si!=se; ++si)
                        {
                            BasicBlock* succ_bb = cast<BasicBlock>(*si);
                            errs()<<"     |->"<<succ_bb->getName()<<"\n";
                        }
                        errs()<<"--------------------------------------------\n";
                        defbb->dump();
                        usebb->dump();
                        llvm_unreachable("Instruction does not dominate all uses!");
                    }
                }
            }
        }
    }
    errs()<<"["<<ANSI_COLOR_GREEN<<"OK"<<ANSI_COLOR_RESET<<"]\n";
    #endif
    #if 1
    errs()<<"  check return type ";
    for (Module::iterator fi = module.begin(), fe = module.end();
            fi != fe; ++fi)
    {
        Function* func = dyn_cast<Function>(fi);
        for(Function::iterator i = func->begin(), e = func->end(); i != e; ++i)
        {
            BasicBlock* blk = dyn_cast<BasicBlock>(i);
            for (BasicBlock::iterator ins = blk->begin(), inse = blk->end(); ins != inse; ++ins)
            {
                Instruction* iii = dyn_cast<Instruction>(ins);
                ReturnInst* ret_inst = dyn_cast<ReturnInst>(iii);
                if (!ret_inst)
                {
                    continue;
                }
                StringRef errtext="";
                if(func->getReturnType()->isVoidTy())
                {
                    if (ret_inst->getReturnValue()==NULL)
                    {
                        continue;
                    }else
                    {
                        errtext="Return non-void for void function!";
                        goto examine_ret_type_fail;
                    }
                }
                if (ret_inst->getReturnValue()==NULL)
                {
                    errtext="Return void for non-void function!";
                    goto examine_ret_type_fail;
                }
                if (ret_inst->getReturnValue()->getType()!=
                        func->getReturnType())
                {
                    errtext = "Return value does not match function return type!";
                    goto examine_ret_type_fail;
                }
                continue;

examine_ret_type_fail:
                errs()<<"["<<ANSI_COLOR_RED<<"BAD"<<ANSI_COLOR_RESET<<"]\n";
                iii->print(errs());
                errs()<<"\n";
                errs()<<"required return type: ";
                func->getReturnType()->dump();
                errs()<<"\n";
                DEBUG(
                    errs()<<"Full function dump:";
                    func->dump();
                );
                llvm_unreachable(errtext.data());
            }
        }
    }
    errs()<<"["<<ANSI_COLOR_GREEN<<"OK"<<ANSI_COLOR_RESET<<"]\n";
    #endif

    //collect final statistics
    for (auto it=stat_set_used.begin(); it!=stat_set_used.end(); ++it)
    {
        if (it->second!=0)
            TotalAliasSetsUsedForMPX++;
    }
}

/*
 * Process each function
 * ---------------------
 *  Need to implement bound propogation and 
 *  insert bound check for each dereference
 *  using the specification in the SysV ABI
 *
 *  according to SDM
 *    1. need to process each call/ret in internal function
 *       and call to external function
 *    2. need to process jmp and jcc in each internal function
 *
 *  might need more work in the backend, so that we can add prefix to 
 *  the function call and return
 * ---------------------
 *  The whole algorithm is organized as follow:
 *    1. gather bound information for each instruction
 *       including bound make, bound check and bound propogation
 *       into a tree? or something else
 *       so that we can know where to insert bound create/check etc.
 *       we call the result of this phase as bound_checklist
 *    2. optimize bound_checklist to elimite redundant checks
 *    3. insert corresponding instruction using bound_checklist
 *
 */
bool llmpx::mpxPass(Module &module)
{
    if (llmpx_harden_cfi)
    {
        //global mpx linkage
        transform_global(module);
        errs()<<ANSI_COLOR_CYAN
            <<"--- harden CFI using MPX ---"
            <<ANSI_COLOR_RESET<<"\n";
        harden_cfi(module);
        errs()<<ANSI_COLOR_CYAN
            <<"--- verify ---"
            <<ANSI_COLOR_RESET<<"\n";
        verify(module);
        errs()<<ANSI_COLOR_CYAN
            <<"--- LLMPX DONE! ---"
            <<ANSI_COLOR_RESET<<"\n";
#if CUSTOM_STATISTICS
        dump_statistics();
#endif
        return false;
    }
#if AA_OPTIMIZATION
    errs()<<ANSI_COLOR_CYAN
        <<"--- mpxPass0: ipaa ---"
        <<ANSI_COLOR_RESET<<"\n";
    aapass(module);
#endif
    //collect safe access
    errs()<<ANSI_COLOR_CYAN
        <<"--- mpxPass0.1: collect_safe_access ---"
        <<ANSI_COLOR_RESET<<"\n";
    collect_safe_access(module);
    errs()<<ANSI_COLOR_CYAN
        <<"--- mpxPass1: transform_functions ---"
        <<ANSI_COLOR_RESET<<"\n";
    transform_functions(module);
    //add code to run on global variable
    //and determin their bound first
    errs()<<ANSI_COLOR_CYAN
        <<"--- mpxPass1.5: transform_global ---"
        <<ANSI_COLOR_RESET<<"\n";
    transform_global(module);
    errs()<<ANSI_COLOR_CYAN
        <<"--- mpxPass2: process_each_function ---"
        <<ANSI_COLOR_RESET<<"\n";
    process_each_function(module);
    errs()<<ANSI_COLOR_CYAN
        <<"--- mpxPass3: cleanup ---"
        <<ANSI_COLOR_RESET<<"\n";
    cleanup(module);
    errs()<<ANSI_COLOR_CYAN
        <<"--- mpxPass4: verify ---"
        <<ANSI_COLOR_RESET<<"\n";
    verify(module);
    errs()<<ANSI_COLOR_CYAN
        <<"--- LLMPX DONE! ---"
        <<ANSI_COLOR_RESET<<"\n";
    #if CUSTOM_STATISTICS
    dump_statistics();
    #endif
    return false;
}
///////////////////////////////////////////////////////////////////////////////
/*
 * Instruction Handler
 */
Value* llmpx::handleAlloca(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);
    Module* module = i->getParent()->getParent()->getParent();

    AllocaInst* alloca_inst = dyn_cast<AllocaInst>(i);

    #if DEBUG_HANDLE_ALLOCA
    errs()<<ANSI_COLOR_GREEN<<"AllocaInst:";
    alloca_inst->print(errs());
    errs()<<ANSI_COLOR_RESET<<"\n";
    #endif

    //operand of alloca need no check
    //
    if (alloca_inst->hasNUsesOrMore(1))
    {
        PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
        //get base
        Instruction* next_i = GetNextInstruction(i);
        IRBuilder<> builder(next_i);
        std::vector<Value *> args;
        Value* ptr_arg_for_bndmk 
            = builder.CreateBitCast(alloca_inst, Int8PtrTy,
                    alloca_inst->getName()+"_bitcast");
        blist->push_back(dyn_cast<Instruction>(ptr_arg_for_bndmk));

        args.push_back(ptr_arg_for_bndmk);

        //get dist(size-1)
        Type* allocated_type = alloca_inst->getAllocatedType();
        unsigned type_size = module
                            ->getDataLayout()
                            .getTypeAllocSize(allocated_type);
        #if DEBUG_HANDLE_ALLOCA
            errs()<<"Allocated Type:";
            allocated_type->dump();
            errs()<<"TypeAllocSize:"<<type_size<<"\n";
        #endif

        if(alloca_inst->isStaticAlloca())
        {
            #if DEBUG_HANDLE_ALLOCA
            errs()<<"This is static alloca\n";
            #endif

            unsigned allocated_size = type_size;

            if (alloca_inst->isArrayAllocation())
            {
                Value* array_size = alloca_inst->getArraySize();
                #if DEBUG_HANDLE_ALLOCA
                errs()<<"Alloca array size:";
                array_size->dump();
                #endif
                ConstantInt* cint_size = dyn_cast<ConstantInt>(array_size);
                assert( cint_size && "Can not cast array size into constantint?" );
                allocated_size *= cint_size->getValue().getZExtValue();
            }
            Constant* dist_arg_for_bndmk 
                = ConstantInt::get(Type::getInt64Ty(*ctx),(allocated_size-1));
            args.push_back(dist_arg_for_bndmk);
        }else
        {
            #if DEBUG_HANDLE_ALLOCA
            errs()<<ANSI_COLOR_RED<<"dynamic alloca!\n"<<ANSI_COLOR_RESET;
            #endif
            if (alloca_inst->isArrayAllocation())
            {
                Value* array_size = alloca_inst->getArraySize();
                #if DEBUG_HANDLE_ALLOCA
                errs()<<"Alloca array size:";
                array_size->dump();
                #endif
            }
            /*
             * a trick to get the size of allocated array
             * see http://nondot.org/sabre/LLVMNotes/
             *          SizeOf-OffsetOf-VariableSizedStructs.txt
             * for more details
             */
            IRBuilder<> builder(next_i);

            Value* allocated_size_0
                = builder.CreateGEP(
                    Constant::getNullValue(allocated_type->getPointerTo()),
                    alloca_inst->getArraySize());

            //FIXME:should be allocated_size_0 - 1....
            #if DEBUG_HANDLE_ALLOCA
            errs()<<" sizeof()=";
            allocated_size_0->dump();
            #endif

            Value* allocated_size_for_bndmk
                = builder.CreatePtrToInt(allocated_size_0,
                                    Type::getInt64Ty(*ctx));

            #if DEBUG_HANDLE_ALLOCA
            errs()<<" casted sizeof()=";
            allocated_size_for_bndmk->dump();
            #endif

            args.push_back(allocated_size_for_bndmk);
        }
        //create bndmk
        Instruction* bndmkcall 
            = CallInst::Create(mpx_bndmk, args,
                alloca_inst->getName()+".alc_bnd", next_i);
        blist->push_back(bndmkcall);
        TotalBNDMKAdded++;

        //key
        if (llmpx_enable_temporal_safety)
        {
            Instruction* lock
                = CallInst::Create(_llmpx_temporal_lock_alloca,
                        args, alloca_inst->getName()+"_key", next_i);
            klist->push_back(lock);
        }
    }else
    {
        //dead node to be deleted
        bound_checklist[ii] = &delete_ii;
        key_checklist[ii] = &delete_ii;
    }
    return ii;
}

Value* llmpx::handleBitCast(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);
    Module* module = i->getParent()->getParent()->getParent();

    //need to propogate bound information from operand
    #if DEBUG_HANDLE_BITCAST>2
    errs()<<"propogate bound for bitcast: ";
    ii->dump();
    #endif
    BitCastInst* bitcast_inst = dyn_cast<BitCastInst>(i);
    blist->push_back(NULL);
    if (llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
    Value* operand = bitcast_inst->getOperand(0);
    if (!operand->getType()->isPointerTy())
    {
        return ii;
    }
    if ((operand->getType()->isPointerTy() 
            && (!operand->getType()->getContainedType(0)->isFunctionTy())))
    {
        std::list<Value*> ilist
            = associate_meta(bitcast_inst, operand);
        blist->push_back(ilist.front());
        if(llmpx_enable_temporal_safety)
        {
            klist->push_back(ilist.back());
        }
        return ii;
    }
    /*
     * The operand is function, maybe its in the transform 
     * function list?
     */
    #if DEBUG_HANDLE_BITCAST
    errs()<<"This bitcast manipulate function type\n";
    ii->print(errs());
    errs()<<"\n";
    #endif
    Function* func = dyn_cast<Function>(operand);
    if (!func)
    {
        //anonymous function pointer
        return ii;
    }

    Function* new_func = find_transformed_function(func);
    if (!new_func)
    {
        return ii;
    }
    IRBuilder<> builder(bitcast_inst);
    Value* new_bitcast_inst 
        = builder.CreateBitCast(new_func, bitcast_inst->getType(),"");
    add_instruction_to_bcl(new_bitcast_inst);

    bitcast_inst->replaceAllUsesWith(new_bitcast_inst);
    bitcast_inst->removeFromParent();

    bound_checklist[new_bitcast_inst]->push_back(NULL);
    bound_checklist[ii] = &delete_ii;
    if (llmpx_enable_temporal_safety)
    {
        key_checklist[new_bitcast_inst]->push_back(NULL);
        key_checklist[ii] = &delete_ii;
    }
    return new_bitcast_inst;
}

Value* llmpx::handleCall(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);
    Module* module = i->getParent()->getParent()->getParent();

    /*
     * need to adhere to abi's calling convention
     * need to handle tail call, if we append bound information
     * we should not use tail call
     * also need to scan parameters for function pointer
     */
    CallInst* callinst = dyn_cast<CallInst>(i);
    assert( callinst && "Not a call inst???\n" );
    blist->push_back(NULL);
    if (llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
    #if DEBUG_HANDLE_CALL
    errs()<<ANSI_COLOR_GREEN<<"CallInst:";
    callinst->print(errs());
    errs()<<ANSI_COLOR_RESET<<"\n";
    #endif
    //need to replace with transformed function
    bool replace_func = false;
    //return bound
    bool ret_bound = false;
    //require bound as argument
    bool req_bound_for_arg = false;
    // may be external
    bool may_be_external = false;
    //may be function pointer
    bool may_be_function_pointer = false;
    //need to replace function ptr in argument
    bool need_to_replace_function_ptr_arg = false;
    bool need_to_cast_function_type = true;
    //this function has mpx wrapper
    bool has_wrapper = false;

    Function* called_func = callinst->getCalledFunction();
    Value* called_value = NULL;
    Value* new_func = NULL;

    //1. the function
    if (!called_func)
    {
        //this is a function pointer
        may_be_function_pointer = true;
        called_value = callinst->getCalledValue();
        if(isa<Instruction>(called_value))
        {
        #if DEBUG_HANDLE_CALL
            errs()<<"inst as function ptr: \n";
        #endif
            called_value = process_each_instruction(called_value);
        #if DEBUG_HANDLE_CALL
            errs()<<"Processed called_value as: ";
            called_value->dump();
        #endif
        }else
        {
            Value* real_called_func = called_value->stripPointerCasts();
            need_to_cast_function_type = true;

        #if DEBUG_HANDLE_CALL
            errs()<<"non-inst function ptr: \n";
            called_value->dump();
            errs()<<"stripped value:";
            real_called_func->dump();
        #endif
            called_func = dyn_cast<Function>(real_called_func);
            if (is_function_orig(called_func))
            {
                new_func = tr_flist[called_func];
                replace_func = true;
                #if DEBUG_HANDLE_CALL
                errs()<<"found transformed function: "
                        <<new_func->getName()
                        <<"\n";
                #endif
            }else
            {
                may_be_external = true;
            }
        }
    }else if((!!called_func) && (!is_function_orig(called_func)))
    {
        //external or internal, maybe no need to replace
        may_be_external = true;
    }else
    {
        //internal
        replace_func = true;
        new_func = find_transformed_function(called_func);
        #if DEBUG_HANDLE_CALL
        errs()<<"internal function, need to replace:"
                <<new_func->getName()
                <<"\n";
        #endif
    }

    if ((!!called_func) && is_in_wrapper_list(called_func->getName()))
    {
        has_wrapper = true;
        replace_func = true;
        called_value = orig_to_cw_flist[called_func];
        new_func = called_value;
        #if DEBUG_HANDLE_CALL
        errs()<<"has wrapper: ";
        called_value->dump();
        #endif
    }

    //2. return type
    if (isa<PointerType>(callinst->getType()))
    {
        ret_bound = true;
    }
    //3. arguments
    //construct argument and its corresponding parameter types
    std::vector<Value*> args;
    AttributeSet attr_set;
    CallingConv::ID ccid;
    std::vector<Value*> args_wobnd;
    std::vector<Value*> bndargs;
    std::vector<Value*> keyargs;
    /*
     * stores the argument type, which might be useful
     * for casting function pointer
     */
    std::vector<Type*> nb_rettype;
    //stores bnd types need to append
    std::vector<Type*> tempbndtype;
    //same thing for key
    std::vector<Type*> tempkeytype;

    ccid = callinst->getCallingConv();

    //errs()<<"-*- Process arguments -*-\n";
    attr_set = callinst->getAttributes();
    #if 0
    /*
     * FIXME: may need to fix these attributes
     */
    if (attr_set.hasAttribute(AttributeSet::ReturnIndex,
                    Attribute::Dereferenceable))
    {
        attr_set = attr_set.removeAttribute(*ctx, AttributeSet::ReturnIndex,
                                Attribute::Dereferenceable);
    }
    if (attr_set.hasAttribute(AttributeSet::ReturnIndex,
                    Attribute::NonNull))
    {
        attr_set = attr_set.removeAttribute(*ctx, AttributeSet::ReturnIndex,
                                Attribute::NonNull);
    }
    if (attr_set.hasAttribute(AttributeSet::ReturnIndex,
                    Attribute::NoAlias))
    {
        attr_set = attr_set.removeAttribute(*ctx, AttributeSet::ReturnIndex,
                                Attribute::NoAlias);
    }
    #endif
    //////////
    int num_arg_operands = callinst->getNumArgOperands();
    for (int oi = 0;oi<num_arg_operands;oi++)
    {
        Value* operand = callinst->getArgOperand(oi);
        args.push_back(operand);
        args_wobnd.push_back(operand);
        nb_rettype.push_back(operand->getType());
        #if DEBUG_HANDLE_CALL
        errs()<<"   arg "<<oi<<":";
        operand->dump();
        #endif
        if(isa<PointerType>(operand->getType()))
        {
            #if DEBUG_HANDLE_CALL
            errs()<<" is pointer\n";
            #endif
            req_bound_for_arg = true;
            Value* true_arg = operand->stripPointerCasts();
            if (isa<Instruction>(true_arg))
            {
                #if DEBUG_HANDLE_CALL
                errs()<<"   is instruction\n";
                #endif
                add_instruction_to_bcl(operand);
                operand = process_each_instruction(operand);
                assert(operand && "processed operand is NULL?");

                #if DEBUG_HANDLE_CALL
                errs()<<"      processed operand:";
                operand->dump();
                #endif
                args.pop_back();
                args.push_back(operand);
                args_wobnd.pop_back();
                args_wobnd.push_back(operand);
            }else if (isa<Function>(true_arg))
            {
                #if DEBUG_HANDLE_CALL
                errs()<<"   is function\n";
                #endif
                Value* new_operand = true_arg;
                Function* temp_func = dyn_cast<Function>(true_arg);
                #if DEBUG_HANDLE_CALL
                errs()<<temp_func->getName()<<"\n";
                #endif
                if (is_function_orig(temp_func))
                {
                    need_to_replace_function_ptr_arg = true;
                    new_operand = find_transformed_function(temp_func);
                    IRBuilder<> builder(callinst);
                    Value* new_arg = builder.CreateBitCast(new_operand, operand->getType());
                    args.pop_back();
                    args.push_back(new_arg);
                    args_wobnd.pop_back();
                    args_wobnd.push_back(new_arg);
                    operand = new_arg;
                }
            }
            operand = process_each_instruction(operand);
            if (!callinst->paramHasAttr(oi+1, Attribute::ByVal))
            {
                Value* bound = get_bound(operand, i);
                if (!bound)
                {
                    bound = get_infinite_bound(callinst);
                    //blist->splice(blist->end(), ilist);
                }
                #if 0
                errs()<<"bound:";
                bound->print(errs());
                errs()<<"\n";
                #endif
                bndargs.push_back(bound);
                tempbndtype.push_back(bound->getType());
                if (llmpx_enable_temporal_safety && (!has_wrapper))
                {
                    Value* key = get_key(operand, i);
                    if (!key)
                    {
                        key = get_anyvalid_key(callinst);
                    }
                    #if 0
                    errs()<<"key:";
                    key->print(errs());
                    errs()<<"\n";
                    #endif
                    keyargs.push_back(key);
                    tempkeytype.push_back(key->getType());
                }
            }else
            {
                #if DEBUG_HANDLE_CALL
                errs()<<" has byval attribute\n";
                #endif
            }
        }
    }
    //errs()<<"--------------------------\n";
    args.insert( args.end(), bndargs.begin(), bndargs.end() );
    nb_rettype.insert( nb_rettype.end(), tempbndtype.begin(), tempbndtype.end() );
    if (llmpx_enable_temporal_safety && (!has_wrapper))
    {
        args.insert(args.end(), keyargs.begin(), keyargs.end());
        nb_rettype.insert(nb_rettype.end(), tempkeytype.begin(), tempkeytype.end());
    }
    /*
     * Create call instruction according to the information we've collected
     * so far. Also need to consider whether we need to cast the function type
     * again because there may be function type cast in original call instruction
     * like 
     *  %call = tail call i32 (%struct.obstack*, i8*, ...)
     *          bitcast (i64 (%struct.obstack*, i8*)* @_obstack_allocated_p
     *                to i32 (%struct.obstack*, i8*, ...)*)
     *          (%struct.obstack* %3, i8* %object)
     */
    CallInst* nc;
    if (may_be_external && (!has_wrapper))
    {
       nc = callinst;
        if (need_to_replace_function_ptr_arg)
        {
            if(called_func)
                nc = CallInst::Create(called_func, args_wobnd, "", i);
            else
                nc = CallInst::Create(called_value, args_wobnd, "", i);

            nc->setAttributes(attr_set);
            nc->setCallingConv(ccid);

            bound_checklist[ii] = &delete_ii;
            ii->replaceAllUsesWith(nc);

            #if DEBUG_HANDLE_CALL
            errs()<<ANSI_COLOR_RED
                <<"erase call inst:";
            callinst->dump();
            errs()<<ANSI_COLOR_RESET;
            #endif
            callinst->removeFromParent();
        }
        blist->push_back(NULL);
        if(llmpx_enable_temporal_safety)
        {
            klist->push_back(NULL);
        }
        return nc;
    }

    if (ret_bound && (!replace_func) && (!req_bound_for_arg))
    {
        Value* bnd = get_infinite_bound(callinst);
        blist->push_back(bnd);
        if(llmpx_enable_temporal_safety)
        {
            Value* key = get_anyvalid_key(callinst);
            klist->push_back(key);
        }
        return ii;
    }

    std::string call_name = "";
    if(!callinst->getType()->isVoidTy())
    {
        call_name.append(callinst->getName().data());
        call_name.append("_wbnd");
    }
    if (may_be_function_pointer && called_func && replace_func && (!has_wrapper))
    {
        /*
         * means that this is a function pointer and we found its
         * original function, and we need to bitcast it
         */
        IRBuilder<> builder(callinst);
        Function* the_new_func = dyn_cast<Function>(new_func);
        /*
         * mutate return type of new function to match required return type
         */
        Type* good_return_type 
            = the_new_func->getFunctionType()->getReturnType();
        if (tr_flist_ret[called_func]==0)
        {
            //no bound was returned
            if (good_return_type != callinst->getType())
            {
                good_return_type = callinst->getType();
            }
        }else
        {
            /*
             * bound was returned,
             * the ret_val will be casted to desired type later
             */
        }

        /*
         * assemble new function type
         */
        Type * nbtype
            = FunctionType::get(
                    good_return_type,
                    nb_rettype, the_new_func->isVarArg())->getPointerTo();
        Value* nb_func = builder.CreateBitCast(new_func, nbtype, "");
        new_func = nb_func;
    }
    Type* nftype = NULL;
    if (replace_func)
    {
        nc = CallInst::Create(new_func, args, "", i);
        nftype = new_func->getType();
        tr_bndinfo_for_rettype[nftype] = tr_flist_ret[called_func];
    }else if(req_bound_for_arg)
    {
        if (called_func)
        {
            errs()<<"Require bound info for args:\n";
            errs()<<"Called function:";
            called_func->dump();
            llvm_unreachable("no way that internal function requires bound\n"
                            "and not transformed");
        }
        #if 0
        if (called_value)
        {
            errs()<<"Require bound info for args:\n";
            errs()<<"Called value:";
            called_value->dump();
        }
        #endif
        Type* ftype = called_value->getType()
                        ->getContainedType(0);
        nftype = transform_function_type(ftype);
        if (!nftype)
        {
            //vararg??
            return ii;
        }

        assert( nftype && "Not a valid function type??" );
        Type* called_value_type = nftype->getPointerTo();
        #if DEBUG_HANDLE_CALL>3
        errs()<<"DBG: transform ftype:";
        called_value->getType()->dump();
        errs()<<"     to ftype:";
        called_value_type->dump();
        errs()<<"args:\n";
        int acnt = 0;
        for (auto ait = args.begin(); ait!=args.end(); ++ait)
        {
            errs()<<"   "<<acnt<<": ";
            acnt++;
            Value*v = *ait;
            v->dump();
        }
        errs()<<"\n";
        #endif
        IRBuilder<> builder(callinst);
        Value* new_called_value 
            = builder.CreateBitCast(called_value, called_value_type, "");
        //llvm_unreachable("need to bitcast function type for function pointer");
        nc = CallInst::Create(new_called_value, args, "", i);
    }else
    {
        //no need to replace anything.
        return ii;
    }

    nc->setAttributes(attr_set);
    nc->setCallingConv(ccid);

    nc->setDebugLoc(callinst->getDebugLoc());
    //DO NOT USE tail call here
    //nc->setTailCall(callinst->isTailCall());
    //nc->setTailCallKind(callinst->getTailCallKind());
    //create new entry for new function call
    add_instruction_to_bcl(nc);
    blist = bound_checklist[nc];
    klist = key_checklist[nc];

    /*
     * for all function that do not return bound
     * we can replace the use directly, because the return type
     * hasn't been changed,
     * otherwise we need to insert instruction to extract 
     * original type value and bound
     * The bound will be associated/tied whenever needed
     */
    Value* ret_val = nc;
    #if DEBUG_HANDLE_CALL
    errs()<<ANSI_COLOR_MAGENTA<<"new call?="<<ANSI_COLOR_RESET;
    nc->dump();
    #endif
    if (tr_bndinfo_for_rettype[nftype]==0)
    {
        callinst->replaceAllUsesWith(nc);
        //no return bound, add NULL to prevent it from further processing
        blist->push_back(NULL);
        if (llmpx_enable_temporal_safety)
        {
            klist->push_back(NULL);
        }
    }else
    {
        //errs()<<"   return with bound, extract orig value into ret_val";
        ret_val 
            = ExtractValueInst::Create(nc, 0, "", GetNextInstruction(nc));
        blist->push_back(nc);
        if (llmpx_enable_temporal_safety)
        {
            klist->push_back(nc);
        }
        //associate bound
        add_instruction_to_bcl(ret_val);
        std::list<Value*>* nblist = bound_checklist[ret_val];
        std::list<Value*>* nklist = key_checklist[ret_val];
        /*
         * if the return value is pointer type,
         * we associate its bound directly to avoid 
         * further processing..
         */
        if (isa<PointerType>(ret_val->getType()))
        {
            Value* ret_val_bnd 
                = ExtractValueInst::Create(nc, 1, "",
                        GetNextInstruction(dyn_cast<Instruction>(ret_val)));
            nblist->push_back(ret_val_bnd);
            if(llmpx_enable_temporal_safety)
            {
                Value* ret_val_key;
                if(has_wrapper)
                {
                    ret_val_key = get_anyvalid_key(callinst);
                }else
                {
                    ret_val_key
                        = ExtractValueInst::Create(nc, 2, "",
                            GetNextInstruction(dyn_cast<Instruction>(ret_val)));
                }
                nklist->push_back(ret_val_key);
            }
        }
        if (callinst->getType() != ret_val->getType())
        {
            //need to bitcast...
            #if (DEBUG_HANDLE_CALL_CAST)
                errs()<<ANSI_COLOR_YELLOW<<"need to bitcast new call result\n";
                errs()<<"target type:";
                callinst->getType()->dump();
                errs()<<"source type:";
                ret_val->getType()->dump();
                errs()<<"nc type:";
                nc->getType()->dump();
                nc->dump();
                callinst->dump();
                errs()<<ANSI_COLOR_RESET;
            #endif
            IRBuilder<> builder(GetNextInstruction(dyn_cast<Instruction>(ret_val)));
            Value* bc_retval = builder.CreateBitOrPointerCast(ret_val, callinst->getType());
            ret_val = bc_retval;
        }
        callinst->replaceAllUsesWith(ret_val);
    }
    /*
     * mark this instruction to be erased from parent
     */
    bound_checklist[ii] = &delete_ii;
    ii->replaceAllUsesWith(ret_val);

    #if DEBUG_HANDLE_CALL
    errs()<<ANSI_COLOR_RED
        <<"erase call inst:";
    callinst->dump();
    errs()<<ANSI_COLOR_RESET;
    #endif
    callinst->removeFromParent();

    return ret_val;
}

/*
 * invoke instruction is only used for C++ code?
 * the way we handle invoke instruction is very similar to callinst
 * except that invoke instruction has NormalDest and UnwindDest
 * we need to set that, when create new invoke instruction
 * and when invoke instruction returns bound, we need to insert 
 * extract value instruction at NormalDest
 * UnwindDest is for exception handling and the invoke result shoud
 * not be used in UnwindDest(not even returned),
 * so that we don't bother handling it there
 */
Value* llmpx::handleInvoke(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);
    Module* module = i->getParent()->getParent()->getParent();

    /*
     * need to adhere to abi's calling convention
     * need to handle tail call, if we append bound information
     * we should not use tail call
     * also need to scan parameters for function pointer
     */
    InvokeInst* invokeinst = dyn_cast<InvokeInst>(i);
    assert( invokeinst && "Not an invoke inst???\n" );
    blist->push_back(NULL);
    if (llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
    #if DEBUG_HANDLE_INVOKE>3
    errs()<<ANSI_COLOR_GREEN
        <<"InvokeInst:"
        <<ANSI_COLOR_RESET;
    invokeinst->print(errs());
    errs()<<"\n";
    #endif
    BasicBlock* normal_dest = invokeinst->getNormalDest();
    BasicBlock* unwind_dest = invokeinst->getUnwindDest();

    #if DEBUG_HANDLE_INVOKE>3
    errs()<<"Current BB is:"<<i->getParent()->getName()<<"\n";
    #endif

    //need to replace with transformed function
    bool replace_func = false;
    //return bound
    bool ret_bound = false;
    //require bound as argument
    bool req_bound_for_arg = false;
    // may be external
    bool may_be_external = false;
    //may be function pointer
    bool may_be_function_pointer = false;
    //need to replace function ptr in argument
    bool need_to_replace_function_ptr_arg = false;
    bool need_to_cast_function_type = true;
    //this function has mpx wrapper
    bool has_wrapper = false;

    Function* called_func = invokeinst->getCalledFunction();
    Value* called_value = NULL;
    Value* new_func = NULL;

    //1. the function
    if (!called_func)
    {
        //this is a function pointer
        may_be_function_pointer = true;
        called_value = invokeinst->getCalledValue();
        if(isa<Instruction>(called_value))
        {
        #if DEBUG_HANDLE_INVOKE>3
            errs()<<"inst as function ptr: \n";
        #endif
            called_value = process_each_instruction(called_value);
        #if DEBUG_HANDLE_INVOKE>3
            errs()<<"Processed called_value as: ";
            called_value->dump();
        #endif
        }else
        {
            Value* real_called_func = called_value->stripPointerCasts();
            need_to_cast_function_type = true;

        #if DEBUG_HANDLE_INVOKE>3
            errs()<<"non-inst function ptr: \n";
            called_value->dump();
            errs()<<"stripped value:";
            real_called_func->dump();
        #endif
            called_func = dyn_cast<Function>(real_called_func);
            if (is_function_orig(called_func))
            {
                new_func = tr_flist[called_func];
                replace_func = true;
                #if DEBUG_HANDLE_INVOKE>3
                errs()<<"found transformed function: "
                        <<new_func->getName()
                        <<"\n";
                #endif
            }else
            {
                may_be_external = true;
            }
        }
    }else if((!!called_func) && (!is_function_orig(called_func)))
    {
        //external or internal, maybe no need to replace
        may_be_external = true;
    }else
    {
        //internal
        replace_func = true;
        new_func = find_transformed_function(called_func);
        #if DEBUG_HANDLE_INVOKE>3
        errs()<<"internal function, need to replace:"
                <<new_func->getName()
                <<"\n";
        #endif
    }

    if ((!!called_func) && is_in_wrapper_list(called_func->getName()))
    {
        has_wrapper = true;
        replace_func = true;
        called_value = orig_to_cw_flist[called_func];
        new_func = called_value;
        #if DEBUG_HANDLE_INVOKE
        errs()<<"has wrapper: ";
        called_value->dump();
        #endif
    }

    //2. return type
    if (isa<PointerType>(invokeinst->getType()))
    {
        ret_bound = true;
    }
    //3. arguments
    //construct argument and its corresponding parameter types
    std::vector<Value*> args;
    AttributeSet attr_set;
    CallingConv::ID ccid;
    std::vector<Value*> args_wobnd;
    std::vector<Value*> bndargs;
    std::vector<Value*> keyargs;
    /*
     * stores the argument type, which might be useful
     * for casting function pointer
     */
    std::vector<Type*> nb_rettype;
    //stores bnd types need to append
    std::vector<Type*> tempbndtype;
    //same thing for key
    std::vector<Type*> tempkeytype;

    ccid = invokeinst->getCallingConv();
    //errs()<<"-*- Process arguments -*-\n";
    attr_set = invokeinst->getAttributes();
    #if 0
    if (attr_set.hasAttribute(AttributeSet::ReturnIndex,
                    Attribute::Dereferenceable))
    {
        attr_set = attr_set.removeAttribute(*ctx, AttributeSet::ReturnIndex,
                                Attribute::Dereferenceable);
    }
    if (attr_set.hasAttribute(AttributeSet::ReturnIndex,
                    Attribute::NonNull))
    {
        attr_set = attr_set.removeAttribute(*ctx, AttributeSet::ReturnIndex,
                                Attribute::NonNull);
    }
    if (attr_set.hasAttribute(AttributeSet::ReturnIndex,
                    Attribute::NoAlias))
    {
        attr_set = attr_set.removeAttribute(*ctx, AttributeSet::ReturnIndex,
                                Attribute::NoAlias);
    }
    #endif
    int num_arg_operands = invokeinst->getNumArgOperands();
    for (int oi = 0;oi<num_arg_operands;oi++)
    {
        Value* operand = invokeinst->getArgOperand(oi);
        args.push_back(operand);
        args_wobnd.push_back(operand);
        nb_rettype.push_back(operand->getType());
        #if DEBUG_HANDLE_INVOKE>3
        errs()<<"   arg "<<oi<<":";
        operand->dump();
        #endif
        if(isa<PointerType>(operand->getType()))
        {
            #if DEBUG_HANDLE_INVOKE>3
            errs()<<" is pointer\n";
            #endif
            req_bound_for_arg = true;
            Value* true_arg = operand->stripPointerCasts();
            if (isa<Instruction>(true_arg))
            {
                #if DEBUG_HANDLE_INVOKE>3
                errs()<<"   is instruction\n";
                #endif
                add_instruction_to_bcl(operand);
                operand = process_each_instruction(operand);
                assert(operand && "processed operand is NULL?");

                #if DEBUG_HANDLE_INVOKE>3
                errs()<<"      processed operand:";
                operand->dump();
                #endif
                args.pop_back();
                args.push_back(operand);
                args_wobnd.pop_back();
                args_wobnd.push_back(operand);
            }else if (isa<Function>(true_arg))
            {
                #if DEBUG_HANDLE_INVOKE>3
                errs()<<"   is function\n";
                #endif
                Value* new_operand = true_arg;
                Function* temp_func = dyn_cast<Function>(true_arg);
                #if DEBUG_HANDLE_INVOKE>3
                errs()<<temp_func->getName()<<"\n";
                #endif
                if (is_function_orig(temp_func))
                {
                    need_to_replace_function_ptr_arg = true;
                    new_operand = find_transformed_function(temp_func);
                    IRBuilder<> builder(invokeinst);
                    Value* new_arg = builder.CreateBitCast(new_operand, operand->getType());
                    args.pop_back();
                    args.push_back(new_arg);
                    args_wobnd.pop_back();
                    args_wobnd.push_back(new_arg);
                    operand = new_arg;
                }
            }
            operand = process_each_instruction(operand);
            if (!invokeinst->paramHasAttr(oi+1, Attribute::ByVal))
            {
                Value* bound = get_bound(operand, i);
                if (!bound)
                {
                    bound = get_infinite_bound(invokeinst);
                    //blist->splice(blist->end(), ilist);
                }
                #if 0
                errs()<<"bound:";
                bound->print(errs());
                errs()<<"\n";
                #endif
                bndargs.push_back(bound);
                tempbndtype.push_back(bound->getType());
                if (llmpx_enable_temporal_safety && (!has_wrapper))
                {
                    Value* key = get_key(operand, i);
                    if (!key)
                    {
                        key = get_anyvalid_key(invokeinst);
                    }
                    #if 0
                    errs()<<"key:";
                    key->print(errs());
                    errs()<<"\n";
                    #endif
                    keyargs.push_back(key);
                    tempkeytype.push_back(key->getType());
                }
            }
        }
    }
    //errs()<<"--------------------------\n";
    args.insert( args.end(), bndargs.begin(), bndargs.end() );
    nb_rettype.insert( nb_rettype.end(), tempbndtype.begin(), tempbndtype.end() );
    if (llmpx_enable_temporal_safety && (!has_wrapper))
    {
        args.insert(args.end(), keyargs.begin(), keyargs.end());
        nb_rettype.insert(nb_rettype.end(), tempkeytype.begin(), tempkeytype.end());
    }
    InvokeInst* nc;
    std::string basename = std::string(invokeinst->hasName()?invokeinst->getName():"");

    if (may_be_external && (!has_wrapper))
    {
        nc = invokeinst;
        if (need_to_replace_function_ptr_arg)
        {
            if(called_func)
                nc = InvokeInst::Create(called_func, normal_dest, unwind_dest,
                                        args_wobnd, basename, i);
            else
                nc = InvokeInst::Create(called_value, normal_dest, unwind_dest,
                                        args_wobnd, basename, i);
            
            nc->setAttributes(attr_set);
            nc->setCallingConv(ccid);

            bound_checklist[ii] = &delete_ii;
            ii->replaceAllUsesWith(nc);

            #if DEBUG_HANDLE_INVOKE>3
            errs()<<ANSI_COLOR_RED
                <<"erase invoke inst:";
            invokeinst->dump();
            errs()<<ANSI_COLOR_RESET;
            #endif
            invokeinst->removeFromParent();
        }
        blist->push_back(NULL);
        if(llmpx_enable_temporal_safety)
        {
            klist->push_back(NULL);
        }
        return nc;
    }

    if (ret_bound && (!replace_func) && (!req_bound_for_arg))
    {
        Value* bnd = get_infinite_bound(invokeinst);
        blist->push_back(bnd);
        if(llmpx_enable_temporal_safety)
        {
            Value* key = get_anyvalid_key(invokeinst);
            klist->push_back(key);
        }
        return ii;
    }

    std::string call_name = "";
    if(!invokeinst->getType()->isVoidTy())
    {
        call_name.append(invokeinst->getName().data());
        call_name.append("_wbnd");
    }
    if (may_be_function_pointer && called_func && replace_func && (!has_wrapper))
    {
        /*
         * means that this is a function pointer and we found its
         * original function, and we need to bitcast it
         */
        IRBuilder<> builder(invokeinst);
        Function* the_new_func = dyn_cast<Function>(new_func);
        /*
         * mutate return type of new function to match required return type
         */
        Type* good_return_type 
            = the_new_func->getFunctionType()->getReturnType();
        if (tr_flist_ret[called_func]==0)
        {
            //no bound was returned
            if (good_return_type != invokeinst->getType())
            {
                good_return_type = invokeinst->getType();
            }
        }else
        {
            /*
             * bound was returned,
             * the ret_val will be casted to desired type later
             */
        }
        /*
         * assemble new function type
         */
        Type * nbtype
            = FunctionType::get(
                    good_return_type,
                    nb_rettype, the_new_func->isVarArg())->getPointerTo();
        Value* nb_func = builder.CreateBitCast(new_func, nbtype, "");
        new_func = nb_func;
    }
    Type* nftype = NULL;
    if (replace_func)
    {
        nc = InvokeInst::Create(new_func, normal_dest, unwind_dest,
                                args, basename, i);
        nftype = new_func->getType();
        tr_bndinfo_for_rettype[nftype] = tr_flist_ret[called_func];
    }else if(req_bound_for_arg)
    {
        if (called_func)
        {
            errs()<<"Require bound info for args:\n";
            errs()<<"Called function:";
            called_func->dump();
            llvm_unreachable("no way that internal function requires bound\n"
                            "and not transformed");
        }
        #if 0
        if (called_value)
        {
            errs()<<"Require bound info for args:\n";
            errs()<<"Called value:";
            called_value->dump();
        }
        #endif
        Type* ftype = called_value->getType()
                        ->getContainedType(0);
        nftype = transform_function_type(ftype);

        if (!nftype)
        {
            //vararg??
            return ii;
        }

        assert( nftype && "Not a valid function type??" );
        Type* called_value_type = nftype->getPointerTo();
        IRBuilder<> builder(invokeinst);
        Value* new_called_value 
            = builder.CreateBitCast(called_value, called_value_type, "");
        //llvm_unreachable("need to bitcast function type for function pointer");
        nc = InvokeInst::Create(new_called_value, normal_dest, unwind_dest,
                                args, basename, i);
    }else
    {
        //no need to replace anything.
        return ii;
    }

    nc->setAttributes(attr_set);
    nc->setCallingConv(ccid);

    nc->setDebugLoc(invokeinst->getDebugLoc());
    //DO NOT USE tail call here
    //nc->setTailCall(invokeinst->isTailCall());
    //nc->setTailCallKind(invokeinst->getTailCallKind());
    //create new entry for new function call
    add_instruction_to_bcl(nc);
    blist = bound_checklist[nc];
    klist = key_checklist[nc];

    /*
     * for all function that do not return bound
     * we can replace the use directly, because the return type
     * hasn't been changed,
     * otherwise we need to insert instruction to extract 
     * original type value and bound
     * The bound will be associated/tied whenever needed
     */
    Value* ret_val = nc;
    if (tr_bndinfo_for_rettype[nftype]==0)
    {
        invokeinst->replaceAllUsesWith(nc);
        //no return bound, add NULL to prevent it from further processing
        blist->push_back(NULL);
        if (llmpx_enable_temporal_safety)
        {
            klist->push_back(NULL);
        }
    }else
    {
        /*
         * NOTE: we can not insert instructions at normal_dest
         * for that invoke instruction may not dominate normal_dest
         * the result is that the inserted instruction at normal dest will 
         * not be dominated by invoke instruction.
         * solution: insert a new BB and replace normal dest of invoke
         * inst to the new BB
         */
        BasicBlock* newbb
            = BasicBlock::Create(*ctx, "msdist",
                invokeinst->getParent()->getParent());

        assert(newbb->getParent() && "newbb has no parent!");
        IRBuilder<> builder0(newbb);

        //errs()<<"   return with bound, extract orig value into ret_val";
        ret_val
            = builder0.CreateExtractValue(nc, 0, basename+"_oret");
        blist->push_back(nc);
        if (llmpx_enable_temporal_safety)
        {
            klist->push_back(nc);
        }
        //associate bound
        add_instruction_to_bcl(ret_val);
        std::list<Value*>* nblist = bound_checklist[ret_val];
        std::list<Value*>* nklist = key_checklist[ret_val];
        //insert place holder
        nblist->push_back(NULL);
        if (llmpx_enable_temporal_safety)
        {
            nklist->push_back(NULL);
        }
        /*
         * if the return value is pointer type,
         * we associate its bound directly to avoid 
         * further processing..
         */
        if (isa<PointerType>(ret_val->getType()))
        {
            Value* ret_val_bnd
                = builder0.CreateExtractValue(nc, 1, basename+"_bnd");
            nblist->push_back(ret_val_bnd);
            if(llmpx_enable_temporal_safety)
            {
                Value* ret_val_key;
                if (has_wrapper)
                {
                    ret_val_key = get_anyvalid_key(invokeinst);
                }else
                {
                    ret_val_key
                        = builder0.CreateExtractValue(nc, 2, basename+"_key");
                }
                nklist->push_back(ret_val_key);
            }
        }else
        {
            //need to deal with aggregated type
            errs()<<ANSI_COLOR_RED
                <<"non-pointer type returned with bound/key?\n"
                <<ANSI_COLOR_RESET;
            //llvm_unreachable("non-pointer type returned with bound/key?");
        }
        if (invokeinst->getType() != ret_val->getType())
        {
            //need to bitcast...
            #if (DEBUG_HANDLE_INVOKE_CAST)
                errs()<<ANSI_COLOR_YELLOW<<"need to bitcast new invoke result\n";
                errs()<<"target type:";
                invokeinst->getType()->dump();
                errs()<<"source type:";
                ret_val->getType()->dump();
                errs()<<"nc type:";
                nc->getType()->dump();
                nc->dump();
                invokeinst->dump();
                errs()<<ANSI_COLOR_RESET;
            #endif
            Value* bc_retval = builder0.CreateBitOrPointerCast(ret_val, invokeinst->getType());
            ret_val = bc_retval;
        }
        ////////////////////////////////////////////////////
        /*
         * NOTE: examine each use of the invoke instruction,
         *       fix incoming block for PHINode especially
         */
        BasicBlock* curbb = invokeinst->getParent();
        for (Value::use_iterator ui = invokeinst->use_begin(),
                                ue = invokeinst->use_end();
                                ui!=ue;)
        {
            Use &use = *ui++;
            Value* uv = cast<Instruction>(use.getUser());
            if (!isa<PHINode>(uv))
            {
                use.set(ret_val);
                continue;
            }
            PHINode* phi = dyn_cast<PHINode>(uv);
            for (int icidx = 0; icidx<phi->getNumIncomingValues(); icidx++)
            {
                Value* incomval = phi->getIncomingValue(icidx);
                if(incomval!=invokeinst)
                {
                    continue;
                }
                phi->setIncomingValue(icidx, ret_val);
                if(phi->getIncomingBlock(icidx)==curbb)
                {
                    phi->setIncomingBlock(icidx, newbb);
                }
            }
        }
        //examine the rest of phinode in succesor
        for (BasicBlock::iterator ins = normal_dest->begin();
            isa<PHINode>(ins); ++ins)
        {
            PHINode* phi = dyn_cast<PHINode>(ins);
            for (int icidx = 0; icidx<phi->getNumIncomingValues(); icidx++)
            {
                if (phi->getIncomingBlock(icidx)==curbb)
                {
                    phi->setIncomingBlock(icidx, newbb);
                }
            }
        }
        ////////////////////////////////////////////////////
        BranchInst *brinst
            = builder0.CreateBr(normal_dest);
        assert(brinst->getParent() && "brinst in newbb has no parent!");

        nc->setNormalDest(newbb);
        #if DEBUG_HANDLE_INVOKE>3
        errs()<<ANSI_COLOR_MAGENTA<<"new invoke ="<<ANSI_COLOR_RESET;
        nc->dump();
        #endif

        #if DEBUG_HANDLE_INVOKE
        errs()<<"NEW BB:";
        newbb->dump();
        #endif
    }
    /*
     * mark this instruction to be erased from parent
     */
    bound_checklist[ii] = &delete_ii;
    ii->replaceAllUsesWith(ret_val);

    #if DEBUG_HANDLE_INVOKE>3
    errs()<<ANSI_COLOR_RED
        <<"erase invoke inst:";
    invokeinst->dump();
    errs()<<ANSI_COLOR_RESET;
    #endif
    invokeinst->removeFromParent();

    return ret_val;
}

/*
 * FIXME: don't know exactly how to handle this..
 */
Value* llmpx::handleInsertElement(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);
    Module* module = i->getParent()->getParent()->getParent();

    blist->push_back(NULL);
    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
   
    return ii;
}

Value* llmpx::handleExtractElement(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);
    Module* module = i->getParent()->getParent()->getParent();

    blist->push_back(NULL);
    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
   
    return ii;
}

Value* llmpx::handleExtractValue(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    #if DEBUG_HANDLE_EXTRACTVALUE>2
    errs()<<"handleExtractValue:\n";
    ii->dump();
    #endif

    Instruction* i = dyn_cast<Instruction>(ii);
    Module* module = i->getParent()->getParent()->getParent();
    ExtractValueInst* evi = dyn_cast<ExtractValueInst>(i);
    //only interested in pointer type
    if (!isa<PointerType>(evi->getType()))
    {
        return ii;
    }
    Value* defval = evi->getAggregateOperand();
    /*
     * TODO: if aggregated value is returned from call
     * instruction and pointer is in aggregated value,
     * bound should also be returned.
     */
    //add placeholder
    blist->push_back(NULL);
    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
    /*
    while(true)
    {
        Instruction* definst = dyn_cast<Instruction>(defval);
        switch(definst->getOpcode())
        {
            default:
                    errs()<<"Need to deal with "
                            << definst->getOpcode()
                            <<"\n";
                    break;
            Instruction::ExtractValueInst:
            {
                break;
            }
            Instruction::CallInst:
            {
                break;
            }
        }
    }*/
    return ii;
}

/*
 * insertvalue instruction is special,
 * because it result in a struct/aggregated
 * type which may contains pointer,
 * we need to process the pointer operand if any
 */
Value* llmpx::handleInsertValue(Value* ii)
{
    Instruction* i = dyn_cast<Instruction>(ii);
    InsertValueInst* ivinst = dyn_cast<InsertValueInst>(i);
    Value* operand = ivinst->getInsertedValueOperand();
    if(operand->getType()->isPointerTy())
    {
        process_each_instruction(operand);
    }
    return ii;
}

Value* llmpx::handleGetElementPtr(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);
    GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(i);
    /*
     * the result of GEP instruction shares the same bound
     * of its operator
     */
#if DEBUG_HANDLE_GEP
    errs()<<ANSI_COLOR_CYAN
        <<"propogate bound for GEP\n"
        <<ANSI_COLOR_RESET;
    gep->dump();
    errs()<<" pointer operand is ?";
    gep->getPointerOperand()->dump();
#endif
#if 0
    /*
     * use infinite bound for all GEP
     */
    blist->push_back(NULL);
    blist->push_back(get_infinite_bound(gep));
    if (llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
        klist->push_back(get_anyvalid_key(gep));
    }
#else
    blist->push_back(NULL);
    std::list<Value*> ilist 
            = associate_meta(gep, gep->getPointerOperand());
    blist->push_back(ilist.front());

    if (llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
        klist->push_back(ilist.back());
    }
#endif
    return ii;
}

Value* llmpx::handleIntToPtr(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);

    /*
     * associate infinite bound to casted ptr
     */
    
    Type* X86_BNDTy = Type::getX86_BNDTy(*ctx);
    PointerType* X86_BNDPtrTy = Type::getX86_BNDPtrTy(*ctx);

    IRBuilder<> builder(GetNextInstruction(i));
    Value* bndptr 
        = ConstantExpr::getPointerCast(bnd_infinite, X86_BNDPtrTy);
    blist->push_back(bndptr);
    Value* bnd = builder.CreateLoad(X86_BNDTy,bndptr,"");
    blist->push_back(bnd);

    if (llmpx_enable_temporal_safety)
    {
        Value* key = get_anyvalid_key(GetNextInstruction(i));
        klist->push_back(key);
    }

    return ii;
}

/*
 * NOTE: need to handle ugly load, which cast pointer to int for store
 * FIXME: we need to keep bound load near when its is needed,
 *        as there are cases like this:
 *          load(p)
 *          bnd0 = bndldx p
 *          store p
 *          bndstx p bnd1
 *          gep p 
 *          bndcl p, bnd0
 *        this may get stall bound
 */
Value* llmpx::handleLoad(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);

    LoadInst* load_inst = dyn_cast<LoadInst>(i);
    Value* ptr_operand = load_inst->getPointerOperand();

    /*
     * don't check accesses from different address space?
     */
    if (ptr_operand)
    {
        Type *PtrTy = cast<PointerType>(ptr_operand->getType()->getScalarType());
        if (PtrTy->getPointerAddressSpace()!=0)
        {
            llvm_unreachable("Found ptr belongs to another address space");
        }
    }
    blist->push_back(NULL);
    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
#if DEBUG_HANDLE_LOAD
    errs()<<"handleLoad:";
    i->dump();
#endif
    //
    //external linkage
    if (GlobalVariable* po = dyn_cast<GlobalVariable>(ptr_operand))
    {
        if (po->getLinkage()==GlobalValue::ExternalLinkage)
        {
            //associate infinite bound to return value...
            if (isa<PointerType>(load_inst->getType())
                && load_inst->hasNUsesOrMore(1))
            {
                Type* X86_BNDTy = Type::getX86_BNDTy(*ctx);
                PointerType* X86_BNDPtrTy = Type::getX86_BNDPtrTy(*ctx);

                IRBuilder<> builder(GetNextInstruction(i));
                Value* bndptr 
                    = ConstantExpr::getPointerCast(bnd_infinite, X86_BNDPtrTy);
                blist->push_back(bndptr);
                Value* bnd = builder.CreateLoad(X86_BNDTy,bndptr,"");
                blist->push_back(bnd);
                if (llmpx_enable_temporal_safety)
                {
                    klist->push_back(NULL);
                }
            }
            return ii;
        }
    }
    //insert check if needed
    std::list<Value*> ilist;
    if (!is_safe_access_cache(load_inst))
    {
        ilist = insert_check(i, ptr_operand, true);
        blist->splice(blist->end(), ilist);
    }else
    {
        ElimSafeAccess++;
#if DEBUG_HANDLE_LOAD
        errs()<<"Safe Access found for handleLoad\n";
#endif
    }
    /*
     * for returned value
     * if it is used, we need to associate bound
     */
    if (!load_inst->hasNUsesOrMore(1))
    {
        return ii;
    }
    Type* val_type = load_inst->getType();
    if (val_type->isFloatingPointTy()
        || val_type->isFPOrFPVectorTy()
        || val_type->isVectorTy())
    {
        return ii;
    }

    bool load_ptr_as_non_ptr = false;
    bool load_ptr_as_ptr = false;
    if (load_inst->getType()->isPointerTy())
    {
        load_ptr_as_ptr = true;
    }else
    {
        //just in case of ugly load
        Value* ldptr = load_inst->getPointerOperand();
        if (find_actual_type(ldptr, true)->isPointerTy())
        {
            load_ptr_as_non_ptr = true;
        }
    }

    if (load_ptr_as_ptr)
    {
        //errs()<<"LoadInst return ptr, need to associate bound\n";
        /*
         * when loaded value is a pointer,
         * the bound should have been stored in MPX BT(bound table)
         * insert an bndldx here to load bound from BT
         */
        Instruction* insertion_point = GetNextInstruction(i);
        std::list<Value*> ilist = 
            insert_bound_load(insertion_point,
                                load_inst->getPointerOperand(),
                                load_inst);
        blist->splice(blist->end(), ilist);
        /*
         * key load should comes after bndldx
         * which is the same insertion point
         * as is used in insert_bound_load
         */
        if (llmpx_enable_temporal_safety)
        {
            ilist = insert_key_load(insertion_point,
                        load_inst->getPointerOperand());
            klist->splice(klist->end(), ilist);
        }
    }else
    {

    }
    return ii;
}
//TODO: add key population logic for PHINode
Value* llmpx::handlePHINode(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);

    /*
     * need to handle special cyclic dependency
     */
    PHINode* phi_node = dyn_cast<PHINode>(i);
    //avoid re-processing??
    blist->push_back(NULL);
    if (llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }
    //we only care about the case when PHINode's result is pointer
    //maybe we also need to care about aggregate type?
    if (!isa<PointerType>(phi_node->getType()))
    {
        return ii;
    }
    #if DEBUG_HANDLE_PHINODE>3
    errs()<<ANSI_COLOR_GREEN<<" - Instrument PHINode: ";
    phi_node->print(errs());
    errs()<<ANSI_COLOR_RESET<<"\n";
    #endif
    /*
     * if incoming value if function pointer,
     * we match for its transformed function
     * and force-cast the function type
     */
    if (this_phi_node_need_transform(phi_node))
    {
        #if DEBUG_HANDLE_PHINODE
        errs()<<"FUCK! this PHI has function type!\n"
              <<" means that function pointer is used here!\n";
        phi_node->print(errs());
        errs()<<"\n";
        #endif
        unsigned incoming_val_cnt = phi_node->getNumIncomingValues();
        for(int j=0;j<incoming_val_cnt;j++)
        {
            Value* incom_val = phi_node->getIncomingValue(j);
            BasicBlock* incom_bb = phi_node->getIncomingBlock(j);
            Function* incom_func = dyn_cast<Function>(incom_val);

            if((!!incom_func) && is_function_orig(incom_func))
            {
                Function* nfunc = tr_flist[incom_func];
                IRBuilder<> builder(incom_bb->getTerminator());
                Value* nfptr 
                    = builder.CreateBitCast(nfunc, incom_func->getType(), "");
                phi_node->setIncomingValue(j, nfptr);
            }else
            {
                #if DEBUG_HANDLE_PHINODE
                errs()<<" the "<<j<<"-th "
                    <<"incoming value is not function or can not"
                    <<" be found in tr_flist?\n";
                #endif
            }
        }
    }

    //insert phi node for bound associated with incoming value
    unsigned incoming_val_cnt = phi_node->getNumIncomingValues();
    Type* bndty = Type::getX86_BNDTy(*ctx);
    PHINode* bnd_phi_node 
            = PHINode::Create(bndty,
                            incoming_val_cnt,
                            "bnd_phi."+phi_node->getName(),
                            GetNextInstruction(phi_node));
    blist->push_back(bnd_phi_node);
    /*
     * there should be only one distinct incoming value for each
     * incoming basicblock
     */
    //set incoming value for bound phi node
    for(int j=0; j<incoming_val_cnt; j++)
    {
        Value* incom_val = phi_node->getIncomingValue(j);
        incom_val = process_each_instruction(incom_val);

        BasicBlock* incom_bb = phi_node->getIncomingBlock(j);
        Value* bnd;
        /*
         * we need to get the bound, if need to insert instruction
         * it should be located at the end of the incoming bb
         */
        bnd = get_bound(incom_val, incom_bb->getTerminator());

        if (bnd==NULL)
        {
            bnd = get_infinite_bound(incom_bb->getTerminator());
        }
        //bnd->print(errs());errs()<<"\n";
        assert((bnd->getType() && bnd->getType()->isX86_BNDTy()) && "bad bnd type?");
        bnd_phi_node->addIncoming(bnd, incom_bb);
    }
    blist->push_back(bnd_phi_node);
    #if DEBUG_HANDLE_PHINODE>3
    errs()<<ANSI_COLOR_RED<<" |- bnd for phi is: ";
    bnd_phi_node->dump();
    errs()<<ANSI_COLOR_RESET;
    #endif
    return ii;

}

/*
 * just copy whatever available in given pointer
 */
Value* llmpx::handlePtrToInt(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    PtrToIntInst* ptr2int_inst = dyn_cast<PtrToIntInst>(ii);

    Value* ptr = ptr2int_inst->getPointerOperand();
    /*
     * FIXME: possible that this can be replaced, if it is a function pointer
     */
    Value* new_ptr = process_each_instruction(ptr);
#if 1
    new_ptr = ptr;
#else
    if (new_ptr!=ptr)
    {
        ptr->dump();
        new_ptr->dump();
        llvm_unreachable("FIXME! ptr was changed in process_each_instruction");
    }
#endif
    blist->push_back(get_bound(ptr, ptr2int_inst));
    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(get_key(ptr, ptr2int_inst));
    }
    return ii;
}

Value* llmpx::handleBinaryOperator(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    //FIXME: is it true that base always appears at operand 0?
    Instruction* inst = dyn_cast<Instruction>(ii);
    Value* operand0 = inst->getOperand(0);
    Value* operand1 = inst->getOperand(1);
#if DEBUG_HANDLE_BINARY_OP
    errs()<<"Handle Binary Operator:";
    ii->dump();
#endif
    Instruction* inst0
        = dyn_cast<Instruction>(process_each_instruction(operand0));
    Instruction* inst1
        = dyn_cast<Instruction>(process_each_instruction(operand1));
    Value* ptr = inst;
//FIXME!
    if ((inst0==NULL) && (inst1==NULL))
    {
        ii->dump();
        llvm_unreachable("unable to find bound!");
    }
    if(inst0!=NULL)
    {
        ptr = inst0;
    }else if(inst1!=NULL)
    {
        ptr = inst1;
    }else if ((inst0!=NULL) && (inst1!=NULL))
    {
#if 0
        ii->dump();
        inst0->dump();
        inst1->dump();
        dump_dbgstk();
        llvm_unreachable("Two bound?!\n");
#endif
    }
    blist->push_back(get_bound(ptr, GetNextInstruction(inst)));
    if (llmpx_enable_temporal_safety)
    {
        klist->push_back(get_key(ptr, GetNextInstruction(inst)));
    }

    return ii;
}

//TOOD: add key population logic for select
Value* llmpx::handleSelect(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }

    Instruction* i = dyn_cast<Instruction>(ii);

    SelectInst* sel_inst = dyn_cast<SelectInst>(i);
    #if DEBUG_HANDLE_SELECT
    errs()<<ANSI_COLOR_CYAN
        <<"Handle selectInst\n"
        <<ANSI_COLOR_RESET;
    ii->dump();
    #endif

    //if it does not return bound then no need to add bound
    if (!isa<PointerType>(sel_inst->getType()))
    {
        return sel_inst;
    }

    //examine the instructiont to see if we need to replace the instruction
    Value* cond = sel_inst->getCondition();
    Value* t_val = sel_inst->getTrueValue();
    Value* f_val = sel_inst->getFalseValue();
    Value* proc_t_val = process_each_instruction(t_val);
    Value* proc_f_val = process_each_instruction(f_val);

    if ((t_val!=proc_t_val)||(f_val!=proc_f_val))
    {
        #if DEBUG_HANDLE_SELECT>2
        errs()<<"We have replaced operand:\n";
        errs()<<"  t:";
        if(proc_t_val!=NULL)
        {
            proc_t_val->print(errs());
        }else
        {
            errs()<<"WTF NULL?";
        }
        errs()<<"\n";
        errs()<<"  f:";
        if(proc_f_val!=NULL)
        {
            proc_f_val->print(errs());
        }else
        {
            errs()<<"WTF NULL?";
        }
        errs()<<"\n";
        #endif
        IRBuilder<> builder(sel_inst);
        /*
         * need to cast type if type is ever changed during
         * instruction processing
         */
        Type* new_t_type = proc_t_val->getType();
        if (new_t_type != t_val->getType())
        {
            proc_t_val = builder.CreateBitCast(proc_t_val,
                                        t_val->getType(),
                                        t_val->getName()+"_bc_true");
        }
        Type* new_f_type = proc_f_val->getType();
        if (new_f_type != f_val->getType())
        {
            proc_f_val = builder.CreateBitCast(proc_f_val,
                                        f_val->getType(),
                                        f_val->getName()+"_bc_false");
        }
        Value* newbitcast 
            = builder.CreateSelect(cond,
                        proc_t_val, proc_f_val,
                        sel_inst->getName()+"_swbnd");
        Value* newresult 
            = builder.CreateBitCast(newbitcast, sel_inst->getType(),
                        sel_inst->getName()+"_scast");
        //need to replace this selectInst
        add_instruction_to_bcl(newbitcast);
        blist = bound_checklist[newbitcast];
        //push back myself to avoid re-processing?
        blist->push_back(newbitcast);
        blist->push_back(newresult);
        
        klist = key_checklist[newbitcast];
        if (llmpx_enable_temporal_safety)
        {
            klist->push_back(NULL);
        }
        bound_checklist[sel_inst] = &delete_ii;
        sel_inst->replaceAllUsesWith(newresult);
        sel_inst->removeFromParent();
        return newresult;
    }
    Value* t_bnd = get_bound(t_val, i);
    if (!t_bnd)
    {
        t_bnd = get_infinite_bound(sel_inst);
    }

    Value* f_bnd = get_bound(f_val, i);
    if (!f_bnd)
    {
        f_bnd = get_infinite_bound(sel_inst);
    }
    
#if 0
    /*
     * select_cc can not emit cmov for x86bnd type
     * we need to transform it into if else condition
     */

    Type* bndty = Type::getX86_BNDTy(*ctx);

    TerminatorInst* new_bb_terminator
        = SplitBlockAndInsertIfThen(cond, GetNextInstruction(sel_inst),false);
    BasicBlock* new_bb = new_bb_terminator->getParent();
    BasicBlock* old_bb = new_bb->getSinglePredecessor();
    BasicBlock* phi_bb = new_bb->getSingleSuccessor();

    /*
     * insert phi node
     */
    IRBuilder<> builder3(phi_bb->getFirstNonPHI());
    
    PHINode* bnd = builder3.CreatePHI(bndty, 2, "select_cc_bnd");
    bnd->addIncoming(t_bnd, new_bb);
    bnd->addIncoming(f_bnd, old_bb);

    blist->push_back(bnd);

    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }

#else
    /*
     * we support bnd select
     * see llvm mpx commit ee1048f88fe380a8be710a57722d2bae78cb52d1
     */
    IRBuilder<> builder(sel_inst);
    Value* sel_bnd_result = builder.CreateSelect(cond, t_bnd, f_bnd);
    blist->push_back(sel_bnd_result);
    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }

#endif
    return sel_inst;
}

Value* llmpx::handleRet(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    Instruction* i = dyn_cast<Instruction>(ii);

    ReturnInst* ret_inst = dyn_cast<ReturnInst>(i);
    Value* ret_val = ret_inst->getReturnValue();

    Function* func = ret_inst->getParent()->getParent();

    #if (DEBUG_HANDLE_RET)
    errs()<<"handleRet: \n";
    if(ret_val)
    {
        errs()<<"  ret_val:";
        ret_val->dump();
        errs()<<"  type:";
        ret_val->getType()->dump();
    }else
    {
        errs()<<"NULL\n";
    }
    errs()<<"Function :"<<func->getName()<<"\n";
    errs()<<" require return type:";
    func->getReturnType()->dump();
    #endif

    /*
     * no need to handle :
     * ret void
     * ret non-pointer value
     */
    if ((!ret_val) || 
        ((!isa<PointerType>(ret_val->getType()))
        && (!ret_val->getType()->isStructTy())))
    {
        return ii;
    }
    Function* new_func = ret_inst->getParent()->getParent();
    //FIXME!
    Function* old_func 
        = dyn_cast<Function>(find_nontransformed_function(new_func));
    if ((tr_flist_ret.find(old_func)==tr_flist_ret.end()) ||
        ((tr_flist_ret.find(old_func)!=tr_flist_ret.end())
        && (tr_flist_ret[old_func]<=0)))
    {
        //no need to instrument this return
        return ii;
    }
    #if (DEBUG_HANDLE_RET)
    errs()<<" have to return bound\n";
    #endif
    ret_val = process_each_instruction(ret_val);
    //collect inserted instructions
    std::list<Value*> iS;
    std::list<Value*> kS;
    Value* new_ret_val;
    //prepare aggregate value for return
    Type* ret_type = new_func->getReturnType();
    Value* agg0 = UndefValue::get(ret_type);
    Instruction* i0 
        = InsertValueInst::Create(agg0, ret_val, 0, "mrv", ret_inst);
    iS.push_back(i0);
    kS.push_back(i0);
    new_ret_val = i0;
    
    ripoff_tail_call(ret_val);
    /*
     * first search whether the pointer has a bound in bound list,
     *   - if found, return that bound,
     *   - otherwise load the bound from bound table
     *     `  this is a bit weired though..
     */
    if (isa<PointerType>(ret_val->getType()))
    {
        //simple cast, append one bound
        Value* bound = get_bound(ret_val, i);
        if (!bound)
        {
            bound = get_infinite_bound(i0);
            //incorrect target blist
            iS.push_back(bound);
        }
        Instruction* i1 
            = InsertValueInst::Create(new_ret_val, bound,
                            1, "mrv", ret_inst);
        iS.push_back(i1);
        new_ret_val = i1;

        if (llmpx_enable_temporal_safety)
        {
            Value* key = get_key(ret_val, i);
            if (!key)
            {
                key = get_anyvalid_key(i0);
                kS.push_back(key);
            }
            Instruction *i2
                = InsertValueInst::Create(new_ret_val, key,
                    2, "mrv", ret_inst);
            kS.push_back(i2);
            new_ret_val = i2;
        }
    }else if (ret_val->getType()->isStructTy())
    {
        //find out bound of pointers in given struct
        bool ret_val_is_const = isa<Constant>(ret_val);
        #if DEBUG_HANDLE_RET
        errs()<<"Returned struct type\n";
        if(ret_val_is_const)
            errs()<<"Constant:";
        ret_val->dump();
        #endif
        /*
         * iterate through every element in aggregated
         * value, if the value is pointer type,
         * create and insert bound to new_ret_val
         * NOTE: all bnd for pointer in aggregated type are
         * now loaded from bound table directly,
         * pointers in subtype should all be stored in 
         * bound table
         */
        /*
         * NOTE: we used to have problem here, we assume whatever pointer
         * returned should have its bound readily stored in BT, which is not
         * always correct. Because there may be cases like this:
         * void* foo()
         * {
         *      return malloc(100);
         * }
         * could be optimized not using memory to store intermediate result
         * and the bound is not stored in the BT either.
         * samething for key load
         */

        Type* ret_val_type = ret_val->getType();
        //collect ptr in struct in ptrlist
        std::list<Value*> ptrlist;
        //collect bound belongs to the ptr
        std::list<Value*> ptrbndlist;
        //collect key belongs to the ptr
        std::list<Value*> ptrkeylist;
        for (int j=0;j<ret_val_type->getStructNumElements();j++)
        {
            Type* etype = ret_type->getStructElementType(j);
            if (etype->isPointerTy())
            {
                //should return bound for this one
                if (ret_val_is_const)
                {
again:
                    Value *Idx[2];
                    Idx[0] = Constant::getNullValue(Type::getInt32Ty(*ctx));
                    Idx[1] = ConstantInt::get(Type::getInt32Ty(*ctx), j);

                    Value* the_ptr
                        = GetElementPtrInst::Create(etype,
                                                    ret_val,
                                                    Idx,
                                                    "", ret_inst);
                    ptrlist.push_back(the_ptr);
                    //load constant bnd from BT
                    std::list<Value*> ilist = 
                        insert_bound_load(ret_inst, the_ptr, ret_val);
                    iS.splice(iS.end(), ilist);
                    Value* the_bnd = iS.back();
                    ptrbndlist.push_back(the_bnd);
                    if (llmpx_enable_temporal_safety)
                    {
                        //load constant key 
                        std::list<Value*> ilist = 
                           insert_key_load(ret_inst, the_ptr);
                        kS.splice(kS.end(), ilist);
                        Value* the_key = kS.back();
                        ptrkeylist.push_back(the_key);
                    }
                }else
                {
                    /*
                     * find out how it is constructed
                     * only handle insertvalue instruction
                     * reject to again if failed to find valid instruction
                     */
#if DEBUG_HANDLE_RET
                    errs()<<"idx:"<<j<<" is PointerType";
#endif
                    bool resolved = false;
                    Value* ret_chain = ret_val;
                    while(!resolved)
                    {
                        if (!isa<Instruction>(ret_chain))
                        {
                            goto again;
                        }
                        Instruction* chain_inst
                            = dyn_cast<Instruction>(ret_chain);
                        if (!isa<InsertValueInst>(chain_inst))
                        {
                            goto again;
                        }
                        InsertValueInst* chain_iv_inst
                            = dyn_cast<InsertValueInst>(chain_inst);
                        if (chain_iv_inst->getInsertedValueOperandIndex() == j)
                        {
                            resolved = true;
                            Value* the_ptr
                                = chain_iv_inst->getInsertedValueOperand();
                            ptrlist.push_back(the_ptr);
                            Value* the_bnd = get_bound(the_ptr, ret_inst);
                            ptrbndlist.push_back(the_bnd);
                            if (llmpx_enable_temporal_safety)
                            {
                                Value* the_key = get_key(the_ptr, ret_inst);
                                ptrkeylist.push_back(the_key);
                            }
                        }else
                        {
                            ret_chain = chain_iv_inst->getAggregateOperand();
                        }
                    }
                }
            }
        }
        
        /*
         * return bound and key
         */
        int meta_idx = 1;//bound start from 1
        for(std::list<Value*>::iterator it = ptrbndlist.begin();
                it != ptrbndlist.end(); ++it)
        {
            Value* the_bnd = *it;
            Instruction* i1
                = InsertValueInst::Create(new_ret_val, the_bnd,
                        meta_idx, "mrv", ret_inst);
            iS.push_back(i1);
            new_ret_val = i1;
            meta_idx++;
        }
        if (llmpx_enable_temporal_safety)
        {
            for(std::list<Value*>::iterator it=ptrkeylist.begin();
                    it != ptrkeylist.end(); ++it)
            {
                Value* the_key = *it;
                Instruction* i1
                    = InsertValueInst::Create(new_ret_val, the_key,
                            meta_idx, "mrv", ret_inst);
                kS.push_back(i1);
                new_ret_val = i1;
                meta_idx++;
            }
        }
    }else
    {
        llvm_unreachable(" return type is neither pointer nor aggregate");
    }
    //create new return instruction
    ReturnInst* new_ret_inst 
        = ReturnInst::Create(*ctx, new_ret_val, ret_inst);
    #if (DEBUG_HANDLE_RET)
    errs()<<"Created return inst:";
    new_ret_inst->dump();
    errs()<<"return type:";
    new_ret_val->getType()->dump();
    #endif

    //double check
    assert((func->getReturnType()==new_ret_val->getType())&&
            "wrong return type!");

    add_instruction_to_bcl(new_ret_inst);
    blist = bound_checklist[new_ret_inst];
    //push back myself to avoid re-processing?
    blist->push_back(new_ret_inst);
    blist->splice(blist->end(), iS);
    bound_checklist[ret_inst] = &delete_ii;

    if (llmpx_enable_temporal_safety)
    {
        klist = key_checklist[new_ret_inst];
        klist->push_back(new_ret_inst);
        klist->splice(klist->end(), kS);
        key_checklist[ret_inst] = &delete_ii;
    }

    ret_inst->removeFromParent();

    return new_ret_inst;
}

/*
 * NOTE: need to handle ugly pointer store
 */
Value* llmpx::handleStore(Value* ii)
{
    std::list<Value*>* blist = bound_checklist[ii];
    std::list<Value*>* klist = key_checklist[ii];

    if(llmpx_enable_temporal_safety)
    {
        klist->push_back(NULL);
    }

    Instruction* i = dyn_cast<Instruction>(ii);

    #if DEBUG_HANDLE_STORE>3
    errs()<<" Instruction::Store - \n";
    i->print(errs());
    errs()<<"\n";
    #endif
    StoreInst* store_inst = dyn_cast<StoreInst>(i);
    //errs()<<"   Insert bound check\n";
    //insert check
    Value* ptr_operand = store_inst->getPointerOperand();
    /*
     * don't check accesses from different address space?
     */
    if (ptr_operand)
    {
        Type *PtrTy = cast<PointerType>(ptr_operand->getType()->getScalarType());
        if (PtrTy->getPointerAddressSpace()!=0)
        {
            llvm_unreachable("Found ptr belongs to another address space");
        }
    }
    //add check if needed
    std::list<Value*> ptrilist;
    if (!is_safe_access_cache(store_inst))
    {
        ptrilist = insert_check(i, ptr_operand, false);
    }else
    {
        ElimSafeAccess++;
#if DEBUG_HANDLE_STORE
        errs()<<"Safe Access found in handleStore\n";
#endif
    }
    //just in case it might be changed
    ptr_operand = store_inst->getPointerOperand();
    
    /* 
     * if the stored value is pointer type
     * wee need to store its bound as well
     *
     * we need to look at both value and pointer operand, if any of them shows
     * sign to be used as a pointer, store the bound.
     *
     * some llvm optimization pass is doing this ugly thing,
     * it cast pointer to i64... for store,
     * which mess up the bound propogation,
     * so that we need also to check the destination type,
     * if it is ** then we also need to propogate the type
     */
    Value* val = store_inst->getValueOperand();
    Type* val_type = val->getType();
    Value* stptr = store_inst->getPointerOperand();
    Type* stptr_type = stptr->getType()->getContainedType(0);
    /*
     * try to figure out the real type of the PointerOperand
     */
    Type* act_stptr_type = find_actual_type(stptr, true);
    Type* act_val_type = find_actual_type(val, false);

    bool store_non_ptr_as_ptr = false;
    bool store_ptr_as_non_ptr = false;
    if ((!val_type->isPointerTy())
        && (stptr_type->isPointerTy() || act_stptr_type->isPointerTy()))
    {
        store_non_ptr_as_ptr = true;
    }
    if ((val_type->isPointerTy() || act_val_type->isPointerTy())
        && (!stptr_type->isPointerTy()))
    {
        store_ptr_as_non_ptr = true;
    }
#if DEBUG_HANDLE_STORE
    if (store_non_ptr_as_ptr)
    {
        errs()<<"Store non pointer as pointer\n";
    }
    if (store_ptr_as_non_ptr)
    {
        errs()<<"Store pointer as non-pointer\n";
    }
    store_inst->dump();
    stptr->dump();
#endif
    //just in case non of them are pointer type
    //and obviously case, FP etc.
    if (((!val_type->isPointerTy()) && (!act_val_type->isPointerTy())
        && (!stptr_type->isPointerTy()) && (!act_stptr_type->isPointerTy()))
        || val_type->isFloatingPointTy() || val_type->isFPOrFPVectorTy()
        || val_type->isVectorTy())
    {
        blist->splice(blist->end(), ptrilist);
        return ii;
    }
    /*
     * value might change because we replaced %call
     * function pointer need to be handled
     */
    Type* containedType = val->getType();
    if(val->getType()->isPointerTy())
        containedType = val->getType()->getContainedType(0);
    if(containedType->isFunctionTy())
    {
        //handle function pointer
        //it is possible that it want to store an function pointer 
        //to somewhere and event the function pointer(val)
        //could be stored in value object 
        //assert( isa<Function>(val)&&"FUCK! not a function?" );
        if (!isa<Function>(val))
        {
            return ii;
        }
        Function* func = dyn_cast<Function>(val);
        //is this function need to be replaced?
        if (!is_function_orig(func))
        {
            //what??
            //errs()<<" no function transformation found, maybe external\n";
            //errs()<<func->getName()<<"\n";
            blist->splice(blist->end(), ptrilist);
            return ii;
        }
        Function *nfunc = tr_flist[func];

        //bitcast nfunc type to old type
        IRBuilder<> builder(store_inst);
        Value* nfptr  = builder.CreateBitCast(nfunc, val->getType(),"");
        StoreInst* nstor = builder.CreateStore(nfptr, ptr_operand);
        add_instruction_to_bcl(nstor);
        blist = bound_checklist[nstor];
        blist->splice(blist->end(), ptrilist);
        blist->push_back(nfptr);
        //insert place holder for bound
        blist->push_back(NULL);
        bound_checklist[i] = &delete_ii;
        if(llmpx_enable_temporal_safety)
        {
            klist = key_checklist[nstor];
            klist->push_back(NULL);
            key_checklist[i] = &delete_ii;
        }
        i->removeFromParent();
        return nstor;
    }else
    {
        blist->splice(blist->end(), ptrilist);
        Value* bnd = NULL;
        if (store_non_ptr_as_ptr || store_ptr_as_non_ptr)
        {
            //in any of these case, load inf bound
        }else
        {
            //normal pointer type
            val = process_each_instruction(val);
            bnd = get_bound(val, i);
        }

        //get the bound of val 
        if (!bnd)
        {
            bnd = get_infinite_bound(store_inst);
            blist->push_back(bnd);
        }

        //store the bound
        std::list<Value*> ins_blist
            = insert_bound_store(store_inst, 
                            store_inst->getPointerOperand(),
                            store_inst->getValueOperand(),
                            bnd);

        if (llmpx_enable_temporal_safety)
        {
            Value* key = get_key(val, i);
            if (!key)
            {
                key = get_anyvalid_key(store_inst);
                klist->push_back(key);
            }
            insert_key_store(store_inst,
                        store_inst->getPointerOperand(), key);
        }
    }
    return ii;
}

////////////////////////////////////////////////////////////////////////
/*
 * process value
 * return the replacement def
 * --------------------------
 *  this not only deal with instructions but also value
 */
Value* llmpx::process_each_instruction(Value *ii)
{
    assert(ii&&"process null value?");
    dbgstk.push(ii);
    if (isa<Function>(ii))
    {
        //whenever a function is passed in for process,
        //we return its substitute function
        #if DEBUG_PROCESS_INST>10
        errs()<<ANSI_COLOR_RED
            <<"process each instruction need to deal with function\n"
            <<ANSI_COLOR_RESET;
        ii->dump();
        #endif
        if (is_function_orig(ii))
        {
            Function* new_func = find_transformed_function(ii);
            dbgstk.pop();
            return new_func;
        }
        dbgstk.pop();
        return ii;
    }
    std::list<Value*>* blist = NULL;
#if 1
    if (bound_checklist.find(ii)==bound_checklist.end())
    {
        #if DEBUG_PROCESS_INST
        errs()<<"Process each instruction: not found in bound_checklist\n";
        ii->dump();
        #endif
        //not found in the pre-gathered value/instruction
        //proceed anyway...
        if (isa<Instruction>(ii))
        {
            #if DEBUG_PROCESS_INST
            errs()<<"is instruction, process anyway\n";
            #endif
            add_instruction_to_bcl(ii);
            blist = bound_checklist[ii];
            goto process_instruction;
        }
        if (isa<ConstantExpr>(ii))
        {
#if 0
            errs()<<"is constant expr, process anyway\n";
            Instruction* iii = dyn_cast<ConstantExpr>(ii)->getAsInstruction();
            Value* iiii = process_each_instruction(iii);
            delete iii;
            return iiii;
#else
            #if DEBUG_PROCESS_INST
            errs()<<"is constant expr, do nothing\n";
            #endif
            dbgstk.pop();
            return ii;
#endif
        }
        if (isa<Constant>(ii))
        {
            Constant* iii = dyn_cast<Constant>(ii);
            assert(iii && "unhandled constant");
            dbgstk.pop();
            return ii;
        }
        if (ii->getType()->isPointerTy())
        {
#if 0
            llvm_unreachable("unhandled non-null Pointer");
#else
            #if DEBUG_PROCESS_INST
            errs()<<"other non-null Pointer\n";
            #endif
            dbgstk.pop();
            return ii;
#endif
        }
        if (isa<Argument>(ii))
        {
            //non pointer argument who also want to have a bound?
            llvm_unreachable("non pointer argument who want to have a bound?");
            dbgstk.pop();
            return ii;
        }
        errs()<<ANSI_COLOR_RED<<"  ";
        ii->dump();
        errs()<<ANSI_COLOR_RESET;
        llvm_unreachable("Can not process value");
    }else
    {
        blist = bound_checklist[ii];
    }
#endif
    /*
     * for each blist
     * - first insert check for its pointer operand
     * - then insert bound for the returned value at the end of
     *   the blist, so that the end of blist is always the bound
     *   of returned value
     */
    if (blist==NULL)
    {
        //this value does not have bound,
        //means that this is an symbol outside of the function
        //need to insert bndldx
#if 0
        llvm_unreachable(" - i this value does not have bound?");
        return NULL;
#else
        //errs()<<"- i this value does not have bound?\n";
        dbgstk.pop();
        return ii;
#endif
    }else if(blist==&delete_ii)
    {

        #if DEBUG_PROCESS_INST>3
        errs()<<"deleted instruction: ";
        ii->print(errs());
        errs()<<"\n";
        #endif
        #if 0
        ins->eraseFromParent();
        llvm_unreachable( "instruction already marked to be deleted\n"
                        "and all uses should have been replaced and erased!\n");
        #endif
        dbgstk.pop();
        return NULL;
    }
    if (blist->size())
    {
        //this instruction has been processed
        dbgstk.pop();
        return ii;
    }

process_instruction:
#if 0
    errs()<<"+process: ";
    ii->print(errs());
    errs()<<"\n";
#endif
    Instruction* i = dyn_cast<Instruction>(ii);
    switch(i->getOpcode())
    {
        default:
            errs()<<ANSI_COLOR_RED;
            i->dump();
            errs()<<ANSI_COLOR_RESET;
            if (i->getType())
            {
                if(i->getType()->isPointerTy())
                {
                    errs()<<" PointerType:";
                }else
                {
                    errs()<<" non-PointerType:";
                }
                i->getType()->dump();
            }
            errs()<<" OPCODE: "<<i->getOpcode()<<"\n";
            llvm_unreachable("WTF, this is impossible");
            dbgstk.pop();
            return NULL;
        /*
         * instructions which has result, need to check
         * its ptr operand first, then try to associate the
         * bound with the returned value(if it is PointerTy)
         */
        case Instruction::Alloca:
        {
            ii = handleAlloca(ii);
            dbgstk.pop();
            return ii;
        }
        case BitCastInst::BitCast:
        {
            ii = handleBitCast(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::Call:
        {
            ii = handleCall(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::Invoke:
        {
            ii = handleInvoke(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::InsertElement:
        {
            ii = handleInsertElement(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::ExtractElement:
        {
            ii = handleExtractElement(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::ExtractValue:
        {
            ii = handleExtractValue(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::InsertValue:
        {
            ii = handleInsertValue(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::GetElementPtr:
        {
            ii = handleGetElementPtr(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::IntToPtr:
        {
            ii = handleIntToPtr(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::Load:
        {   
            ii = handleLoad(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::PHI:
        {
            ii = handlePHINode(ii);
            dbgstk.pop();
            return ii;

        }
        case Instruction::Select:
        {
            ii = handleSelect(ii);
            dbgstk.pop();
            return ii;
        }
        /*
         * instruction which has no result
         * only need to check the bound of pointer operand
         */
        case Instruction::Ret:
        {
            ii = handleRet(ii);
            dbgstk.pop();
            return ii;
        }
        case Instruction::Store:
        {
            ii = handleStore(ii);
            dbgstk.pop();
            return ii;
        }
        //for pointer manipulation by using integer math
        case Instruction::PtrToInt:
        {
            ii = handlePtrToInt(ii);
            dbgstk.pop();
            return ii;
        }
        case BinaryOperator::Add:
        case BinaryOperator::Sub:
        case BinaryOperator::Mul:
        #if 0
        {
            ii = handleBinaryOperator(ii);
            dbgstk.pop();
            return ii;
        }
        #endif
        /*
         * reject all floating point operator
         *            div/mul/rem operator
         *            trunc/sext/zext operator
         *            logical operator
         */
        case BinaryOperator::FAdd:
        case BinaryOperator::FSub:
        case BinaryOperator::FMul:
        case BinaryOperator::UDiv:
        case BinaryOperator::SDiv:
        case BinaryOperator::FDiv:
        case BinaryOperator::URem:
        case BinaryOperator::SRem:
        case BinaryOperator::FRem:
        case Instruction::FPToUI:
        case Instruction::FPToSI:
        case Instruction::UIToFP:
        case Instruction::SIToFP:
        case Instruction::FPTrunc:
        case Instruction::FPExt:
        case Instruction::Trunc:
        case Instruction::ZExt:
        case Instruction::SExt:
        case Instruction::Shl:
        case Instruction::LShr:
        case Instruction::AShr:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
        {
            break;
        }

    }
    return ii;
}

/*
 * process bound check list
 */
void llmpx::process_bound_checklist()
{
    for (auto I: bound_checklist)
    {
        process_each_instruction(I.first);
    }
    //delete all removed instructions
    for (auto I: bound_checklist)
    {
        if (I.second==&delete_ii)
        {
            //check its use
            Value* v = I.first;
            if (!v->use_empty())
            {
                errs()<<"deleting ";
                I.first->dump();
            }
            for (auto ui = v->use_begin(), ue = v->use_end();
                ui!=ue;)
            {
                Use& use = *ui++;
                Instruction* uv = cast<Instruction>(use.getUser());
                errs()<<ANSI_COLOR_RED<<"USE:"<<ANSI_COLOR_RESET;
                uv->dump();
            }
            assert( v->use_empty() && "Value still have use(s) when deleting!" );
            /*
             * dont touch, if it is not orphan
             */
            if(isa<Instruction>(v))
            {
                if(!dyn_cast<Instruction>(v)->getParent())
                {
                    delete I.first;
                }
            }else
            {
                delete I.first;
            }
        }
    }
}

/*
 * handle bound make/check and propogation for each instruction
 * and store the result in bound_checklist
 */
void llmpx::gen_bound_checklist(Instruction *I)
{
    unsigned opcode = I->getOpcode();
    switch(opcode)
    {
        default:
                return;
        case Instruction::GetElementPtr:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::Store:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::Load:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::PHI:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::PtrToInt:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case BitCastInst::BitCast:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::Alloca:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::Call:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::Invoke:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::Select:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::IntToPtr:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::Ret:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::ExtractElement:
        {
            add_instruction_to_bcl(I);
            break;
        }
        case Instruction::ExtractValue:
        {
            add_instruction_to_bcl(I);
            break;
        }
    }
}

bool llmpx::is_safe_access_cache(Value* i)
{
#if SAFE_ACCESS_ELIMINATION
    return (safe_access_list.find(i)!=safe_access_list.end());
#else
    return false;
#endif
}

bool llmpx::is_safe_access(Value* addr, uint64_t type_size)
{
#if SAFE_ACCESS_ELIMINATION
    uint64_t size;
    uint64_t offset;
    bool result;
    #if DEBUG_IS_SAFE_ACCESS
        std::string reason;
        errs()<<ANSI_COLOR_YELLOW
                <<"? is_safe_access ("<<(type_size/8)<<"):"
                <<ANSI_COLOR_RESET;
        addr->dump();
    #endif

    if(isa<GlobalVariable>(addr))
    {
        GlobalVariable* gv = dyn_cast<GlobalVariable>(addr);
        if(gv->getLinkage()==GlobalValue::ExternalLinkage)
        {
            goto fallthrough;
        }
        if (!gv->hasInitializer())
        {
            //we have no idea???
            goto fallthrough;
        }
        Constant* initializer = gv->getInitializer();
        Type* itype = initializer->getType();
        unsigned allocated_size = module->getDataLayout()
                        .getTypeAllocSize(itype);
        size = allocated_size;
        offset = 0;
    }else
    {
fallthrough:
        const TargetLibraryInfo * TLI = 
            &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
        const DataLayout &dl = cfunc->getParent()->getDataLayout();
        obj_size_vis = new ObjectSizeOffsetVisitor(dl, TLI, *ctx, true);

        SizeOffsetType size_offset = obj_size_vis->compute(addr);
        if (!obj_size_vis->bothKnown(size_offset))
        {
#if DEBUG_IS_SAFE_ACCESS
            if (obj_size_vis->knownSize(size_offset))
            {
                reason += "size: " + size_offset.first.getZExtValue();
            }else
            {
                reason += "size: NA ";
            }
            reason += " ";
            if (obj_size_vis->knownOffset(size_offset))
            {
                reason += "Offset: " + size_offset.second.getSExtValue();
            }else
            {
                reason += "offset: NA";
            }
#endif
            result = false;
            goto dead_or_alive;
        }
        size = size_offset.first.getZExtValue();
        offset = size_offset.second.getSExtValue();
    }
    result = (offset >= 0) && (size >= uint64_t(offset)) &&
        ((size - uint64_t(offset)) >= (type_size / 8));

dead_or_alive:
#if DEBUG_IS_SAFE_ACCESS
    if (result)
    {
        errs()<<ANSI_COLOR_GREEN
            <<"SAFE"
            <<ANSI_COLOR_RESET<<"\n";
    }else
    {
        errs()<<ANSI_COLOR_RED
            <<"NOT SAFE, "
            <<ANSI_COLOR_RESET
            <<"Reason:"
            <<reason<<"\n";
    }
#endif
    return result;
#else//SAFE_ACCESS_ELIMINATION
    return false;
#endif
}

/*
 * find actual type, for ugly load/store
 * This doesn't really work, in order to implement it correct,
 * we need more information from frontend...
 * Track down the use chain for evidence of pointer type
 * val - the value to track down
 * begin_from_ptr - don't count the first ptr type (val is ptr)
 *                - just ignore the very first occurance of pointer type
 */
Type* llmpx::find_actual_type(Value* val, bool begin_from_ptr)
{
    return find_actual_type(val, begin_from_ptr, 0);
}

Type* llmpx::find_actual_type(Value* val, bool begin_from_ptr, int depth)
{
#if DEBUG_FIND_ACTUAL_TYPE
    errs()<<ANSI_COLOR_YELLOW
          <<"Find actual type for ("<<begin_from_ptr<<"):"
          <<ANSI_COLOR_RESET;
    val->dump();
#endif
    Type* type = NULL;
    bool resolved = false;

    Value* orig_val = val;
    bool ignore_first_ptr = false;
    if(val->getType()->isPointerTy() && begin_from_ptr)
    {
        ignore_first_ptr = true;
    }
    if (ignore_first_ptr)
    {
        if (at_cache0[val]!=NULL)
        {
            resolved = true;
            type = at_cache0[val];
            goto  out;
        }
    }else
    {
        if (at_cache1[val]!=NULL)
        {
            resolved = true;
            type = at_cache1[val];
            goto  out;
        }
    }
    //simple case:** type return directly
    if (val->getType()->isPointerTy())
    {
        Type* type = val->getType()->getContainedType(0);
        if(type->isPointerTy())
        {
            return type;
        }
    }
    if (!begin_from_ptr)
    {
        if (val->getType()->isPointerTy())
        {
            return val->getType();
        }
    }

    /*
     * prevent this from going too deep
     */
    if (depth>_LLMPX_MAX_FIND_DEPTH_)
    {
        type = val->getType();
        if (begin_from_ptr)
        {
            if(type->isPointerTy())
            {
                type = type->getContainedType(0);
            }
        }
        return type;
    }
    while (!resolved)
    {
#if DEBUG_FIND_ACTUAL_TYPE
    val->dump();
#endif
        /*
         * reject all FP Type
         *            MMX Type
         *            BND Type
         *            Function Type
         *            Vector Type
         */
        type = val->getType();
        if (type->isFPOrFPVectorTy()
            ||type->isX86_MMXTy()
            ||type->isX86_BNDTy()
            ||type->isFunctionTy()
            ||type->isVectorTy())
        {
            resolved = true;
            break;
        }
        if(val->getType()->isPointerTy())
        {
            if ((!ignore_first_ptr)||(val!=orig_val))
            {
                type = val->getType();
                resolved = true;
                break;
            }
        }
        if(isa<ConstantExpr>(val))
        {
            ConstantExpr* cexpr = dyn_cast<ConstantExpr>(val);
            type = cexpr->getOperand(0)->getType();
            resolved = true;
            break;
        }else if(isa<GlobalVariable>(val) || isa<GlobalValue>(val))
        {
            type = val->getType();
            resolved = true;
            if((ignore_first_ptr) && (val==orig_val))
            {
                if(type->isPointerTy())
                    type = type->getContainedType(0);
            }
            break;
        }else if(isa<Constant>(val))
        {
            type = val->getType();
            resolved = true;
            if((ignore_first_ptr) && (val==orig_val))
            {
                if(type->isPointerTy())
                    type = type->getContainedType(0);
            }
            break;
        }

        if(!isa<Instruction>(val))
        {
            //reject if it is not Instruction
#if DEBUG_FIND_ACTUAL_TYPE
            errs()<<ANSI_COLOR_RED
                    <<"Val is not one of them:"
                    <<" Constant*, GlobalV* nor Instruction."
                    <<" Possible function argument.\n"
                    <<ANSI_COLOR_RESET;
            val->dump();
#endif
            type = val->getType();
            resolved = true;
            break;
        }
        Instruction* inst = dyn_cast<Instruction>(val);
        switch(inst->getOpcode())
        {
            case(Instruction::BitCast):
            {
                BitCastInst* bci = dyn_cast<BitCastInst>(inst);
                val = bci->getOperand(0);
                type = val->getType();
                if((ignore_first_ptr) && (val==orig_val))
                {
                    if(type->isPointerTy())
                        type = type->getContainedType(0);
                }
                break;
            }
            case(Instruction::Alloca):
            {
                AllocaInst* alloc_inst = dyn_cast<AllocaInst>(inst);
                type = alloc_inst->getType();
                resolved = true;
                if((ignore_first_ptr) && (val==orig_val))
                {
                    if(type->isPointerTy())
                        type = type->getContainedType(0);
                }
                break;
            }
            case(Instruction::Load):
            {
                LoadInst* ldinst = dyn_cast<LoadInst>(inst);
                if((ignore_first_ptr) && (val==orig_val))
                {
                    type = ldinst->getType();
                    if (type->isPointerTy())
                    {
                        //resolved
                        type = type->getContainedType(0);
                        if(type->isPointerTy())
                        {
                            break;
                        }
                    }
                }
                val = ldinst->getPointerOperand();
                type = find_actual_type(val, true, depth+1);
                if(type->isPointerTy())
                    type = type->getContainedType(0);
                resolved = true;
                break;
            }
            case(Instruction::Store):
            {
                llvm_unreachable("No way this instruction can be used!");
                #if 0
                StoreInst* stinst = dyn_cast<StoreInst>(inst);
                type = stinst->getValueOperand()->getType();
                if (type->isPointerTy())
                {
                    //resolved
                    break;
                }
                val = stinst->getPointerOperand();
                type = find_actual_type(val, false, depth+1);
                #endif
                break;
            }
            case(Instruction::IntToPtr):
            {
                //should always return pointer
                IntToPtrInst* int2ptr = dyn_cast<IntToPtrInst>(inst);
                type = int2ptr->getType();
                resolved = true;
                break;
            }
            case(Instruction::PtrToInt):
            {
                //should always cast from pointer
                PtrToIntInst* ptr2int = dyn_cast<PtrToIntInst>(inst);
                val = ptr2int->getPointerOperand();
                type = val->getType();
                resolved = true;
                break;
            }
            case(Instruction::GetElementPtr):
            {
                //should always return pointer
                //if the contained type if wanted type
                GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(inst);
                type = gep->getType()->getContainedType(0);
                resolved = true;
                break;
            }
            case(Instruction::PHI):
            {
                PHINode* phi_node = dyn_cast<PHINode>(inst);
                #if 0
                unsigned incoming_val_cnt = phi_node->getNumIncomingValues();
                for(int j=0;j<incoming_val_cnt;j++)
                {
                    Value* incom_val = phi_node->getIncomingValue(j);
                    if(val!=incom_val)
                    {
                        type = find_actual_type(incom_val, false, depth+1);
                    }else
                    {
                        type = val->getType();
                    }
                    if(type->isPointerTy())
                    {
                        resolved = true;
                        break;
                    }
                }
                #endif
                type = phi_node->getType();
                if((ignore_first_ptr) && (val==orig_val))
                {
                    if(type->isPointerTy())
                        type = type->getContainedType(0);
                }
                resolved = true;
                break;
            }
            case(Instruction::Select):
            {
                SelectInst* sel_inst = dyn_cast<SelectInst>(inst);
                val = sel_inst->getTrueValue();
                type = find_actual_type(val, false, depth+1);
                if(type->isPointerTy())
                {
                    resolved = true;
                    break;
                }
                val = sel_inst->getFalseValue();
                type = find_actual_type(val, false, depth+1);
                if(type->isPointerTy())
                {
                    resolved = true;
                    break;
                }
                break;
            }
            case(Instruction::ExtractValue):
            {
                ExtractValueInst* evi = dyn_cast<ExtractValueInst>(inst);
                type = evi->getType();
                resolved = true;
                if((ignore_first_ptr) && (val==orig_val))
                {
                    if(type->isPointerTy())
                        type = type->getContainedType(0);
                }

                break;
            }
            case(Instruction::ExtractElement):
            {
                ExtractElementInst* eei = dyn_cast<ExtractElementInst>(inst);
                type = eei->getType();
                resolved = true;
                if((ignore_first_ptr) && (val==orig_val))
                {
                    if(type->isPointerTy())
                        type = type->getContainedType(0);
                }

                break;
            }
            /*
             * possible pointer manipulation through binary operators
             */
            case BinaryOperator::Add:
            case BinaryOperator::Sub:
            case BinaryOperator::Mul:
            #if 0
            {
                for(int i=0;i<2;i++)
                {
                    val = inst->getOperand(i);
                    type = find_actual_type(val,false,depth+1);
                    if(type->isPointerTy())
                    {
                        resolved = true;
                        break;
                    }
                }
                break;
            }
            #endif
            /*
             * reject all floating point operator
             *            div/mul/rem operator
             *            trunc/sext/zext operator
             *            logical operator
             */
            case BinaryOperator::FAdd:
            case BinaryOperator::FSub:
            case BinaryOperator::FMul:
            case BinaryOperator::UDiv:
            case BinaryOperator::SDiv:
            case BinaryOperator::FDiv:
            case BinaryOperator::URem:
            case BinaryOperator::SRem:
            case BinaryOperator::FRem:
            case Instruction::FPToUI:
            case Instruction::FPToSI:
            case Instruction::UIToFP:
            case Instruction::SIToFP:
            case Instruction::FPTrunc:
            case Instruction::FPExt:
            case Instruction::Trunc:
            case Instruction::ZExt:
            case Instruction::SExt:
            case Instruction::Shl:
            case Instruction::LShr:
            case Instruction::AShr:
            case Instruction::And:
            case Instruction::Or:
            case Instruction::Xor:
            case Instruction::ICmp:
            case Instruction::FCmp:
            {
                type = inst->getType();
                resolved = true;
                break;
            }
            /*
             * stop at call boundary
             */
            case Instruction::Call:
            case Instruction::Invoke:
            {
                type = inst->getType();
                resolved = true;
                break;
            }
            case Instruction::AtomicRMW:
            {
                type = inst->getType();
                resolved = true;
                break;
            }
            default:
            {
                val->dump();
                llvm_unreachable(ANSI_COLOR_RED
                                "WTF, can not handle"
                                ANSI_COLOR_RESET);
                break;
            }
        }
        //just track down struct type
        while (type->isStructTy())
        {
            StructType *sty = dyn_cast<StructType>(type);
            if (sty->getNumElements()<=0)
            {
                //FIXME: how come structtype has 0 elements?
                break;
            }
            type = sty->getElementType(0);
        }
        if (type->isPointerTy())
        {
            resolved = true;
        }
    }

out:
    assert(type!=NULL && "NULL TYPE!!!");
    while (type->isStructTy())
    {
        StructType *sty = dyn_cast<StructType>(type);
        if (sty->getNumElements()<=0)
        {
            //FIXME: how come structtype has 0 elements?
            break;
        }
        type = sty->getElementType(0);
    }
#if DEBUG_FIND_ACTUAL_TYPE
    errs()<<ANSI_COLOR_GREEN
        <<"Found type:"
        <<ANSI_COLOR_RESET;
    type->dump();
    errs()<<" `-For : ";
    orig_val->dump();
#endif
    //store into cache
    if (ignore_first_ptr)
    {
        at_cache0[orig_val] = type;
    }else
    {
        at_cache1[orig_val] = type;
    }
    return type;
}

//////////////////////////////////////////////////////////////////////////////
#if USE_MPX_TESTER
/*
 * This is MPX extension tester,
 * it grab the result of second alloc and insert the following code
 * to test the functionality of mpx 
 *
 *   bndmk r, bnd0
 *   bndstx r, bnd0
 *   bndldx r, bnd0
 *   bndcl r, bnd0
 *   bndcu r, bnd0
 *   r+=10
 *   bndcl r, bnd0; will generate #BR exception
 *
 * ----------------------------------
 *  int main()
 *  {
 *        char p[16] = "123";
 *        printf("%s\n", p);
 *        return 0;
 *  }
 */
bool llmpx::mpxTester(Module &module)
{
    for (Module::iterator f_begin = module.begin(), f_end = module.end();
            f_begin != f_end; ++f_begin)
    {
        Function *func_ptr = dyn_cast<Function>(f_begin);
        errs()<<"Function : ";
        errs()<<func_ptr->getName();
        if (func_ptr->isDeclaration())
        {
            errs()<<" is external \n";
            continue;
        }
        errs()<<"\n";
        //find and get a pointer
        
        PointerType* Int8PtrTy = Type::getInt8PtrTy(*ctx);
        //Function::iterator bb_begin = func_ptr->begin();
        BasicBlock* bb_begin = & func_ptr->getEntryBlock();
        BasicBlock::iterator II = bb_begin->begin();
        int num_alloc = 0;
        while(II!=bb_begin->end())
        {
            Instruction *I = dyn_cast<Instruction>(II);
            if (isa<AllocaInst>(I) 
                && I->getType()->isPointerTy())
            {
                errs()<<"Found AllocaInst\n";
                I->print(errs());
                errs()<<"\n";
                errs()<<" return type:";
                I->getType()->print(errs());
                errs()<<"\n";
                num_alloc++;
                if (num_alloc==2)
                {
                    break;
                }
            }
            ++II;
        }

        errs()<<"Begin instrument\n";
        /*
         * insert bndmk
         */
        Instruction *srcI = dyn_cast<Instruction>(II);

        Instruction *insertPoint = dyn_cast<Instruction>(++II);
        IRBuilder<> builder(insertPoint);
        Instruction *I = dyn_cast<Instruction>(II);
#if 1
        std::vector<Value *> args;
        //args.push_back(ConstantPointerNull::get(Int8PtrTy));

        Value* ptr_arg_for_bndmk = builder.CreateBitCast(srcI,Int8PtrTy,"");
        args.push_back(ptr_arg_for_bndmk);
        Constant* dist_arg_for_bndmk = ConstantInt::get(Type::getInt64Ty(*ctx),(9));
        args.push_back(dist_arg_for_bndmk);

#if 1
        Function *func = Intrinsic::getDeclaration(&module, 
                        Intrinsic::x86_bndmk);

        I = dyn_cast<Instruction>(II);

        errs()<<"Insert here:";
        I->print(errs());
        errs()<<"\n";

        Instruction* bndmkcall = CallInst::Create(func, args, "", I);
#endif
#if 1
        /*
         * insert bndstx
         */
        Function *func2 = Intrinsic::getDeclaration(&module, 
                        Intrinsic::x86_bndstx);
        std::vector<Value *> args2;

        args2.push_back(ptr_arg_for_bndmk);
        args2.push_back(ptr_arg_for_bndmk);
        args2.push_back(ConstantInt::get(Type::getInt64Ty(*ctx), 0));
        args2.push_back(bndmkcall);
        
        errs()<<"Insert here:";
        I = dyn_cast<Instruction>(II);
        I->print(errs());
        errs()<<"\n";

        CallInst::Create(func2, args2, "", I);
#endif
#if 1
        /*
         * insert bndldx
         */
        Function *func3 = Intrinsic::getDeclaration(&module, 
                        Intrinsic::x86_bndldx);
        std::vector<Value *> args3;

        args3.push_back(ptr_arg_for_bndmk);
        args3.push_back(ptr_arg_for_bndmk);
        args3.push_back(ConstantInt::get(Type::getInt64Ty(*ctx), 0));
        
        errs()<<"Insert here:";
        I = dyn_cast<Instruction>(II);
        I->print(errs());
        errs()<<"\n";

        Instruction* loaded_bnd = CallInst::Create(func3, args3, "", I);
#else
        Instruction* loaded_bnd = bndmkcall;
#endif
#if 1
        /*
         * insert bndcl
         */
        Function *func4 = Intrinsic::getDeclaration(&module, 
                        Intrinsic::x86_bndclrr);
        std::vector<Value *> args4;

        args4.push_back(loaded_bnd);
        args4.push_back(ptr_arg_for_bndmk);
        
        errs()<<"Insert here:";
        I = dyn_cast<Instruction>(II);
        I->print(errs());
        errs()<<"\n";

        CallInst::Create(func4, args4, "", I);
#endif
#if 1
        /*
         * insert bndcu
         */
        Function *func5 = Intrinsic::getDeclaration(&module, 
                        Intrinsic::x86_bndcurm);
        std::vector<Value *> args5;

        args5.push_back(loaded_bnd);
        args5.push_back(ptr_arg_for_bndmk);//base
        args5.push_back(ConstantInt::get(Type::getInt64Ty(*ctx),0));//index
        args5.push_back(ConstantInt::get(Type::getInt8Ty(*ctx),0));//scale
        args5.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),0));//disp

        errs()<<"Insert here:";
        I = dyn_cast<Instruction>(II);
        I->print(errs());
        errs()<<"\n";

        CallInst::Create(func5, args5, "", I);
#endif
/*
 * make some bound violations here
 */
#if 1
        /*
         * increment the ptr by one
         */
        IRBuilder<> builder2(I);
        std::vector<Value*> temp;
        temp.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),(0)));
        temp.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),(10)));

        Value* newptr = builder2.CreateGEP(srcI, temp);
        /*
         * insert bndcu
         */
        Function *func6 = Intrinsic::getDeclaration(&module, 
                        Intrinsic::x86_bndcurm);
        std::vector<Value *> args6;

        args6.push_back(loaded_bnd);
        args6.push_back(newptr);
        args6.push_back(ConstantInt::get(Type::getInt64Ty(*ctx),0));//index
        args6.push_back(ConstantInt::get(Type::getInt8Ty(*ctx),0));//scale
        args6.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),0));//disp

        errs()<<"Insert here:";
        I = dyn_cast<Instruction>(II);
        I->print(errs());
        errs()<<"\n";

        CallInst::Create(func6, args6, "", I);
#endif

#endif
        FuncCounter++;
    }
    return false;
}
#endif //USE_MPX_TESTER
//-----------------------------------------------------------------------------
/*
 * This is for CFI hardening using MPX
 * the idea is that if there's control data overwritten by buffer overflow,
 * the control data will be a mismatch with the registered bound in MPX table
 * currently we assume that return address is protected by stack protector, and
 * we only handle indirect call instruction
 * as indirect jmp is not possible to handle from IR?
 * FIXME: do we also need to care register spill that might be overwritten 
 *        by buffer overflow?
 *        For C++, we also need to protect vtable pointer
 */
void llmpx::harden_cfi(Module& module)
{

    std::set<Value*> checked_val;
    std::set<Value*> chk_cxx_vt;
    for (Module::iterator f_begin = module.begin(), f_end = module.end();
        f_begin != f_end; ++f_begin)
    {
        Function *func_ptr = dyn_cast<Function>(f_begin);
        if (func_ptr->isDeclaration())
            continue;
        if (is_in_do_not_instrument_list(func_ptr->getName()))
            continue;
        //errs()<<ANSI_COLOR_GREEN<<func_ptr->getName()<<ANSI_COLOR_RESET<<"\n";
        std::set<BasicBlock*> bb_visited;
        std::queue<BasicBlock*> bb_work_list;
        bb_work_list.push(&func_ptr->getEntryBlock());
        while (bb_work_list.size())
        {
            BasicBlock* bb = bb_work_list.front();
            bb_work_list.pop();
            if (bb_visited.count(bb))
                continue;
            bb_visited.insert(bb);
            for (BasicBlock::iterator ii = bb->begin(), ie = bb->end();
                ii!=ie; ++ii)
            {
                Instruction *I = dyn_cast<Instruction>(ii);
                if (checked_val.count(I))
                {
                    continue;
                }
                checked_val.insert(I);
                //if this is a indirect call/invoke instruction,
                //we need to verify the address
                switch (I->getOpcode())
                {
                    #if 0
                    case (Instruction::Alloca):
                    {
                        //should protect object pointer, since this points to 
                        //vtable and should be protected from buffer overflow
                        AllocaInst* ai = dyn_cast<AllocaInst>(I);
                        if (!ai->hasNUsesOrMore(1))
                        {
                            //dead alloca will be deleted
                            break;
                        }
                        if (!ai->getType()->getContainedType(0)->isPointerTy())
                        {
                            //Alloca of non-pointer type is not checked
                            break;
                        }
                        if (!ai->getType()
                            ->getContainedType(0)
                            ->getContainedType(0)
                            ->isStructTy())
                        {
                            //class is actually struct, we are protecting all 
                            //struct type as well...
                            break;
                        }
                        //add check at use site of those alloca
                        chk_cxx_vt.insert(ai);
                        break;
                    }
                    #endif
                    case (Instruction::Call):
                    {
                        CallInst* call_inst = dyn_cast<CallInst>(I);
                        Function* called_func = call_inst->getCalledFunction();
                        if (called_func)
                            break;
                        Value* called_value = call_inst->getCalledValue();
                        //need to verify whether called value is good or not...
                        if (!isa<LoadInst>(called_value))
                        {
                            //TODO: handle non-load instruction
                            //errs()<<ANSI_COLOR_RED<<"unchecked call:";
                            //I->dump();
                            //called_value->dump();
                            //errs()<<ANSI_COLOR_RESET;
                            break;
                        }
                        //skip if already checked
                        if (checked_val.count(called_value))
                        {
                            break;
                        }
                        //great, we have a load instruction,
                        //we need to load its bound and do the check before
                        //calling the function pointer
                        LoadInst* li = dyn_cast<LoadInst>(called_value);
                        Value* ptr_operand = li->getPointerOperand();
                        IRBuilder<> builder(I);
                        Value* ptr_cast = builder.CreateBitCast(ptr_operand, Type::getInt8PtrTy(*ctx), ""); 
                        Value* loaded_ptr_cast = builder.CreateBitCast(li, Type::getInt8PtrTy(*ctx), ""); 
                        std::vector<Value*> args0;
                        args0.push_back(ptr_cast);//addr
                        args0.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(*ctx)));//ptrval
                        args0.push_back(ConstantInt::get(Type::getInt32Ty(*ctx), 0));//offset
                        Value* bnd = builder.CreateCall(mpx_bndldx, args0, "");
                        std::vector<Value*> args1;
                        args1.push_back(bnd);
                        args1.push_back(loaded_ptr_cast);
                        args1.push_back(ConstantInt::get(Type::getInt64Ty(*ctx),0));
                        args1.push_back(ConstantInt::get(Type::getInt8Ty(*ctx),1));
                        args1.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),0));

                        builder.CreateCall(mpx_bndclrm, args1);
                        builder.CreateCall(mpx_bndcurm, args1);
                        TotalBNDLDXAdded++;
                        TotalChecksAdded++;
                        break;
                    }
                    case (Instruction::Invoke):
                    {
                        InvokeInst* invoke_inst = dyn_cast<InvokeInst>(I);
                        Function* called_func = invoke_inst->getCalledFunction();
                        if (called_func)
                            break;
                        Value* called_value = invoke_inst->getCalledValue();
                        //need to verify whether called value is good or not...
                        if (!isa<LoadInst>(called_value))
                        {
                            //TODO: handle non-load instruction
                            //errs()<<ANSI_COLOR_RED<<"unchecked call:";
                            //I->dump();
                            //called_value->dump();
                            //errs()<<ANSI_COLOR_RESET;
                            break;
                        }
                        //skip if already checked
                        if (checked_val.count(called_value))
                        {
                            break;
                        }
                        //great, we have a load instruction,
                        //we need to load its bound and do the check before
                        //calling the function pointer
                        LoadInst* li = dyn_cast<LoadInst>(called_value);
                        Value* ptr_operand = li->getPointerOperand();
                        IRBuilder<> builder(I);
                        Value* ptr_cast = builder.CreateBitCast(ptr_operand, Type::getInt8PtrTy(*ctx), ""); 
                        Value* loaded_ptr_cast = builder.CreateBitCast(li, Type::getInt8PtrTy(*ctx), ""); 
                        std::vector<Value*> args0;
                        args0.push_back(ptr_cast);//addr
                        args0.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(*ctx)));//ptrval
                        args0.push_back(ConstantInt::get(Type::getInt32Ty(*ctx), 0));//offset
                        Value* bnd = builder.CreateCall(mpx_bndldx, args0, "");
                        std::vector<Value*> args1;
                        args1.push_back(bnd);
                        args1.push_back(loaded_ptr_cast);
                        args1.push_back(ConstantInt::get(Type::getInt64Ty(*ctx),0));
                        args1.push_back(ConstantInt::get(Type::getInt8Ty(*ctx),1));
                        args1.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),0));

                        builder.CreateCall(mpx_bndclrm, args1);
                        builder.CreateCall(mpx_bndcurm, args1);
                        TotalBNDLDXAdded++;
                        TotalChecksAdded++;

                        break;
                    }
                    case (Instruction::Ret):
                    {
                        //if return a function pointer, then it need to be checked
                        ReturnInst* ri = dyn_cast<ReturnInst>(I);
                        Value* ret_val = ri->getReturnValue();
                        if (!ret_val)
                            break;
                        if (!ret_val->getType()->isPointerTy())
                            break;
                        if (!ret_val->getType()->getContainedType(0)->isFunctionTy())
                            break;
                        //errs()<<ANSI_COLOR_GREEN<<"return function pointer:";
                        //I->dump();
                        //errs()<<ANSI_COLOR_YELLOW<<"   in "
                        //    <<func_ptr->getName()
                        //    <<ANSI_COLOR_RESET<<"\n";
                        assert(checked_val.count(ret_val) && "This returned function pointer should have been checked" );
                        break;
                    }
                    case (Instruction::Load):
                    {
                        //load of function pointer?
                        LoadInst* li = dyn_cast<LoadInst>(I);
                        if (!li->getType()->isPointerTy())
                            break;
                        Type* ctype = li->getType()->getContainedType(0);
                        if (ctype->isFunctionTy())
                            goto load_chk_cond2;
                        //C++ load of ``this'' pointer
                        if (ctype->isStructTy())
                        {
                            StructType* st = dyn_cast<StructType>(ctype);
                            assert(st->hasName() && "should have name");
                            if (st->getName().find("class.") == 0)
                            {
                                goto load_chk_cond2;
                            }
                            //errs()<<ANSI_COLOR_YELLOW<<"NC";
                            //li->dump();
                            //errs()<<ANSI_COLOR_RESET;
                            break;
                        }else
                        {
                            break;
                        }
                        #if 1
load_chk_cond2:
                        //skip if not stored, not used by call/invoke
                        //TODO: need to check through all use chain!
                        bool need_check = false;
                        for (auto *U: li->users())
                        {
                            if (isa<CallInst>(U) || isa<InvokeInst>(U))
                            {
                                need_check = true;
                                break;
                            }else if (isa<StoreInst>(U))
                            {
                                need_check = true;
                                break;
                            }/*else if(isa<GetElementPtrInst>(U))
                            {
                                need_check = true;
                                break;
                            }*/
                        }
                        if (!need_check)
                            break;
                        #endif
                        //add check
                        Value* ptr_operand = li->getPointerOperand();
                        IRBuilder<> builder(GetNextInstruction(I));
                        Value* ptr_cast = builder.CreateBitCast(ptr_operand, Type::getInt8PtrTy(*ctx), ""); 
                        Value* loaded_ptr_cast = builder.CreateBitCast(li, Type::getInt8PtrTy(*ctx), ""); 
                        std::vector<Value*> args0;
                        args0.push_back(ptr_cast);//addr
                        args0.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(*ctx)));//ptrval
                        args0.push_back(ConstantInt::get(Type::getInt32Ty(*ctx), 0));//offset
                        Value* bnd = builder.CreateCall(mpx_bndldx, args0, "");
                        std::vector<Value*> args1;
                        args1.push_back(bnd);
                        args1.push_back(loaded_ptr_cast);
                        args1.push_back(ConstantInt::get(Type::getInt64Ty(*ctx),0));
                        args1.push_back(ConstantInt::get(Type::getInt8Ty(*ctx),1));
                        args1.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),0));

                        builder.CreateCall(mpx_bndclrm, args1);
                        builder.CreateCall(mpx_bndcurm, args1);
                        TotalBNDLDXAdded++;
                        TotalChecksAdded++;

                        //errs()<<ANSI_COLOR_GREEN<<"load function pointer:";
                        //I->dump();
                        //errs()<<ANSI_COLOR_RESET;
                        break;
                    }
                    case (Instruction::Store):
                    {
                        //this currently have a problem, if the val is overwritten
                        //by attacker, we will generate new bound corresponding to
                        //the new address, thus not correct, we need to propagate its
                        //original bound instead of creating a new bound

                        //if store type is function pointer,
                        //store this in MPX table as well
                        StoreInst* si = dyn_cast<StoreInst>(I);
                        Value* ptr_operand = si->getPointerOperand();//destination
                        Value* val_operand = si->getValueOperand();//function pointer
                        //find out what is actually being stored
                        if (val_operand->getType()->isPointerTy())
                        {
                            Type* ctype = val_operand->getType()->getContainedType(0);
                            if (ctype->isFunctionTy())
                            {
                                goto store_check;
                            }
                            if (ctype->isStructTy())
                            {
                                StructType* st = dyn_cast<StructType>(ctype);
                                assert(st->hasName() && "struct should have name");
                                if (st->getName().find("class.") == 0)
                                {
                                    goto store_check;
                                }
                                //store of pointer whos is not pointing to struct 
                                //starting with name "class"
                                break;
                            }
                            if (ctype->isPointerTy())
                            {
                                //need to know whether this is doing vtable binding
                                //should store to class type pointer address
                                //goto store_check;
                            }
                            //errs()<<ANSI_COLOR_RED<<"NS";
                            //si->dump();
                            //errs()<<ANSI_COLOR_RESET;
                            break;
                        }else
                        {
                            if (!ptr_operand->stripPointerCasts()->getType()->isPointerTy())
                            {
                                break;
                            }
                            if (ptr_operand->stripPointerCasts()->getType()->getContainedType(0)->isFunctionTy())
                            {
                                goto store_check;
                            }
                            break;
                        }
store_check:
                        //we are storing a function pointer
                        IRBuilder<> builder(I);
                        Value* bnd;
                        //investigate stored value
                        //using function address directly
                        //      ConstantNull pointer,
                        //      function argument
                        //      return'd function address
                        //      PHINode
                        //      SelectInst
                        // is considered safe, and should have been validated
                        Value* bare_val = val_operand->stripPointerCasts();
                        if (isa<Function>(bare_val) 
                            || isa<ConstantPointerNull>(bare_val)
                            || (!(isa<GlobalVariable>(bare_val)||isa<Instruction>(bare_val)))//function argument
                            || (isa<CallInst>(bare_val))
                            || (isa<PHINode>(bare_val))
                            || (isa<SelectInst>(bare_val))
                            || (isa<InvokeInst>(bare_val)))
                        {
                            //use bndmk if its absolute function pointer
                            std::vector<Value*> args0;
                            args0.push_back(builder.CreateBitCast(val_operand, Type::getInt8PtrTy(*ctx), ""));
                            args0.push_back(ConstantInt::get(Type::getInt64Ty(*ctx),0));
                            bnd = builder.CreateCall(mpx_bndmk, args0, "");
                        }else if (isa<LoadInst>(bare_val))
                        {
                            //should have been checked
                            //val_operand->dump();
                            //errs()<<ANSI_COLOR_RED
                            //    <<"store loaded function pointer?"
                            //    <<ANSI_COLOR_RESET<<"\n";
                            std::vector<Value*> args0;
                            args0.push_back(builder.CreateBitCast(val_operand, Type::getInt8PtrTy(*ctx), ""));
                            args0.push_back(ConstantInt::get(Type::getInt64Ty(*ctx),0));
                            bnd = builder.CreateCall(mpx_bndmk, args0, "");
                            assert(checked_val.count(bare_val) && "This load should have been checked" );
                        }else
                        {
                            //use bndldx if it is something else
                            errs()<<ANSI_COLOR_YELLOW<<"unhandled:";
                            I->dump();
                            errs()<<ANSI_COLOR_RED<<"  ";
                            val_operand->dump();
                            errs()<<ANSI_COLOR_RESET;
                            assert( 0 && "unhandled instruction" );
                            break;
                        }

                        std::vector<Value*> args1;
                        args1.push_back(builder.CreateBitCast(ptr_operand, Type::getInt8PtrTy(*ctx), ""));
                        args1.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(*ctx)));
                        args1.push_back(ConstantInt::get(Type::getInt32Ty(*ctx),0));
                        args1.push_back(bnd);
                        builder.CreateCall(mpx_bndstx, args1, "");
                        TotalBNDMKAdded++;
                        TotalBNDSTXAdded++;
                        break;
                    }

                    default:
                        //dont care
                        break;
                }
            }
            for (succ_iterator si = succ_begin(bb), se = succ_end(bb);
                si != se; ++si)
            {
                bb_work_list.push(cast<BasicBlock>(*si));
            }
        }
    }
}

static RegisterPass<llmpx>
XXX("llmpx", "llmpx Pass (with getAnalysisUsage implemented)");

