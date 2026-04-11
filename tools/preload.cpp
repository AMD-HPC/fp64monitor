/*
 * Preload wrapper for __libc_start_main
 * 
 * This intercepts the C runtime startup to run cleanup code after main()
 * returns but BEFORE atexit handlers run. This ensures we shut down the
 * memory monitor and sampling thread before rocm-smi or other libraries
 * are torn down.
 *
 * Execution order:
 *   1. __libc_start_main called by C runtime
 *   2. main() executes
 *   3. main() returns
 *   4. Our wrapper runs cleanup HERE  <-- before atexit
 *   5. atexit handlers run (LIFO)
 *   6. Static destructors
 *   7. Library destructors / tool_fini via rocprofiler
 *
 * Based on pattern from:
 *   https://github.com/UO-OACISS/zerosum/blob/main/src/preload.cpp
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Forward declaration - defined in fp64_monitor_tool.cpp
extern "C" void fp64_monitor_fini();

namespace {

// Trampoline for the real main()
static int (*main_real)(int, char**, char**);

int fp64_monitor_main(int argc, char** argv, char** envp) {
    // Skip wrapping for shell scripts to avoid interference
    if (argv[0] != nullptr) {
        size_t len = strlen(argv[0]);
        
        // Skip bash
        if (len >= 4 && strncmp(argv[0] + (len - 4), "bash", 4) == 0) {
            return main_real(argc, argv, envp);
        }
        // Skip csh/tcsh
        if (len >= 3 && strncmp(argv[0] + (len - 3), "csh", 3) == 0) {
            return main_real(argc, argv, envp);
        }
        // Skip gdb
        if (len >= 3 && strncmp(argv[0] + (len - 3), "gdb", 3) == 0) {
            return main_real(argc, argv, envp);
        }
    }
    
    // Prevent re-entry
    static int reentry = 0;
    if (reentry > 0) {
        return main_real(argc, argv, envp);
    }
    reentry = 1;

    // Indicate that our wrapper is active (verbose mode only)
    if (getenv("FP64_MONITOR_VERBOSE")) {
        fputs("[FP64 Monitor] __libc_start_main wrapper active\n", stderr);
    }
    
    // Call real main
    int ret = main_real(argc, argv, envp);
    
    // Cleanup BEFORE atexit handlers run.
    // This ensures we shut down memory monitoring before rocm-smi
    // or other libraries are torn down via their atexit handlers.
    // fp64_monitor_fini has internal guard against double-call.
    fp64_monitor_fini();
    
    return ret;
}

typedef int (*libc_start_main_t)(
    int (*)(int, char**, char**), int, char**,
    int (*)(int, char**, char**), void (*)(void),
    void (*)(void), void*);

} // anonymous namespace

extern "C" {

// Must be visible for symbol interposition to work with LD_PRELOAD
__attribute__((visibility("default")))
int
__libc_start_main(int (*_main)(int, char**, char**), int _argc, char** _argv,
                  int (*_init)(int, char**, char**), void (*_fini)(void),
                  void (*_rtld_fini)(void), void* _stack_end)
{
    // Prevent re-entry
    static int reentry = 0;
    if (reentry > 0) return -1;
    reentry = 1;

    // Get address of this function for comparison
    void* this_func = __builtin_return_address(0);
    
    // Save real main function pointer
    main_real = _main;

    // Find the real __libc_start_main
    libc_start_main_t real_start_main = 
        reinterpret_cast<libc_start_main_t>(dlsym(RTLD_NEXT, "__libc_start_main"));

    if (real_start_main && reinterpret_cast<void*>(real_start_main) != this_func) {
        // Call real __libc_start_main with our wrapper as the main function
        return real_start_main(fp64_monitor_main, _argc, _argv, 
                               _init, _fini, _rtld_fini, _stack_end);
    } else {
        fputs("[FP64 Monitor] Error: could not find real __libc_start_main\n", stderr);
        return -1;
    }
}

} // extern "C"
