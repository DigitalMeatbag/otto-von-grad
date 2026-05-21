#include "ovg_error.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static OvgFatalFn      g_handler  = NULL;
static volatile int    g_in_fatal = 0;

void ovg_set_fatal_handler(OvgFatalFn fn) {
    g_handler  = fn;
    g_in_fatal = 0;  /* longjmp in a capture handler skips the reset; clear it here */
}

void ovg_fatal(const char *fmt, ...) {
    if (g_in_fatal) {
        /* Re-entrant call: the custom handler itself triggered a fatal error.
         * Abort immediately to avoid infinite recursion. */
        fputs("ovg_fatal: re-entrant call, aborting\n", stderr);
        abort();
    }
    g_in_fatal = 1;

    char msg[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    if (g_handler) {
        g_handler(msg);
    } else {
        fprintf(stderr, "%s\n", msg);
    }

    exit(1);
}
