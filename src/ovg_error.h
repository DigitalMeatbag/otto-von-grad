#ifndef OVG_ERROR_H
#define OVG_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Centralized fatal error handling for otto-von-grad.
 *
 * ovg_fatal() is called throughout the library instead of fprintf+exit(1).
 * It formats the message, invokes the installed handler (if any), then always
 * calls exit(1) — the handler is a pre-exit hook, not a recovery path.
 *
 * Thread safety: ovg_set_fatal_handler() is NOT thread-safe. Call it once at
 * program startup before any threads are spawned (same convention as
 * sqlite3_config / curl_global_init).
 */

typedef void (*OvgFatalFn)(const char *msg);

/* Install a custom handler. Pass NULL to restore the default handler.
 * The default handler prints msg to stderr; exit(1) is always called after. */
void ovg_set_fatal_handler(OvgFatalFn fn);

/* Format a fatal error message, call the installed handler, then exit(1).
 * Never returns. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
void ovg_fatal(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* OVG_ERROR_H */
