/*
 * wasm_trace.c — the bench's view into a running Mimir (browser build only).
 *
 * The browser build first runs tools/wasm_instrument.py, which copies src/
 * and puts MIMIR_TRACE("name") at the top of every function.  That reports
 * the function to the page on entry and, through C's cleanup attribute, on
 * exit by any return path.  The page (Workbench Lab's Mimir bench) turns the
 * stream into pulses running along the board's traces.  Any function added
 * later is traced the same way with no edits.  (clang's -finstrument-functions
 * would do this too, but it needs __builtin_return_address, which plain
 * WebAssembly does not have.)
 *
 * The hooks themselves must not be instrumented, or they would call
 * themselves forever.
 */
#ifdef MIMIR_WASM

#define NO_TRACE __attribute__((no_instrument_function))

__attribute__((import_module("mimir"), import_name("trace_enter")))
void js_trace_enter(void *fn);

__attribute__((import_module("mimir"), import_name("trace_exit")))
void js_trace_exit(void *fn);

__attribute__((import_module("mimir"), import_name("probe")))
void js_probe(const char *name, float value);

/* Called by MIMIR_TRACE("name") at the top of every function (inserted at
 * build time by tools/wasm_instrument.py).  The page caches names by
 * pointer, so each name string is read once. */
NO_TRACE int mimir_trace_enter(const char *name) {
    js_trace_enter((void *)name);
    return 0;
}

/* Runs automatically when the traced function returns, on every path. */
NO_TRACE void mimir_trace_leave(int *unused) {
    (void)unused;
    js_trace_exit(0);
}

/* MIMIR_PROBE("name", value) in any function shows a live number on its chip. */
NO_TRACE void mimir_probe(const char *name, float value) {
    js_probe(name, value);
}

#endif
