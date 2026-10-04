/*************************************************************************
 * Drivenets 2026-08-16 — host-side (no GPU) test for the v5 tuner port.
 *
 * Loads BOTH built plugins via dlopen and checks:
 *   1. v4 and v5 getCollInfo produce byte-identical cost tables and
 *      nChannels for the same rule file across a size sweep (parity).
 *   2. v5 init logs constants and leaves them untouched when
 *      NCCL_TUNER_CONSTANTS_FILE is unset.
 *   3. Overrides land in exactly the right slot and nowhere else.
 *   4. A malformed / out-of-range override fails init (no silent runs).
 *
 * Build & run:  make test  (see Makefile)
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdarg.h>
#include "tuner.h"
#include "tuner_v5.h"

static int failures = 0;
#define CHECK(cond, ...) do { \
  if (!(cond)) { failures++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
  else { printf("PASS: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static int log_lines = 0;
static void testLogger(ncclDebugLogLevel level, unsigned long flags, const char* file, int line,
                       const char* fmt, ...) {
  (void)level; (void)flags; (void)file; (void)line;
  log_lines++;
  if (getenv("TEST_VERBOSE")) {
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap); printf("\n");
  }
}

// v4 API struct (matches nccl/tuner.h)
typedef struct {
  const char* name;
  ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context);
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels);
  ncclResult_t (*destroy)(void* context);
} tuner_v4_api;

static void fillConstantsSentinels(ncclTunerConstants_v5_t* c) {
  double* d = (double*)c;
  for (size_t i = 0; i < sizeof(*c)/sizeof(double); i++) d[i] = 10000.0 + (double)i;
}

int main(void) {
  void* h4 = dlopen("./librccl-tunerv4-dn.so", RTLD_NOW | RTLD_LOCAL);
  void* h5 = dlopen("./librccl-tunerv5-dn.so", RTLD_NOW | RTLD_LOCAL);
  if (!h4 || !h5) { printf("dlopen failed: %s\n", dlerror()); return 1; }
  tuner_v4_api* v4 = (tuner_v4_api*)dlsym(h4, "ncclTunerPlugin_v4");
  ncclTuner_v5_t* v5 = (ncclTuner_v5_t*)dlsym(h5, "ncclTunerPlugin_v5");
  CHECK(v4 && v5, "both plugin symbols resolve (v4=%p v5=%p)", (void*)v4, (void*)v5);
  CHECK(dlsym(h4, "ncclTunerPlugin_v5") == NULL, "v4 .so does not export a v5 symbol");
  CHECK(dlsym(h5, "ncclTunerPlugin_v4") == NULL, "v5 .so does not export a v4 symbol");
  if (!v4 || !v5) return 1;

  const char* conf = getenv("TEST_RULES_FILE");
  if (!conf) { printf("set TEST_RULES_FILE\n"); return 1; }
  setenv("NCCL_TUNER_CONFIG_FILE", conf, 1);
  unsetenv("NCCL_TUNER_CONSTANTS_FILE");

  const size_t nRanks = 24, nNodes = 3;

  // ---- 1. getCollInfo parity across a size sweep --------------------------
  void *ctx4 = NULL, *ctx5 = NULL;
  ncclNvlDomainInfo_v5_t dom = { (int)nNodes, 8, 8 };
  ncclTunerConstants_v5_t cons;
  fillConstantsSentinels(&cons);
  ncclTunerConstants_v5_t consBefore = cons;

  CHECK(v4->init(nRanks, nNodes, testLogger, &ctx4) == ncclSuccess, "v4 init");
  CHECK(v5->init(&ctx5, 0xabcdef, nRanks, nNodes, testLogger, &dom, &cons) == ncclSuccess, "v5 init");
  CHECK(memcmp(&cons, &consBefore, sizeof(cons)) == 0,
        "constants untouched when NCCL_TUNER_CONSTANTS_FILE unset");

  int mismatches = 0, fired = 0;
  for (size_t nBytes = 1024; nBytes <= (size_t)1 << 30; nBytes *= 2) {
    for (int coll = 0; coll < 5; coll++) {
      float t4[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS], t5[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      float ref[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
          t4[a][p] = t5[a][p] = ref[a][p] = (a == NCCL_ALGO_COLLNET_DIRECT) ? NCCL_ALGO_PROTO_IGNORE : 1.0f;
      int ch4 = -7, ch5 = -7;
      ncclResult_t r4 = v4->getCollInfo(ctx4, (ncclFunc_t)coll, nBytes, 1,
                                        (float**)t4, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS, 0, &ch4);
      ncclResult_t r5 = v5->getCollInfo(ctx5, (ncclFunc_t)coll, nBytes, 1,
                                        (float**)t5, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS, 0, &ch5);
      if (memcmp(t4, ref, sizeof(ref)) != 0 || ch4 != 0) fired++;
      if (r4 != r5 || ch4 != ch5 || memcmp(t4, t5, sizeof(t4)) != 0) {
        mismatches++;
        printf("  mismatch at coll=%d bytes=%zu: rc %d/%d ch %d/%d\n", coll, nBytes, r4, r5, ch4, ch5);
      }
    }
  }
  CHECK(mismatches == 0, "getCollInfo parity over 1K..1G x 5 collectives (%d mismatches)", mismatches);
  CHECK(fired > 0, "at least one rule fired during the sweep (%d queries modified the outcome)", fired);
  v4->destroy(ctx4);
  v5->finalize(ctx5);

  // ---- 2. overrides land in the right slots -------------------------------
  const char* ovr_path = "/tmp/dn_tuner_test_overrides.txt";
  FILE* f = fopen(ovr_path, "w");
  fprintf(f, "# test overrides\n");
  fprintf(f, "llMaxBws[1][2]=123.5\n");
  fprintf(f, "hwLatencies[2][6][1]=9.25\n");
  fprintf(f, "baseLatencies[0][0]=0.125\n");
  fclose(f);
  setenv("NCCL_TUNER_CONSTANTS_FILE", ovr_path, 1);

  fillConstantsSentinels(&cons);
  consBefore = cons;
  ctx5 = NULL;
  CHECK(v5->init(&ctx5, 1, nRanks, nNodes, testLogger, &dom, &cons) == ncclSuccess, "v5 init with overrides");
  CHECK(cons.llMaxBws[1][2] == 123.5, "llMaxBws[1][2] override applied (got %g)", cons.llMaxBws[1][2]);
  CHECK(cons.hwLatencies[2][6][1] == 9.25, "hwLatencies[2][6][1] override applied (got %g)", cons.hwLatencies[2][6][1]);
  CHECK(cons.baseLatencies[0][0] == 0.125, "baseLatencies[0][0] override applied (got %g)", cons.baseLatencies[0][0]);
  // everything else untouched
  consBefore.llMaxBws[1][2] = 123.5;
  consBefore.hwLatencies[2][6][1] = 9.25;
  consBefore.baseLatencies[0][0] = 0.125;
  CHECK(memcmp(&cons, &consBefore, sizeof(cons)) == 0, "no other slot was modified");
  v5->finalize(ctx5);

  // ---- 3. malformed / out-of-range overrides fail init --------------------
  const char* bad[] = { "llMaxBws[4][0]=1.0",        // index out of range (d0=4)
                        "noSuchField[0][0]=1.0",      // unknown field
                        "hwLatencies[0][0]=1.0",      // wrong dimension count
                        "llMaxBws[0][0] 1.0" };       // unparseable
  for (int b = 0; b < 4; b++) {
    f = fopen(ovr_path, "w");
    fprintf(f, "%s\n", bad[b]);
    fclose(f);
    fillConstantsSentinels(&cons);
    ctx5 = NULL;
    ncclResult_t r = v5->init(&ctx5, 1, nRanks, nNodes, testLogger, &dom, &cons);
    CHECK(r != ncclSuccess, "bad override '%s' fails init (rc=%d)", bad[b], r);
    if (r == ncclSuccess) v5->finalize(ctx5);
  }

  // ---- 4. constants get logged --------------------------------------------
  unsetenv("NCCL_TUNER_CONSTANTS_FILE");
  log_lines = 0;
  fillConstantsSentinels(&cons);
  ctx5 = NULL;
  v5->init(&ctx5, 1, nRanks, nNodes, testLogger, &dom, &cons);
  CHECK(log_lines >= 48, "v5 init logs the full constants table (%d log lines)", log_lines);
  v5->finalize(ctx5);

  printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
  return failures ? 1 : 0;
}
