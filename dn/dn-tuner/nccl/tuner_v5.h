/*************************************************************************
 * Drivenets 2026-08-16 — v5 tuner API, matched to the DEPLOYED librccl.
 *
 * Source of truth: RCCL 2.28.3-develop:2e42aa8 (the build at
 * /opt/shared/ylerman/GPU-107/bin/librccl.so.1.0, built 2026-07-12).
 * This header reproduces src/include/plugin/tuner/tuner_v5.h as of public
 * ROCm/rccl commit 4ca9ee6b65 (the parent of 212d82678, "Enable tuning of
 * RCCL tuning constants", 2026-03-09), because the deployed binary predates
 * that commit:
 *   readelf -sW librccl.so.1.0 | grep TunerConstantsDefaults
 *     -> _ZL26ncclTunerConstantsDefaults  1152 OBJECT
 *   1152 bytes = 144 doubles = the layout WITHOUT the bwRatio field that
 *   212d82678 appended. Do NOT add bwRatio here: RCCL passes a pointer to a
 *   1152-byte struct inside ncclComm, and writing past it corrupts the comm.
 *
 * Requires the common types (ncclFunc_t, ncclResult_t, ncclDebugLogger_t,
 * NCCL_NUM_* ) from tuner.h — include that first.
 ************************************************************************/

#ifndef DN_TUNER_V5_H_
#define DN_TUNER_V5_H_

#include <stdint.h>

// NVL domain information struct
typedef struct {
  int nNvlDomains;                    // number of NVLink domains
  int minRanksPerNvlDomain;           // minimum ranks across all NVLink domains
  int maxRanksPerNvlDomain;           // maximum ranks across all NVLink domains
} ncclNvlDomainInfo_v5_t;

#define NCCL_NUM_ALGORITHMS_V5 7 // Tree/Ring/CollNet*/PAT
#define NCCL_NUM_PROTOCOLS_V5 3 // Simple/LL/LL128
#define NCCL_NUM_HW_LINKS_V5 3
#define NCCL_NUM_COMPCAPS_V5 4
#define NCCL_NUM_TUNING_SCALES_V5 3

typedef struct {
  double baseLatencies [NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5];
  double hwLatencies [NCCL_NUM_HW_LINKS_V5][NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5];

  double llMaxBws [NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
  double perChMaxRingLL128Bws [NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
  double perChMaxTreeLL128Bws [NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
  double perChMaxTreeBws [NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
  double perChMaxNVLSTreeBws [NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
  // NO bwRatio field on this runtime — see header comment before adding one.
} ncclTunerConstants_v5_t;

_Static_assert(sizeof(ncclTunerConstants_v5_t) == 1152,
               "must match ncclTunerConstantsDefaults (1152 bytes) in the deployed librccl.so.1.0");

// API to be implemented by external tuner
typedef struct {
  // Name of the tuner
  const char* name;

  // Initializes tuner states.
  // Inputs:
  //   - commId: communicator identifier
  //   - nRanks: number of ranks in current communicator. Each communicator initialize its own tuner.
  //   - nNodes: number of nodes in current communicator.
  //   - logFunction: a logFunction can be useful to integrate logging together with NCCL core.
  //   - nvlDomainInfo: NVL domain information struct
  // Outputs:
  //   - context: tuner context object
  // Input/Output:
  //   - constants: tuner constants
  ncclResult_t (*init)(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction,
                      ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_v5_t* constants);

  // Identical to v4 getCollInfo — see tuner.h.
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels);

  // Terminates the plugin and cleans up any resources that the plugin allocated.
  // context: tuner context object
  ncclResult_t (*finalize)(void* context);
} ncclTuner_v5_t;

#endif
