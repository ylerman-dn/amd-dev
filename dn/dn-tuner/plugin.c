/*************************************************************************
 * Drivenets 23/12/2025
 ************************************************************************/

#include "tuner.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#ifdef DN_TUNER_V5
#include "tuner_v5.h"
#endif

#define __hidden __attribute__ ((visibility("hidden")))
#define MAX_LINE_LENGTH 256

// CSV field indices for configuration parsing
// Format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
#define CONFIG_FIELD_COLLTYPE     0
#define CONFIG_FIELD_MINBYTES     1
#define CONFIG_FIELD_MAXBYTES     2
#define CONFIG_FIELD_ALGORITHM    3
#define CONFIG_FIELD_PROTOCOL     4
#define CONFIG_FIELD_CHANNELS     5
#define CONFIG_FIELD_NNODES       6
#define CONFIG_FIELD_NRANKS       7
#define CONFIG_FIELD_PIPEOPS      8  // Optional field
#define CONFIG_FIELD_REGBUFF      9  // Optional field

// Field count constants
#define CONFIG_FIELDS_REQUIRED    8   // Minimum required fields (up to nRanks)
#define CONFIG_FIELDS_WITH_PIPEOPS 9  // Fields including numPipeOps
#define CONFIG_FIELDS_WITH_REGBUFF 10 // Fields including both numPipeOps and regBuff
#define CONFIG_FIELDS_MAX         10  // Maximum number of fields supported

typedef struct {
  ncclFunc_t collType;
  size_t minBytes;
  size_t maxBytes;
  int algorithm;
  int protocol;
  int nChannels;
  int nNodes;
  int nRanks;
  int numPipeOps;
  int regBuff;
} TuningConfig;

typedef struct {
  TuningConfig* configs;  // Changed from static array to dynamic pointer
  int numConfigs;
  int maxConfigs;         // Added to track allocated size
  size_t nRanks;
  size_t nNodes;
  ncclDebugLogger_t logFunction;
} TunerContext;

// Parse collective type from string
static ncclFunc_t parseCollType(const char* str) {
  if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
  return ncclFuncAllReduce; // default
}

// Convert collective type to string
static const char* collTypeToString(ncclFunc_t collType) {
  switch (collType) {
    case ncclFuncBroadcast: return "broadcast";
    case ncclFuncReduce: return "reduce";
    case ncclFuncAllGather: return "allgather";
    case ncclFuncReduceScatter: return "reducescatter";
    case ncclFuncAllReduce: return "allreduce";
    default: return "unknown";
  }
}

// Parse algorithm from string (-1 means wildcard/let RCCL decide)
static int parseAlgorithm(const char* str) {
  if (strcmp(str, "-1") == 0) return -1;  // Wildcard: let RCCL decide
  if (strcmp(str, "tree") == 0) return NCCL_ALGO_TREE;
  if (strcmp(str, "ring") == 0) return NCCL_ALGO_RING;
  if (strcmp(str, "collnet_direct") == 0) return NCCL_ALGO_COLLNET_DIRECT;
  if (strcmp(str, "collnet_chain") == 0) return NCCL_ALGO_COLLNET_CHAIN;
  if (strcmp(str, "nvls") == 0) return NCCL_ALGO_NVLS;
  if (strcmp(str, "nvls_tree") == 0) return NCCL_ALGO_NVLS_TREE;
  if (strcmp(str, "pat") == 0) return NCCL_ALGO_PAT;
  return NCCL_ALGO_RING; // default
}

// Convert algorithm to string
static const char* algorithmToString(int algorithm) {
  switch (algorithm) {
    case -1: return "any";  // Wildcard: let RCCL decide
    case NCCL_ALGO_TREE: return "tree";
    case NCCL_ALGO_RING: return "ring";
    case NCCL_ALGO_COLLNET_DIRECT: return "collnet_direct";
    case NCCL_ALGO_COLLNET_CHAIN: return "collnet_chain";
    case NCCL_ALGO_NVLS: return "nvls";
    case NCCL_ALGO_NVLS_TREE: return "nvls_tree";
    case NCCL_ALGO_PAT: return "pat";
    default: return "unknown";
  }
}

// Parse protocol from string (-1 means wildcard/let RCCL decide)
static int parseProtocol(const char* str) {
  if (strcmp(str, "-1") == 0) return -1;  // Wildcard: let RCCL decide
  if (strcmp(str, "ll") == 0) return NCCL_PROTO_LL;
  if (strcmp(str, "ll128") == 0) return NCCL_PROTO_LL128;
  if (strcmp(str, "simple") == 0) return NCCL_PROTO_SIMPLE;
  return NCCL_PROTO_SIMPLE; // default
}

// Convert protocol to string
static const char* protocolToString(int protocol) {
  switch (protocol) {
    case -1: return "any";  // Wildcard: let RCCL decide
    case NCCL_PROTO_LL: return "ll";
    case NCCL_PROTO_LL128: return "ll128";
    case NCCL_PROTO_SIMPLE: return "simple";
    default: return "unknown";
  }
}

// Helper function to count valid configuration lines in file
static int countConfigLines(const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    return 0;
  }

  char line[MAX_LINE_LENGTH];
  int count = 0;

  while (fgets(line, sizeof(line), file)) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Check if line has content
    if (strlen(line) > 0) {
      count++;
    }
  }

  fclose(file);
  return count;
}

// Load configuration from file
static ncclResult_t loadConfig(TunerContext* ctx, const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "DN-TUNER/Plugin: Config file %s not found, using defaults", filename);
    }
    return ncclSuccess; // Not finding config file is not an error
  }

  // First pass: count valid configuration lines
  int configCount = countConfigLines(filename);
  if (configCount == 0) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "DN-TUNER/Plugin: No valid configurations found in %s", filename);
    }
    fclose(file);
    return ncclSuccess;
  }

  // Allocate memory for configurations based on actual count
  ctx->configs = (TuningConfig*)malloc(configCount * sizeof(TuningConfig));
  if (!ctx->configs) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "DN-TUNER/Plugin: Failed to allocate memory for %d configurations", configCount);
    }
    fclose(file);
    return ncclSystemError;
  }

  ctx->maxConfigs = configCount;
  ctx->numConfigs = 0;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "DN-TUNER/Plugin: Allocated memory for %d configurations", configCount);
  }

  // Reset file pointer to beginning
  fseek(file, 0, SEEK_SET);

  char line[MAX_LINE_LENGTH];

  while (fgets(line, sizeof(line), file) && ctx->numConfigs < ctx->maxConfigs) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Parse CSV format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
    char* token;
    char* tokens[CONFIG_FIELDS_MAX];
    int tokenCount = 0;

    // Make a copy of the line for tokenizing
    char lineCopy[MAX_LINE_LENGTH];
    strncpy(lineCopy, line, sizeof(lineCopy));
    lineCopy[sizeof(lineCopy) - 1] = '\0';

    // Tokenize by comma
    token = strtok(lineCopy, ",");
    while (token != NULL && tokenCount < CONFIG_FIELDS_MAX) {
      // Trim whitespace
      while (*token == ' ' || *token == '\t') token++;
      char* end = token + strlen(token) - 1;
      while (end > token && (*end == ' ' || *end == '\t')) {
        *end = '\0';
        end--;
      }
      tokens[tokenCount++] = token;
      token = strtok(NULL, ",");
    }

    // Validate field count: support required fields (8), with pipeOps (9), or with regBuff (10)
    if (tokenCount >= CONFIG_FIELDS_REQUIRED && tokenCount <= CONFIG_FIELDS_MAX) {
      TuningConfig* config = &ctx->configs[ctx->numConfigs];
      config->collType = parseCollType(tokens[CONFIG_FIELD_COLLTYPE]);
      config->minBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MINBYTES], NULL, 10);
      config->maxBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MAXBYTES], NULL, 10);
      config->algorithm = parseAlgorithm(tokens[CONFIG_FIELD_ALGORITHM]);
      config->protocol = parseProtocol(tokens[CONFIG_FIELD_PROTOCOL]);
      config->nChannels = atoi(tokens[CONFIG_FIELD_CHANNELS]);
      config->nNodes = atoi(tokens[CONFIG_FIELD_NNODES]);
      config->nRanks = atoi(tokens[CONFIG_FIELD_NRANKS]);

      // numPipeOps is optional (9th field, index 8)
      if (tokenCount >= CONFIG_FIELDS_WITH_PIPEOPS) {
        config->numPipeOps = atoi(tokens[CONFIG_FIELD_PIPEOPS]);
      } else {
        config->numPipeOps = -1; // -1 means match any numPipeOps
      }

      // regBuff is optional (10th field, index 9)
      if (tokenCount >= CONFIG_FIELDS_WITH_REGBUFF) {
        config->regBuff = atoi(tokens[CONFIG_FIELD_REGBUFF]);
      } else {
        config->regBuff = -1; // -1 means match any regBuff value
      }

      ctx->numConfigs++;

      if (ctx->logFunction) {
        if (config->numPipeOps == -1 && config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "DN-TUNER/Plugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks);
        } else if (config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "DN-TUNER/Plugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps);
        } else if (config->numPipeOps == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "DN-TUNER/Plugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->regBuff);
        } else {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "DN-TUNER/Plugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
        }
      }
    }
  }

  fclose(file);
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "DN-TUNER/Plugin: Loaded %d tuning configurations from %s", ctx->numConfigs, filename);
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) {
  TunerContext* ctx = (TunerContext*)malloc(sizeof(TunerContext));
  if (!ctx) return ncclSystemError;

  ctx->configs = NULL;     // Initialize to NULL
  ctx->numConfigs = 0;
  ctx->maxConfigs = 0;     // Initialize to 0
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;

  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "DN-TUNER/Plugin: Initializing tuner for %zu nodes, %zu ranks", nNodes, nRanks);
  }

  // Try to load config file from environment variable or default location
  const char* configFile = getenv("NCCL_TUNER_CONFIG_FILE");
  if (!configFile) {
    configFile = "nccl_tuner.conf"; // default config file name
  }

  ncclResult_t result = loadConfig(ctx, configFile);
  if (result != ncclSuccess) {
    if (ctx->configs) {
      free(ctx->configs);  // Clean up allocated memory on error
    }
    free(ctx);
    return result;
  }

  *context = ctx;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  TunerContext* ctx = (TunerContext*)context;
  if (!ctx) return ncclInternalError;

  // Set default channels to 0 to ensure RCCL uses its default channel selection logic in case no match is found or wildcard is used in config.
  *nChannels = 0;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "DN-TUNER/Plugin: pluginGetCollInfo called - collType=%s, nBytes=%zu, numPipeOps=%d, regBuff=%d, numConfigs=%d",
                     collTypeToString(collType), nBytes, numPipeOps, regBuff, ctx->numConfigs);
  }

  // Cast the collCostTable pointer to a 2D array to fix the segmentation fault
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  // Look for matching configuration
  for (int i = 0; i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];

    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                       "DN-TUNER/Plugin: Checking config %d - collType=%s, minBytes=%zu, maxBytes=%zu, algo=%s, proto=%s, nNodes=%d, nRanks=%d, numPipeOps=%d, regBuff=%d",
                       i, collTypeToString(config->collType), config->minBytes, config->maxBytes, algorithmToString(config->algorithm), protocolToString(config->protocol),
                       config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
    }

    // Check if this config matches the current collective, size range, topology, pipeline ops, and regBuff
    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps) &&
        (config->regBuff == -1 || config->regBuff == regBuff)) {

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "DN-TUNER/Plugin: Config matches. Applying algo=%s, proto=%s, channels=%d",
                         algorithmToString(config->algorithm), protocolToString(config->protocol), config->nChannels);
      }

      // Check if this is "channels-only" mode (both algo and proto are wildcards)
      if (config->algorithm == -1 && config->protocol == -1) {
        // Only set channels, let RCCL determine algo/proto using its internal cost model
        if (config->nChannels != -1) {
          *nChannels = config->nChannels;
        }

        if (ctx->logFunction) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "DN-TUNER/Plugin: Applied channels-only config for collType=%s, bytes=%zu: channels=%d (algo/proto determined by RCCL, nodes=%d, ranks=%d)",
                           collTypeToString(config->collType), nBytes, config->nChannels, config->nNodes, config->nRanks);
        }
        return ncclSuccess;
      }

      // Check bounds for explicit algo/proto specification
      if (config->algorithm < numAlgo && config->protocol < numProto) {
        if (table[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "DN-TUNER/Plugin: Setting cost table[%s][%s] (%p) = 0.0 (was %.1f)",
                             algorithmToString(config->algorithm), protocolToString(config->protocol),
                             &table[config->algorithm][config->protocol], table[config->algorithm][config->protocol]);
          }
          table[config->algorithm][config->protocol] = 0.0; // Set low cost to prefer this configuration

          // Only override channels if not set to -1 (keep default)
          if (config->nChannels != -1) {
            *nChannels = config->nChannels;
          }

          if (ctx->logFunction) {
            if (config->nChannels == -1) {
              ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                               "DN-TUNER/Plugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=default (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nNodes, config->nRanks);
            } else {
              ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                               "DN-TUNER/Plugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=%d (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nChannels, config->nNodes, config->nRanks);
            }
          }
          return ncclSuccess;
        } else {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "DN-TUNER/Plugin: Algorithm/protocol combination [%s][%s] is marked as IGNORE",
                             algorithmToString(config->algorithm), protocolToString(config->protocol));
          }
        }
      } else {
        if (ctx->logFunction) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "DN-TUNER/Plugin: Algorithm/protocol out of bounds - algo=%s (max %d), proto=%s (max %d)",
                           algorithmToString(config->algorithm), numAlgo, protocolToString(config->protocol), numProto);
        }
      }
    } else {
      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "DN-TUNER/Plugin: Config does not match - collType match=%d, size match=%d, nodes match=%d, ranks match=%d, pipeOps match=%d, regBuff match=%d",
                         config->collType == collType,
                         (nBytes >= config->minBytes && nBytes <= config->maxBytes),
                         (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes),
                         (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks),
                         (config->numPipeOps == -1 || config->numPipeOps == numPipeOps),
                         (config->regBuff == -1 || config->regBuff == regBuff));
      }
    }
  }

  // If no specific config found, apply default behavior
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "DN-TUNER/Plugin: No matching config found");
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginDestroy(void* context) {
  if (context) {
    TunerContext* ctx = (TunerContext*)context;
    if (ctx->configs) {
      free(ctx->configs);  // Free dynamically allocated configs array
    }
    free(context);
  }
  return ncclSuccess;
}

#define PLUGIN_NAME "DN-TUNER"

#ifdef DN_TUNER_V5
/* ---------------------------------------------------------------------------
 * v5 entry points. Same rule-file behaviour as v4 (getCollInfo is shared and
 * byte-identical); additionally v5 receives RCCL's internal cost-model
 * constants at init and may rewrite them. By default they are logged and left
 * UNTOUCHED — overrides are applied only when NCCL_TUNER_CONSTANTS_FILE is
 * set, so the parity build and the experiment build are the same binary.
 * ------------------------------------------------------------------------- */

typedef struct {
  const char* name;
  size_t offset;   // into ncclTunerConstants_v5_t
  int nd;          // 2 or 3 dimensions
  int d0, d1, d2;  // d2 unused when nd==2; innermost dimension is always 3
} ConstantsField;

static const ConstantsField kConstantsFields[] = {
  {"baseLatencies",        offsetof(ncclTunerConstants_v5_t, baseLatencies),        2, NCCL_NUM_ALGORITHMS_V5, NCCL_NUM_PROTOCOLS_V5, 0},
  {"hwLatencies",          offsetof(ncclTunerConstants_v5_t, hwLatencies),          3, NCCL_NUM_HW_LINKS_V5, NCCL_NUM_ALGORITHMS_V5, NCCL_NUM_PROTOCOLS_V5},
  {"llMaxBws",             offsetof(ncclTunerConstants_v5_t, llMaxBws),             2, NCCL_NUM_COMPCAPS_V5, NCCL_NUM_TUNING_SCALES_V5, 0},
  {"perChMaxRingLL128Bws", offsetof(ncclTunerConstants_v5_t, perChMaxRingLL128Bws), 2, NCCL_NUM_COMPCAPS_V5, NCCL_NUM_TUNING_SCALES_V5, 0},
  {"perChMaxTreeLL128Bws", offsetof(ncclTunerConstants_v5_t, perChMaxTreeLL128Bws), 2, NCCL_NUM_COMPCAPS_V5, NCCL_NUM_TUNING_SCALES_V5, 0},
  {"perChMaxTreeBws",      offsetof(ncclTunerConstants_v5_t, perChMaxTreeBws),      2, NCCL_NUM_COMPCAPS_V5, NCCL_NUM_TUNING_SCALES_V5, 0},
  {"perChMaxNVLSTreeBws",  offsetof(ncclTunerConstants_v5_t, perChMaxNVLSTreeBws),  2, NCCL_NUM_COMPCAPS_V5, NCCL_NUM_TUNING_SCALES_V5, 0},
};
#define NUM_CONSTANTS_FIELDS ((int)(sizeof(kConstantsFields)/sizeof(kConstantsFields[0])))

// Dump every constant RCCL handed us. This is the record of RCCL's default
// cost model on this system, and doubles as a layout check: a struct-size
// mismatch with the runtime would show up here as garbage values.
static void logConstants(TunerContext* ctx, const ncclTunerConstants_v5_t* c) {
  if (!ctx->logFunction) return;
  for (int f = 0; f < NUM_CONSTANTS_FIELDS; f++) {
    const ConstantsField* fld = &kConstantsFields[f];
    const double* base = (const double*)((const char*)c + fld->offset);
    if (fld->nd == 2) {
      for (int i = 0; i < fld->d0; i++)
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "DN-TUNER/Plugin: constants %s[%d] = %.6g %.6g %.6g",
                         fld->name, i, base[i*fld->d1], base[i*fld->d1+1], base[i*fld->d1+2]);
    } else {
      for (int i = 0; i < fld->d0; i++)
        for (int j = 0; j < fld->d1; j++)
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "DN-TUNER/Plugin: constants %s[%d][%d] = %.6g %.6g %.6g",
                           fld->name, i, j, base[(i*fld->d1+j)*fld->d2],
                           base[(i*fld->d1+j)*fld->d2+1], base[(i*fld->d1+j)*fld->d2+2]);
    }
  }
}

// Apply overrides from NCCL_TUNER_CONSTANTS_FILE. One per line:
//   llMaxBws[1][2]=25.0
//   hwLatencies[2][0][1]=3.5
// '#' comments and blank lines are skipped. Any unparseable line, unknown
// field, wrong index count or out-of-range index FAILS init loudly rather
// than silently measuring the wrong thing (RCCL then runs with no tuner,
// which the validator's --plugin-must-fire converts into a failed run).
static ncclResult_t applyConstantsOverrides(TunerContext* ctx, ncclTunerConstants_v5_t* c, const char* path) {
  FILE* file = fopen(path, "r");
  if (!file) {
    if (ctx->logFunction)
      ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                       "DN-TUNER/Plugin: NCCL_TUNER_CONSTANTS_FILE=%s set but unreadable — failing init", path);
    return ncclInternalError;
  }
  char line[MAX_LINE_LENGTH];
  int applied = 0;
  while (fgets(line, sizeof(line), file)) {
    char* s = line;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '#' || *s == '\n' || *s == '\0') continue;
    s[strcspn(s, "\n")] = 0;

    char name[64];
    int i = -1, j = -1, k = -1, nidx;
    double val;
    if (sscanf(s, "%63[A-Za-z0-9_][%d][%d][%d]=%lf", name, &i, &j, &k, &val) == 5) nidx = 3;
    else if (sscanf(s, "%63[A-Za-z0-9_][%d][%d]=%lf", name, &i, &j, &val) == 4) nidx = 2;
    else {
      if (ctx->logFunction)
        ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                         "DN-TUNER/Plugin: cannot parse constants override '%s' — failing init", s);
      fclose(file);
      return ncclInternalError;
    }

    const ConstantsField* fld = NULL;
    for (int f = 0; f < NUM_CONSTANTS_FIELDS; f++)
      if (strcmp(kConstantsFields[f].name, name) == 0) { fld = &kConstantsFields[f]; break; }
    if (!fld || fld->nd != nidx ||
        i < 0 || i >= fld->d0 || j < 0 || j >= fld->d1 ||
        (nidx == 3 && (k < 0 || k >= fld->d2))) {
      if (ctx->logFunction)
        ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                         "DN-TUNER/Plugin: bad constants override '%s' (unknown field, wrong dim count or index out of range) — failing init", s);
      fclose(file);
      return ncclInternalError;
    }

    double* base = (double*)((char*)c + fld->offset);
    size_t idx = (fld->nd == 3) ? ((size_t)i*fld->d1 + j)*fld->d2 + k : (size_t)i*fld->d1 + j;
    double old = base[idx];
    base[idx] = val;
    applied++;
    if (ctx->logFunction)
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "DN-TUNER/Plugin: constants override %s = %.6g (was %.6g)", s, val, old);
  }
  fclose(file);
  if (ctx->logFunction)
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "DN-TUNER/Plugin: applied %d constants override(s) from %s", applied, path);
  return ncclSuccess;
}

__hidden ncclResult_t pluginInit_v5(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                    ncclDebugLogger_t logFunction, ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                    ncclTunerConstants_v5_t* constants) {
  ncclResult_t res = pluginInit(nRanks, nNodes, logFunction, context);
  if (res != ncclSuccess) return res;
  TunerContext* ctx = (TunerContext*)*context;

  if (ctx->logFunction)
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "DN-TUNER/Plugin: v5 init commId=0x%lx nvlDomains=%d minRanksPerDomain=%d maxRanksPerDomain=%d",
                     (unsigned long)commId,
                     nvlDomainInfo ? nvlDomainInfo->nNvlDomains : -1,
                     nvlDomainInfo ? nvlDomainInfo->minRanksPerNvlDomain : -1,
                     nvlDomainInfo ? nvlDomainInfo->maxRanksPerNvlDomain : -1);

  if (constants) {
    logConstants(ctx, constants);
    const char* cfile = getenv("NCCL_TUNER_CONSTANTS_FILE");
    if (cfile && cfile[0]) {
      res = applyConstantsOverrides(ctx, constants, cfile);
      if (res != ncclSuccess) {
        pluginDestroy(*context);
        *context = NULL;
        return res;
      }
    }
  } else if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                     "DN-TUNER/Plugin: v5 init received NULL constants pointer");
  }
  return ncclSuccess;
}

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v5,
  .getCollInfo = pluginGetCollInfo,
  .finalize = pluginDestroy
};

#else /* !DN_TUNER_V5 — the unchanged v4 plugin */

const ncclTuner_v4_t ncclTunerPlugin_v4 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .destroy = pluginDestroy
};

#endif
