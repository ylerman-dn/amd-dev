# v5 tuner port — step-0 evidence (runtime contract verification)

Date: 2026-08-16. All binary probes against the DEPLOYED runtime
`/opt/shared/ylerman/GPU-107/bin/librccl.so.1.0` ("RCCL 2.28.3-develop:2e42aa8",
built 2026-07-12, md5 `53a5c003a247a1d0a94851855037ad85`), via `ssh amd-mi355x-1`.

## 1. The runtime probes tuner plugin v2–v5 (v5 preferred)

    strings -a librccl.so.1.0 | grep -i "tunerplugin_v" | sort -u
      -> ncclTunerPlugin_v2 / _v3 / _v4 / _v5      (no v6)

Loader order confirmed v5-first in the era source (`src/plugin/tuner.cc`, see §3).
Load lines available for run verification:

    strings -a librccl.so.1.0 | grep "TUNER/Plugin"
      -> "TUNER/Plugin: Using %s (v4)" / "TUNER/Plugin: Using %s (v5)" etc.

## 2. The runtime's v5 constants struct is the OLD 1152-byte layout (no bwRatio)

    readelf -sW librccl.so.1.0 | grep TunerConstants
      -> _ZL26ncclTunerConstantsDefaults  1152 OBJECT   (= 144 doubles)
      -> _Z26ncclTopoInitTunerConstantsP8ncclComm  29 FUNC  (a bare struct copy)
      -> 3x _ZL14ncclTuner_init...ncclNvlDomainInfo_v5_t*...ncclTunerConstants_v5_t*
         (the v2/v3/v4 adapter shims; proves the v5-shaped init signature)

Public ROCm/rccl commit 212d82678 ("[RCCL][Tuner Plugin] Enable tuning of RCCL
tuning constants", 2026-03-09) appended `bwRatio[2][7][3]` (struct -> 1488 bytes)
and rewrote `ncclTopoInitTunerConstants` with copy-loops from `rcclTuningModel`.
The deployed binary has the 1152-byte struct and the 29-byte init — that commit is
ABSENT. Therefore the runtime source sits between the NCCL 2.28.3 import
(f1308997d, 2025-09-02) and 212d82678's parent (4ca9ee6b65). Our
`dn/dn-tuner/nccl/tuner_v5.h` reproduces the 4ca9ee6b65 header and carries a
`_Static_assert(sizeof == 1152)`.

`/opt/shared/ylerman/rccl-src` is NCCL **2.30.4** (`makefiles/version.mk`) — NOT
the runtime's source; its `tuner_v5.h` includes `bwRatio` and must not be used
for this runtime (writing past 1152 bytes would corrupt `ncclComm`).

## 3. On this runtime, the AMD path likely reads NONE of the v5 constants (hypothesis → probe)

From `src/graph/tuning.cc` and `src/init.cc` at 4ca9ee6b65 (the era source):

- init order is right for tuning: `ncclTopoInitTunerConstants` (defaults) →
  `comm->tuner->init(..., &comm->tunerConstants)` → `ncclTopoTuneModel` (init.cc:1961/1964/1966).
- BUT in `ncclTopoTuneModel`, the reads of `tunerConstants.llMaxBws` /
  `perChMax*Bws` sit inside `#if !defined(__HIP_PLATFORM_AMD__)` blocks; the AMD
  branch uses `rcclTuningModel[...]` (bwRatio, hwLat, llProtoRanges,
  channelThresholds, correction factors) and the static `baseLat` array.
- `tunerConstants.baseLatencies` is read nowhere in that file's era version.

STATUS: hypothesis (source archaeology on a nearby public commit, not the exact
internal build). Falsifier queued: extreme-constants probe (`constants_extreme.txt`
— all bw tables 0.0001, TREE latencies 10000) must change RCCL's selected
algo/proto somewhere if any field is live. Selections are computed from the static
cost model, so this probe is valid even with fabric co-tenants.

## 4. Port verification (host-side, no GPU)

`dn/dn-tuner/test_v5_host.c` (make test), run on dev VM and on amd-mi355x-1
(gcc 11.4.0): both .so's export exactly one plugin symbol each; getCollInfo
byte-identical v4-vs-v5 over 1K..1G x 5 collectives (0 mismatches, 26 resp. 6
rule-firing queries with the xai conf resp. gen_3n_t1.validated.csv); constants
untouched without `NCCL_TUNER_CONSTANTS_FILE`; overrides land in exactly the
addressed slot; malformed/out-of-range overrides fail init (rc=3) so a typo can
never silently measure the wrong thing.
