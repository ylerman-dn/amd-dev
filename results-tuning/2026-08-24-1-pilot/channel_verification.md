=== CONFIG ARM (r1, all 8 ranks) - plugin-applied vs executed ===
      size         applied ch  executed algo/proto  executed ch
      4096                 16              TREE/LL            2   <-- differs
      8192                  4              TREE/LL            4
     16384                 40              TREE/LL            8   <-- differs
     32768                 16              TREE/LL           16
     65536                 32              TREE/LL           32
    131072                 32              TREE/LL           32
    262144                 32              RING/LL           32
    524288                 48              RING/LL           43   <-- differs
   1048576                 48              RING/LL           47   <-- differs
   2097152                 48          RING/SIMPLE           43   <-- differs
   4194304                 40          RING/SIMPLE           40
   8388608                 48          RING/SIMPLE           47   <-- differs
  16777216                 48          RING/SIMPLE           48
  33554432                 40          RING/SIMPLE           40
  67108864                 48          RING/SIMPLE           48
 134217728                 48          RING/SIMPLE           48
 268435456                 48          RING/SIMPLE           48
 536870912                 48          RING/SIMPLE           48

=== DEFAULT ARM (r1) - RCCL's own choice, executed ===
      size                  -  executed algo/proto  executed ch
      4096                  -              TREE/LL            1
      8192                  -              TREE/LL            1
     16384                  -              TREE/LL            1
     32768                  -              TREE/LL            2
     65536                  -              TREE/LL            4
    131072                  -              TREE/LL            8
    262144                  -              TREE/LL           16
    524288                  -          RING/SIMPLE           16
   1048576                  -          RING/SIMPLE           32
   2097152                  -          RING/SIMPLE           52
   4194304                  -          RING/SIMPLE           52
   8388608                  -          RING/SIMPLE           54
  16777216                  -          RING/SIMPLE           56
  33554432                  -          RING/SIMPLE           56
  67108864                  -          RING/SIMPLE          222
 134217728                  -          RING/SIMPLE          222
 268435456                  -          RING/SIMPLE          223
 536870912                  -          RING/SIMPLE          224
