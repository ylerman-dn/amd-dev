// Functional test: DN_TUNER_HOT_RELOAD=1 makes the plugin re-read a changed conf file.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include "tuner.h"
typedef struct {
  const char* name;
  ncclResult_t (*init)(size_t, size_t, ncclDebugLogger_t, void**);
  ncclResult_t (*getCollInfo)(void*, ncclFunc_t, size_t, int, float**, int, int, int, int*);
  ncclResult_t (*destroy)(void*);
} tuner_v4_api;
static void lg(ncclDebugLogLevel l, unsigned long f, const char* fi, int li, const char* fmt, ...) {}
static void writeConf(const char* p, const char* algo) {
  FILE* f=fopen(p,"w");
  fprintf(f,"collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff\n");
  fprintf(f,"allreduce,262144,262144,%s,simple,16,1,8,-1,-1\n",algo);
  fclose(f);
}
int main(void){
  const char* conf="/tmp/hr_test.conf";
  writeConf(conf,"ring");
  setenv("NCCL_TUNER_CONFIG_FILE",conf,1);
  setenv("DN_TUNER_HOT_RELOAD","1",1);
  void* h=dlopen("./librccl-tunerv4-dn.so",RTLD_NOW|RTLD_LOCAL);
  if(!h){printf("dlopen: %s\n",dlerror());return 1;}
  tuner_v4_api* v4=(tuner_v4_api*)dlsym(h,"ncclTunerPlugin_v4");
  void* ctx; v4->init(8,1,lg,&ctx);
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int nch=0, fails=0;
  #define RESET for(int i=0;i<NCCL_NUM_ALGORITHMS;i++){for(int j=0;j<NCCL_NUM_PROTOCOLS;j++)table[i][j]=1.0;}
  RESET; v4->getCollInfo(ctx,ncclFuncAllReduce,262144,1,(float**)table,NCCL_NUM_ALGORITHMS,NCCL_NUM_PROTOCOLS,0,&nch);
  if(table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]==0.0 && nch==16) printf("PASS: initial rule applied (ring)\n");
  else {printf("FAIL: initial rule not applied\n"); fails++;}
  sleep(2);                      // ensure new mtime second
  writeConf(conf,"tree");        // swap rule on disk
  sleep(2);                      // pass the 1s throttle
  RESET; nch=0;
  v4->getCollInfo(ctx,ncclFuncAllReduce,262144,1,(float**)table,NCCL_NUM_ALGORITHMS,NCCL_NUM_PROTOCOLS,0,&nch);
  if(table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]==0.0 && table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]==1.0)
    printf("PASS: hot-reload picked up tree rule\n");
  else {printf("FAIL: hot-reload did not take effect\n"); fails++;}
  unsetenv("DN_TUNER_HOT_RELOAD");
  void* ctx2; v4->init(8,1,lg,&ctx2);
  writeConf(conf,"ring"); sleep(2);
  RESET; nch=0;
  v4->getCollInfo(ctx2,ncclFuncAllReduce,262144,1,(float**)table,NCCL_NUM_ALGORITHMS,NCCL_NUM_PROTOCOLS,0,&nch);
  if(table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]==0.0)
    printf("PASS: with hot-reload OFF, file change ignored (old behavior)\n");
  else {printf("FAIL: reload happened with feature off\n"); fails++;}
  printf(fails?"FAILURES\n":"ALL HOT-RELOAD TESTS PASSED\n");
  return fails;
}
