
// setup code
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

# define LEN 122

uint64_t start = 0;
uint64_t end = 0;

float random_float() {
    return (float)(rand()) / (float)(RAND_MAX);
}

uint8_t random_uint8() {
    return (uint8_t)(rand() % 256 - 128);  
}

int32_t random_int() {
    return rand();  
}

// include statements 
#include "include/gemmini_params.h" 
#include "include/gemmini.h"
//# define LEN 200, change as needed
//note elem_t is defined in gemmini_params.h and is defaulted to int8_t

void ol_l2_cpu2_gemmini(elem_t n, elem_t pred[LEN][LEN], elem_t truth[LEN][LEN], elem_t out[LEN][LEN]){
    static elem_t temp0[LEN][LEN]; 
    for (int i = 0; i < n; i++) { 
     	 temp0[i][0] = truth[i][0]; 
     } 
    static elem_t temp1[LEN][LEN]; 
    for (int i = 0; i < n; i++) { 
     	 temp1[i][0] = pred[i][0]; 
     } 
    tiled_resadd_auto(n, n, 1, -1, 1, temp0[0], temp1[0], out[0], false, WS); 

}

int32_t* ol_l2_cpu2_gemmini_glued (int32_t n, int32_t pred[LEN], int32_t truth[LEN]){
    static elem_t glued_32[LEN][LEN];

    for (int i = 0; i < LEN; i++) { 
        glued_32[i][0] = pred[i];
    }

    static elem_t glued_33[LEN][LEN];

    for (int i = 0; i < LEN; i++) { 
        glued_33[i][0] = truth[i];
    }

    static int32_t out [LEN][LEN];
    ol_l2_cpu2_gemmini(n, glued_32, glued_33, out);
    static int32_t out_postprocess [LEN]; 


    for (int i = 0; i < LEN; i++) {
        out_postprocess[i] = out[i][0];
    }

    return out_postprocess;
}    


int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);
    unsigned long long totalTime = 0;

    static int32_t w[LEN][LEN];
    for (int i = 0; i < LEN; i++) {
        w[i][0] = random_int();
    }
    static int32_t w2[LEN][LEN];
    for (int i = 0; i < LEN; i++) {
        w2[i][0] = random_int();
    }
    int32_t n = LEN;
    static int32_t out [LEN][LEN];
    start = read_cycles();
    ol_l2_cpu2_gemmini(n, w, w2, out);
    end = read_cycles();
    totalTime += end - start;

  
  
    printf("ol_l2_cpu2_gemmini_gemmini");
    printf("%llu\n", totalTime);
    printf("%llu\n", totalTime);
    exit(0);
}
