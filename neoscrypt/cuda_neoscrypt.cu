
/*
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include <stdint.h>
#include <memory.h>
*/
#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h"

// extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
//cudaStream_t stream[4];
#define vectype uintx64bis
#define vectypeS uint28
//#define vectype ulonglong16
//#define vectypeS ulonglong4

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif


__device__ __align__(16) vectypeS *  W;
__device__  __align__(16) vectypeS * W2;

__device__  vectypeS* Tr;
__device__  vectypeS* Tr2;
__device__ vectypeS* Input;
__device__ vectypeS* B2;

//vectypeS *d_output;
uint32_t *d_NNonce[MAX_GPUS];
uint32_t *d_nnounce[MAX_GPUS];
unsigned long long *d_time[MAX_GPUS];

// Global streams array:
cudaStream_t g_stream[MAX_GPUS*2];

__constant__  uint32_t pTarget[8];
__constant__  uint32_t key_init[16];
__constant__  uint32_t input_init[16];
__constant__  uint32_t  c_data[80];
//__constant__  uint8_t  c_data2[320];


#define SALSA_SMALL_UNROLL 1
#define CHACHA_SMALL_UNROLL 1
#define BLAKE2S_BLOCK_SIZE    64U
#define BLAKE2S_OUT_SIZE      32U
#define BLAKE2S_KEY_SIZE      32U
#define BLOCK_SIZE            64U
#define FASTKDF_BUFFER_SIZE  256U
#define PASSWORD_LEN          80U
/// constants ///

static const __constant__  uint8 BLAKE2S_IV_Vec =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};


static const  uint8 BLAKE2S_IV_Vechost =
{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint32_t BLAKE2S_SIGMA_host[10][16] =
{
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

__constant__ uint32_t BLAKE2S_SIGMA[10][16] =
{
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

/*
__constant__ uint2 BLAKE2S_SIGMA2[80] =
{
	 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,
	 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 ,
	 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 ,
	 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 ,
	 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 ,
	 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 ,
	 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 ,
	 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 ,
	 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 ,
	 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 ,
};
*/


#define SALSA(a,b,c,d) { \
    t =a+d; t=rotate(t,  7);b^=t;    \
    t =b+a; t=rotate(t,  9);c^=t;    \
    t =c+b; t=rotate(t, 13);d^=t;    \
    t =d+c; t=rotate(t, 18);a^=t;     \
}



#define SALSA_CORE(state) { \
\
SALSA(state.s0,state.s4,state.s8,state.sc); \
SALSA(state.s5,state.s9,state.sd,state.s1); \
SALSA(state.sa,state.se,state.s2,state.s6); \
SALSA(state.sf,state.s3,state.s7,state.sb); \
SALSA(state.s0,state.s1,state.s2,state.s3); \
SALSA(state.s5,state.s6,state.s7,state.s4); \
SALSA(state.sa,state.sb,state.s8,state.s9); \
SALSA(state.sf,state.sc,state.sd,state.se); \
	}

static __forceinline__ __device__ void shift256R4(uint32_t * ret, const uint8 &vec4, uint32_t shift2)
{
	uint32_t shift = 32 - shift2;
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[0]) : "r"(0), "r"(vec4.s0), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[1]) : "r"(vec4.s0), "r"(vec4.s1), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[2]) : "r"(vec4.s1), "r"(vec4.s2), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[3]) : "r"(vec4.s2), "r"(vec4.s3), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[4]) : "r"(vec4.s3), "r"(vec4.s4), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[5]) : "r"(vec4.s4), "r"(vec4.s5), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[6]) : "r"(vec4.s5), "r"(vec4.s6), "r"(shift));
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(ret[7]) : "r"(vec4.s6), "r"(vec4.s7), "r"(shift));
	asm("shr.b32         %0, %1, %2;"     : "=r"(ret[8]) : "r"(vec4.s7), "r"(shift));


}

static __device__ __inline__ void chacha_step(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d)
{
	asm("{\n\t"
	    "add.u32 %0,%0,%1; \n\t"
	    "xor.b32 %3,%3,%0; \n\t"
	    "prmt.b32 %3, %3, 0, 0x1032; \n\t"
	    "add.u32 %2,%2,%3; \n\t"
	    "xor.b32 %1,%1,%2; \n\t"
	    "shf.l.wrap.b32 %1, %1, %1, 12; \n\t"
	    "add.u32 %0,%0,%1; \n\t"
	    "xor.b32 %3,%3,%0; \n\t"
	    "prmt.b32 %3, %3, 0, 0x2103; \n\t"
	    "add.u32 %2,%2,%3; \n\t"
	    "xor.b32 %1,%1,%2; \n\t"
	    "shf.l.wrap.b32 %1, %1, %1, 7; \n\t}"
	    : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
}

static __device__ __inline__ void chacha_step3(uint32_t* ptr)
{
	asm volatile ("{\n\t"
		".reg .u32 s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15; \n\t"
		"ld.v4.u32 {s0,s1,s2,s3}, [%0]; \n\t"
		"ld.v4.u32 {s4,s5,s6,s7}, [%0+16]; \n\t"
		"ld.v4.u32 {s8,s9,s10,s11}, [%0+32]; \n\t"
		"ld.v4.u32 {s12,s13,s14,s15}, [%0+48]; \n\t"

		// %0 -> s0
		// %1 -> s4
		// %2 -> s8
		// %3 -> s12
    "add.u32 s0,s0,s4; \n\t"
    "xor.b32 s12,s12,s0; \n\t"
    "prmt.b32 s12, s12, 0, 0x1032; \n\t"
    "add.u32 s8,s8,s12; \n\t"
    "xor.b32 s4,s4,s8; \n\t"
    "shf.l.wrap.b32 s4, s4, s4, 12; \n\t"
    "add.u32 s0,s0,s4; \n\t"
    "xor.b32 s12,s12,s0; \n\t"
    "prmt.b32 s12, s12, 0, 0x2103; \n\t"
    "add.u32 s8,s8,s12; \n\t"
    "xor.b32 s4,s4,s8; \n\t"
    "shf.l.wrap.b32 s4, s4, s4, 7; \n\t"


    "st.v4.u32  [%0], {s0,s1,s2,s3}; \n\t"
    "st.v4.u32  [%0+16], {s4,s5,s6,s7}; \n\t"
    "st.v4.u32  [%0+32], {s8,s9,s10,s11}; \n\t"
    "st.v4.u32  [%0+48], {s12,s13,s14,s15}; \n\t"
    "}"
    :: "l"(ptr));
}

static __device__ __inline__ void chacha_step4(uint32_t* X)
{
	asm volatile ("{\n\t"

		// %0 -> %0
		// %1 -> %4
		// %2 -> %8
		// %3 -> %12
    "add.u32 %0,%0,%4; \n\t"
    "xor.b32 %12,%12,%0; \n\t"
    "prmt.b32 %12, %12, 0, 0x1032; \n\t"
    "add.u32 %8,%8,%12; \n\t"
    "xor.b32 %4,%4,%8; \n\t"
    "shf.l.wrap.b32 %4, %4, %4, 12; \n\t"
    "add.u32 %0,%0,%4; \n\t"
    "xor.b32 %12,%12,%0; \n\t"
    "prmt.b32 %12, %12, 0, 0x2103; \n\t"
    "add.u32 %8,%8,%12; \n\t"
    "xor.b32 %4,%4,%8; \n\t"
    "shf.l.wrap.b32 %4, %4, %4, 7; \n\t"

		// %0 -> %1
		// %1 -> %5
		// %2 -> %9
		// %3 -> %13
    "add.u32 %1,%1,%5; \n\t"
    "xor.b32 %13,%13,%1; \n\t"
    "prmt.b32 %13, %13, 0, 0x1032; \n\t"
    "add.u32 %9,%9,%13; \n\t"
    "xor.b32 %5,%5,%9; \n\t"
    "shf.l.wrap.b32 %5, %5, %5, 12; \n\t"
    "add.u32 %1,%1,%5; \n\t"
    "xor.b32 %13,%13,%1; \n\t"
    "prmt.b32 %13, %13, 0, 0x2103; \n\t"
    "add.u32 %9,%9,%13; \n\t"
    "xor.b32 %5,%5,%9; \n\t"
    "shf.l.wrap.b32 %5, %5, %5, 7; \n\t"

		// %0 -> %2
		// %1 -> %6
		// %2 -> %10
		// %3 -> %14
    "add.u32 %2,%2,%6; \n\t"
    "xor.b32 %14,%14,%2; \n\t"
    "prmt.b32 %14, %14, 0, 0x1032; \n\t"
    "add.u32 %10,%10,%14; \n\t"
    "xor.b32 %6,%6,%10; \n\t"
    "shf.l.wrap.b32 %6, %6, %6, 12; \n\t"
    "add.u32 %2,%2,%6; \n\t"
    "xor.b32 %14,%14,%2; \n\t"
    "prmt.b32 %14, %14, 0, 0x2103; \n\t"
    "add.u32 %10,%10,%14; \n\t"
    "xor.b32 %6,%6,%10; \n\t"
    "shf.l.wrap.b32 %6, %6, %6, 7; \n\t"

		// %0 -> %3
		// %1 -> %7
		// %2 -> %11
		// %3 -> %15
    "add.u32 %3,%3,%7; \n\t"
    "xor.b32 %15,%15,%3; \n\t"
    "prmt.b32 %15, %15, 0, 0x1032; \n\t"
    "add.u32 %11,%11,%15; \n\t"
    "xor.b32 %7,%7,%11; \n\t"
    "shf.l.wrap.b32 %7, %7, %7, 12; \n\t"
    "add.u32 %3,%3,%7; \n\t"
    "xor.b32 %15,%15,%3; \n\t"
    "prmt.b32 %15, %15, 0, 0x2103; \n\t"
    "add.u32 %11,%11,%15; \n\t"
    "xor.b32 %7,%7,%11; \n\t"
    "shf.l.wrap.b32 %7, %7, %7, 7; \n\t"

		// %0 -> %0
		// %1 -> %5
		// %2 -> %10
		// %3 -> %15
    "add.u32 %0,%0,%5; \n\t"
    "xor.b32 %15,%15,%0; \n\t"
    "prmt.b32 %15, %15, 0, 0x1032; \n\t"
    "add.u32 %10,%10,%15; \n\t"
    "xor.b32 %5,%5,%10; \n\t"
    "shf.l.wrap.b32 %5, %5, %5, 12; \n\t"
    "add.u32 %0,%0,%5; \n\t"
    "xor.b32 %15,%15,%0; \n\t"
    "prmt.b32 %15, %15, 0, 0x2103; \n\t"
    "add.u32 %10,%10,%15; \n\t"
    "xor.b32 %5,%5,%10; \n\t"
    "shf.l.wrap.b32 %5, %5, %5, 7; \n\t"

		// %0 -> %1
		// %1 -> %6
		// %2 -> %11
		// %3 -> %12
    "add.u32 %1,%1,%6; \n\t"
    "xor.b32 %12,%12,%1; \n\t"
    "prmt.b32 %12, %12, 0, 0x1032; \n\t"
    "add.u32 %11,%11,%12; \n\t"
    "xor.b32 %6,%6,%11; \n\t"
    "shf.l.wrap.b32 %6, %6, %6, 12; \n\t"
    "add.u32 %1,%1,%6; \n\t"
    "xor.b32 %12,%12,%1; \n\t"
    "prmt.b32 %12, %12, 0, 0x2103; \n\t"
    "add.u32 %11,%11,%12; \n\t"
    "xor.b32 %6,%6,%11; \n\t"
    "shf.l.wrap.b32 %6, %6, %6, 7; \n\t"

		// %0 -> %2
		// %1 -> %7
		// %2 -> %8
		// %3 -> %13
    "add.u32 %2,%2,%7; \n\t"
    "xor.b32 %13,%13,%2; \n\t"
    "prmt.b32 %13, %13, 0, 0x1032; \n\t"
    "add.u32 %8,%8,%13; \n\t"
    "xor.b32 %7,%7,%8; \n\t"
    "shf.l.wrap.b32 %7, %7, %7, 12; \n\t"
    "add.u32 %2,%2,%7; \n\t"
    "xor.b32 %13,%13,%2; \n\t"
    "prmt.b32 %13, %13, 0, 0x2103; \n\t"
    "add.u32 %8,%8,%13; \n\t"
    "xor.b32 %7,%7,%8; \n\t"
    "shf.l.wrap.b32 %7, %7, %7, 7; \n\t"

		// %0 -> %3
		// %1 -> %4
		// %2 -> %9
		// %3 -> %14
    "add.u32 %3,%3,%4; \n\t"
    "xor.b32 %14,%14,%3; \n\t"
    "prmt.b32 %14, %14, 0, 0x1032; \n\t"
    "add.u32 %9,%9,%14; \n\t"
    "xor.b32 %4,%4,%9; \n\t"
    "shf.l.wrap.b32 %4, %4, %4, 12; \n\t"
    "add.u32 %3,%3,%4; \n\t"
    "xor.b32 %14,%14,%3; \n\t"
    "prmt.b32 %14, %14, 0, 0x2103; \n\t"
    "add.u32 %9,%9,%14; \n\t"
    "xor.b32 %4,%4,%9; \n\t"
    "shf.l.wrap.b32 %4, %4, %4, 7; \n\t"

    "}"
    : "+r"(X[0]), "+r"(X[1]), "+r"(X[2]), "+r"(X[3]), 
      "+r"(X[4]), "+r"(X[5]), "+r"(X[6]), "+r"(X[7]),
      "+r"(X[8]), "+r"(X[9]), "+r"(X[10]),"+r"(X[11]),
      "+r"(X[12]),"+r"(X[13]),"+r"(X[14]),"+r"(X[15])); //, "+r"(X[4]), "+r"(X[5]), "+r"(X[6]), "+r"(X[7])
}
static __device__ __inline__ void chacha_step2(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
																							 uint32_t &e, uint32_t &f, uint32_t &g, uint32_t &h)
{
	asm("{\n\t"
			".reg .u r0,r1,r2,r3; \n\t"
			"mov.b64 r0, {%0,%4}; \n\t"
			"mov.b64 r1, {%1,%5}; \n\t"
			"mov.b64 r2, {%2,%6}; \n\t"
			"mov.b64 r3, {%3,%7}; \n\t"

			"add.u64 r0,r0,r1; \n\t"
	    "xor.b64 r3,r3,r0; \n\t"
	    "prmt.b64 r3, r3, 0, 0x1032; \n\t"


	    "add.u32 %0,%0,%1; \n\t"
	    "xor.b32 %3,%3,%0; \n\t"
	    "prmt.b32 %3, %3, 0, 0x1032; \n\t"
	    "add.u32 %2,%2,%3; \n\t"
	    "xor.b32 %1,%1,%2; \n\t"
	    "shf.l.wrap.b32 %1, %1, %1, 12; \n\t"
	    "add.u32 %0,%0,%1; \n\t"
	    "xor.b32 %3,%3,%0; \n\t"
	    "prmt.b32 %3, %3, 0, 0x2103; \n\t"
	    "add.u32 %2,%2,%3; \n\t"
	    "xor.b32 %1,%1,%2; \n\t"
	    "shf.l.wrap.b32 %1, %1, %1, 7; \n\t"

	    "add.u32 %4,%4,%5; \n\t"
	    "xor.b32 %7,%7,%4; \n\t"
	    "prmt.b32 %7, %7, 0, 0x1032; \n\t"
	    "add.u32 %6,%6,%7; \n\t"
	    "xor.b32 %5,%5,%6; \n\t"
	    "shf.l.wrap.b32 %5, %5, %5, 12; \n\t"
	    "add.u32 %4,%4,%5; \n\t"
	    "xor.b32 %7,%7,%4; \n\t"
	    "prmt.b32 %7, %7, 0, 0x2103; \n\t"
	    "add.u32 %6,%6,%7; \n\t"
	    "xor.b32 %5,%5,%6; \n\t"
	    "shf.l.wrap.b32 %5, %5, %5, 7; \n\t"


	    "}"
	    : "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e), "+r"(f), "+r"(g), "+r"(h));
}

#if __CUDA_ARCH__ >=500

#define CHACHA_STEP(a,b,c,d) { \
a += b; d = __byte_perm(d^a,0,0x1032); \
c += d; b = rotate(b^c, 12); \
a += b; d = __byte_perm(d^a,0,0x2103); \
c += d; b = rotate(b^c, 7); \
	}

//#define CHACHA_STEP(a,b,c,d) chacha_step(a,b,c,d)
#else
#define CHACHA_STEP(a,b,c,d) { \
a += b; d = rotate(d^a,16); \
c += d; b = rotate(b^c, 12); \
a += b; d = rotate(d^a,8); \
c += d; b = rotate(b^c, 7); \
	}
#endif

#define CHACHA_CORE_PARALLEL(state)	 { \
 \
  chacha_step(state.lo.s0, state.lo.s4, state.hi.s0, state.hi.s4); \
  chacha_step(state.lo.s1, state.lo.s5, state.hi.s1, state.hi.s5); \
  chacha_step(state.lo.s2, state.lo.s6, state.hi.s2, state.hi.s6); \
	chacha_step(state.lo.s3, state.lo.s7, state.hi.s3, state.hi.s7); \
	chacha_step(state.lo.s0, state.lo.s5, state.hi.s2, state.hi.s7); \
  chacha_step(state.lo.s1, state.lo.s6, state.hi.s3, state.hi.s4); \
  chacha_step(state.lo.s2, state.lo.s7, state.hi.s0, state.hi.s5); \
	chacha_step(state.lo.s3, state.lo.s4, state.hi.s1, state.hi.s6); \
\
}

  // chacha_step(state.lo.s2, state.lo.s7, state.hi.s0, state.hi.s5); \
	// chacha_step(state.lo.s3, state.lo.s4, state.hi.s1, state.hi.s6); \

#define CHACHA_CORE_PARALLEL_B(state)	 { \
 \
  chacha_step4((uint32_t*)&state); \
\
}

// #define CHACHA_CORE_PARALLEL_B(state)	 { \
//  \
//   chacha_step2(state.lo.s0, state.lo.s4, state.hi.s0, state.hi.s4, state.lo.s1, state.lo.s5, state.hi.s1, state.hi.s5); \
//   chacha_step2(state.lo.s2, state.lo.s6, state.hi.s2, state.hi.s6, state.lo.s3, state.lo.s7, state.hi.s3, state.hi.s7); \
// 	chacha_step2(state.lo.s0, state.lo.s5, state.hi.s2, state.hi.s7, state.lo.s1, state.lo.s6, state.hi.s3, state.hi.s4); \
//   chacha_step2(state.lo.s2, state.lo.s7, state.hi.s0, state.hi.s5, state.lo.s3, state.lo.s4, state.hi.s1, state.hi.s6); \
// \
// }

#define CHACHA_CORE_PARALLEL2(i0,state)	 { \
 \
    CHACHA_STEP(state[2*i0].x.x, state[2*i0].z.x, state[2*i0+1].x.x, state[2*i0+1].z.x); \
    CHACHA_STEP(state[2*i0].x.y, state[2*i0].z.y, state[2*i0+1].x.y, state[2*i0+1].z.y); \
    CHACHA_STEP(state[2*i0].y.x, state[2*i0].w.x, state[2*i0+1].y.x, state[2*i0+1].w.x); \
	CHACHA_STEP(state[2*i0].y.y, state[2*i0].w.y, state[2*i0+1].y.y, state[2*i0+1].w.y); \
	CHACHA_STEP(state[2*i0].x.x, state[2*i0].z.y, state[2*i0+1].y.x, state[2*i0+1].w.y); \
    CHACHA_STEP(state[2*i0].x.y, state[2*i0].w.x, state[2*i0+1].y.y, state[2*i0+1].z.x); \
    CHACHA_STEP(state[2*i0].y.x, state[2*i0].w.y, state[2*i0+1].x.x, state[2*i0+1].z.y); \
	CHACHA_STEP(state[2*i0].y.y, state[2*i0].z.x, state[2*i0+1].x.y, state[2*i0+1].w.x); \
\
	}



// Blake2S

#define BLAKE2S_BLOCK_SIZE    64U
#define BLAKE2S_OUT_SIZE      32U
#define BLAKE2S_KEY_SIZE      32U

#if __CUDA_ARCH__ >= 500
#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
    a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
idx = BLAKE2S_SIGMA[idx0][idx1+1]; a += key[idx]; \
    a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
	}
#else
#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
    a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
idx = BLAKE2S_SIGMA[idx0][idx1+1]; a += key[idx]; \
    a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
		}
#endif

#if __CUDA_ARCH__ >= 500


#define BLAKE(a, b, c, d, key1,key2) { \
   \
    a += key1; \
    a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
    a += key2; \
    a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
	 	 }

#define BLAKE_G_PRE(idx0,idx1, a, b, c, d, key) { \
    a += key[idx0]; \
    a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
    a += key[idx1]; \
    a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
		}

#define BLAKE_G_PRE0(idx0,idx1, a, b, c, d, key) { \
    \
    a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
    \
    a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
				}

#define BLAKE_G_PRE1(idx0,idx1, a, b, c, d, key) { \
    a += key[idx0]; \
    a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
    \
    a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
				}

#define BLAKE_G_PRE2(idx0,idx1, a, b, c, d, key) { \
    \
    a += b; d = __byte_perm(d^a,0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
    a += key[idx1]; \
    a += b; d = __byte_perm(d^a,0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
				}

#else
#define BLAKE(a, b, c, d, key1,key2) { \
  \
    a += key1; \
    a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
    a += key2; \
    a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
	}


#define BLAKE_G_PRE(idx0, idx1, a, b, c, d, key) { \
    a += key[idx0]; \
    a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
    a += key[idx1]; \
    a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
				}

#define BLAKE_G_PRE0(idx0, idx1, a, b, c, d, key) { \
     \
    a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
    \
    a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
								}

#define BLAKE_G_PRE1(idx0, idx1, a, b, c, d, key) { \
    a += key[idx0]; \
    a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
    a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
								}

#define BLAKE_G_PRE2(idx0, idx1, a, b, c, d, key) { \
     \
    a += b; d = rotate(d^a,16); \
	c += d; b = rotateR(b^c, 12); \
    a += key[idx1]; \
    a += b; d = rotateR(d^a,8); \
	c += d; b = rotateR(b^c, 7); \
								}


#endif




#define BLAKE_Ghost(idx0, idx1, a, b, c, d, key) { \
idx = BLAKE2S_SIGMA_host[idx0][idx1]; a += key[idx]; \
    a += b; d = ROTR32(d^a,16); \
	c += d; b = ROTR32(b^c, 12); \
idx = BLAKE2S_SIGMA_host[idx0][idx1+1]; a += key[idx]; \
    a += b; d = ROTR32(d^a,8); \
	c += d; b = ROTR32(b^c, 7); \
		}


static __forceinline__ __device__ void Blake2S(uint32_t *out, const uint32_t* __restrict__  inout, const  uint32_t * __restrict__ TheKey)
{
	uint16 V;

	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;


	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE0(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE0(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE1(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE1(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);



//		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE2(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//		{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE1(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE1(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


//		{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE2(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


//		{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE0(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//		{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE1(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE0(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

//		{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE2(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;


	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	for (int x = 4; x < 10; ++x)
	{
		BLAKE_G(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_G(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_G(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_G(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_G(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_G(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_G(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_G(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	((uint8*)out)[0] = V.lo;

}

static __forceinline__ __device__ void Blake2S_v2(uint32_t *out, const uint32_t* __restrict__  inout, const  uint32_t * __restrict__ TheKey)
{
	uint16 V;

	uint2 idx;
	uint8 tmpblock;
	//	uint16 inout[1];
	//	inout[0] = ((uint16*)inoutE)[0];

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;


	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE0(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE0(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE1(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE1(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);



	//		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE2(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE1(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE1(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE2(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);


	//		{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE0(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE1(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE0(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	//		{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE2(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;


	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);


	//		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	//#pragma unroll

	//		13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	//		6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
	//		10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[9], inout[0]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[5], inout[7]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[2], inout[4]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[10], inout[15]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[14], inout[1]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[11], inout[12]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[6], inout[8]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[3], inout[13]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[2], inout[12]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[6], inout[10]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[0], inout[11]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[8], inout[3]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[4], inout[13]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[7], inout[5]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[15], inout[14]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[1], inout[9]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[12], inout[5]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[1], inout[15]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[14], inout[13]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[4], inout[10]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[0], inout[7]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[6], inout[3]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[9], inout[2]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[8], inout[11]);

	//		13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	//		6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[13], inout[11]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[7], inout[14]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[12], inout[1]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[3], inout[9]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[5], inout[0]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[15], inout[4]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[8], inout[6]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[2], inout[10]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[6], inout[15]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[14], inout[9]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[11], inout[3]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[0], inout[8]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[12], inout[2]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[13], inout[7]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[1], inout[4]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[10], inout[5]);
	//		10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[10], inout[2]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[8], inout[4]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[7], inout[6]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[1], inout[5]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[15], inout[11]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[9], inout[14]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[3], inout[12]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[13], inout[0]);




	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	((uint8*)out)[0] = V.lo;

}


static __forceinline__ __device__ uint16 salsa_small_scalar_rnd(const uint16 &X)
{
	uint16 state = X;
	uint32_t t;

	#pragma unroll 1
	for (int i = 0; i < 10; ++i) { SALSA_CORE(state);}

	return (X + state);
}

static __device__ __forceinline__ uint16 chacha_small_parallel_rnd(const uint16 &X)
{
	uint16 st = X;
	#pragma nounroll
	for (int i = 0; i < 10; ++i) {
		CHACHA_CORE_PARALLEL_B(st);
	}
	return (X + st);
}


static __device__ __forceinline__ void neoscrypt_chacha(uint16 *XV)
{
	XV[0] ^= XV[3];
	uint16 temp;

	XV[0] = chacha_small_parallel_rnd(XV[0]); 
	XV[1] ^= XV[0];
	temp = chacha_small_parallel_rnd(XV[1]); 
	XV[2] ^= temp;
	XV[1] = chacha_small_parallel_rnd(XV[2]); 
	XV[3] ^= XV[1];
	XV[3] = chacha_small_parallel_rnd(XV[3]);
	XV[2] = temp;
}

static __device__ __forceinline__ void neoscrypt_salsa(uint16 *XV)
{

	XV[0] ^= XV[3];
	uint16 temp;

	XV[0] = salsa_small_scalar_rnd(XV[0]);
	XV[1] ^= XV[0];
	temp = salsa_small_scalar_rnd(XV[1]);
	XV[2] ^= temp;
	XV[1] = salsa_small_scalar_rnd(XV[2]);
	XV[3] ^= XV[1];
	XV[3] = salsa_small_scalar_rnd(XV[3]);
	XV[2] = temp;
}



static __forceinline__ __host__ void Blake2Shost(uint32_t * inout, const uint32_t * inkey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;



	V.hi = BLAKE2S_IV_Vechost;
	V.lo = BLAKE2S_IV_Vechost;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inkey);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inkey);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;


	V.hi = BLAKE2S_IV_Vechost;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0] = V.lo;
}

static __forceinline__ __device__ void fastkdf256_v1(int thread, const uint32_t nonce, const uint32_t * __restrict__  s_data) //, vectypeS * output)
{
	vectypeS output[8];
	uint8_t bufidx = 0;
	uchar4 bufhelper;
	uint32_t data18 = s_data[18];
	uint32_t data20 = s_data[0];
	uint32_t B[64];

	((uintx64*)(B))[0] = ((uintx64*)s_data)[0];
	((uint32_t*)B)[19] = nonce;
	((uint32_t*)B)[39] = nonce;
	((uint32_t*)B)[59] = nonce;

	uint32_t input[BLAKE2S_BLOCK_SIZE / 4]; uint32_t key[BLAKE2S_BLOCK_SIZE / 4] = { 0 };

	((uint816*)input)[0] = ((uint816*)input_init)[0];
	((uint48*)key)[0] = ((uint48*)key_init)[0];

	uint32_t qbuf, rbuf, bitbuf;

#pragma unroll  1
	for (int i = 0; i < 31; ++i)
	{

		bufhelper = ((uchar4*)input)[0];
		for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

		qbuf = bufidx / 4;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;


		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);

		//#pragma unroll
		uint32_t temp[9];

		for (int k = 0; k < 9; ++k) {
			uint32_t indice = (k + qbuf) & 63;
			temp[k] = ((uint32_t*)B)[indice];
			temp[k] ^= shifted[k];
			((uint32_t*)B)[indice] = temp[k];
		}


		uint32_t a = ((uint32_t*)s_data)[qbuf % 64], b;
		//#pragma unroll
		for (int k = 0; k < 8; k++) {
			b = s_data[(qbuf + 2 * k + 1) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + 2 * k + 2) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}


//               #pragma unroll
//				for (int k = 0; k<16; k++)
//					asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[k]) : "r"(((uint32_t*)s_data)[qbuf + k]), "r"(((uint32_t*)s_data)[qbuf + k + 1]), "r"(bitbuf));

		uint32_t noncepos = 19 - qbuf % 20;
		if (noncepos <= 16 && qbuf < 60) {
			if (noncepos != 0)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if (noncepos != 16)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}

		for (int k = 0; k < 8; k++)
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[k]) : "r"(temp[k]), "r"(temp[k + 1]), "r"(bitbuf));

		Blake2S(input, input, key); //yeah right...

	}
	bufhelper = ((uchar4*)input)[0];
	for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	for (int i = 0; i < 64; i++)
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(((uint32_t*)output)[i]) : "r"(((uint32_t*)(B))[(qbuf + i)%64]), "r"(((uint32_t*)(B))[(qbuf + i + 1)%64]), "r"(bitbuf));


	//for (int i=0;i<8;i++)
	((ulonglong4*)output)[0] ^= ((ulonglong4*)input)[0];

	((uintx64*)output)[0] ^= ((uintx64*)s_data)[0];
	((uint32_t*)output)[19] ^= nonce;
	((uint32_t*)output)[39] ^= nonce;
	((uint32_t*)output)[59] ^= nonce;


	for (int i = 0; i < 8; i++)
		(Input + 8 * thread)[i] = output[i];

}

static __forceinline__ __device__ void fastkdf256_v2(int thread, const uint32_t nonce, const  uint32_t* __restrict__ s_data) //, vectypeS * output)
{
	vectypeS output[8];
	uint8_t bufidx = 0;
	uchar4 bufhelper;
	uint32_t data18 = s_data[18];
	uint32_t data20 = s_data[0];
#define Bshift 16*thread

	uint32_t* B = (uint32_t*)&B2[Bshift];
	((uintx64*)(B))[0] = ((uintx64*)s_data)[0];


	((uint32_t*)B)[19] = nonce;
	((uint32_t*)B)[39] = nonce;
	((uint32_t*)B)[59] = nonce;
	uint32_t input[16];
	uint32_t key[16] = { 0 };

	((ulonglong4*)input)[0] = ((ulonglong4*)input_init)[0];
	((uint28*)key)[0] = ((uint28*)key_init)[0];

	uint32_t qbuf, rbuf, bitbuf;

#pragma unroll  1
	for (int i = 0; i < 31; ++i)
	{

		bufhelper = ((uchar4*)input)[0];
		for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

		qbuf = bufidx / 4;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);

		//#pragma unroll
		uint32_t temp[9];


		for (int k = 0; k < 9; ++k)
			temp[k] = __ldg(&B[(k + qbuf) & 63]);

		for (int k = 0; k < 9; ++k)
			temp[k] ^= shifted[k];



		uint32_t a = s_data[qbuf % 64], b;
		//#pragma unroll

		for (int k = 0; k < 8; k++) {
			b = s_data[(qbuf + 2 * k + 1) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + 2 * k + 2) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}


		uint32_t noncepos = 19 - qbuf % 20;
		if (noncepos <= 16 && qbuf < 60) {
			if (noncepos != 0)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if (noncepos != 16)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}


		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[0]) : "r"(temp[0]), "r"(temp[1]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[1]) : "r"(temp[1]), "r"(temp[2]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[2]) : "r"(temp[2]), "r"(temp[3]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[3]) : "r"(temp[3]), "r"(temp[4]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[4]) : "r"(temp[4]), "r"(temp[5]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[5]) : "r"(temp[5]), "r"(temp[6]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[6]) : "r"(temp[6]), "r"(temp[7]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[7]) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));


		Blake2S_v2(input, input, key);

		for (int k = 0; k < 9; k++)
			B[(k + qbuf) & 63] = temp[k];

	}

	bufhelper = ((uchar4*)input)[0];
	for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	for (int i = 0; i < 64; i++) {
		uint32_t a = (qbuf + i) & 63, b = (qbuf + i + 1) & 63;
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(((uint32_t*)output)[i]) : "r"(__ldg(&B[a])), "r"(__ldg(&B[b])), "r"(bitbuf));
	}



	output[0] ^= ((uint28*)input)[0];
	for (int i = 0; i < 8; i++) output[i] ^= ((uint28*)s_data)[i];
//	((ulonglong16 *)output)[0] ^= ((ulonglong16*)s_data)[0];
	((uint32_t*)output)[19] ^= nonce;
	((uint32_t*)output)[39] ^= nonce;
	((uint32_t*)output)[59] ^= nonce;;
	((ulonglong16 *)(Input + 8 * thread))[0] = ((ulonglong16*)output)[0];


}

static __forceinline__ __device__ void fastkdf32_v1(int thread, const  uint32_t  nonce, const uint32_t * __restrict__ salt, const uint32_t * __restrict__  s_data, uint32_t &output)
{



	uint8_t bufidx = 0;
	uchar4 bufhelper;

	uint32_t temp[9];

//	uint32_t  B0[64];
#define Bshift 16*thread

	uint32_t* B0 = (uint32_t*)&B2[Bshift];
	uint32_t cdata7 = s_data[7];
	uint32_t data18 = s_data[18];
	uint32_t data20 = s_data[0];


	((uintx64*)B0)[0] = ((uintx64*)salt)[0];
	uint32_t input[BLAKE2S_BLOCK_SIZE / 4]; uint32_t key[BLAKE2S_BLOCK_SIZE / 4] = { 0 };
	((uint816*)input)[0] = ((uint816*)s_data)[0];
	((uint48*)key)[0] = ((uint48*)salt)[0];
	uint32_t qbuf, rbuf, bitbuf;

#pragma nounroll
	for (int i = 0; i < 31; i++)
	{
#if __CUDA_ARCH__ < 500
		Blake2S(input, input, key);
#else
		Blake2S_v2(input, input, key);
#endif
		bufidx = 0;
		bufhelper = ((uchar4*)input)[0];
		for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
		qbuf = bufidx / 4;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);

		for (int k = 0; k < 9; k++) {
			temp[k] = ((uint32_t *)B0)[(k + qbuf) % 64];
		}

		((uint28*)temp)[0] ^= ((uint28*)shifted)[0];
		temp[8] ^= shifted[8];




		uint32_t a = s_data[qbuf % 64], b;
		//#pragma unroll
		for (int k = 0; k < 8; k++) {
			b = s_data[(qbuf + 2 * k + 1) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + 2 * k + 2) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}



		uint32_t noncepos = 19 - qbuf % 20;
		if (noncepos <= 16 && qbuf < 60) {
			if (noncepos != 0)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if (noncepos != 16)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[0]) : "r"(temp[0]), "r"(temp[1]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[1]) : "r"(temp[1]), "r"(temp[2]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[2]) : "r"(temp[2]), "r"(temp[3]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[3]) : "r"(temp[3]), "r"(temp[4]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[4]) : "r"(temp[4]), "r"(temp[5]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[5]) : "r"(temp[5]), "r"(temp[6]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[6]) : "r"(temp[6]), "r"(temp[7]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[7]) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));

		for (int k = 0; k < 9; k++) {
			((uint32_t *)B0)[(k + qbuf) & 63] = temp[k];
		}


	}

#if __CUDA_ARCH__ < 500
	Blake2S(input, input, key);
#else
	Blake2S_v2(input, input, key);
#endif
	bufidx = 0;
	bufhelper = ((uchar4*)input)[0];
	for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;


	for (int k = 7; k < 9; k++) {
		temp[k] = ((uint32_t *)B0)[(k + qbuf) % 64];
	}
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));
	output ^= input[7];
	output ^= cdata7;

}


static __forceinline__ __device__ void fastkdf32_v3(int thread, const  uint32_t  nonce, const uint32_t * __restrict__ salt, const uint32_t * __restrict__  s_data, uint32_t &output)
{

	uint32_t temp[9];


	uint8_t bufidx = 0;
	uchar4 bufhelper;

	//	uint32_t temp[9];

	//	uint32_t  B0[64];
#define Bshift 16*thread

	uint32_t* B0 = (uint32_t*)&B2[Bshift];
	uint32_t cdata7 = s_data[7];
	uint32_t data18 = s_data[18];
	uint32_t data20 = s_data[0];


	((uintx64*)B0)[0] = ((uintx64*)salt)[0];
//	((ulonglong4*)B0)[8] = ((ulonglong4*)salt)[0];
	uint32_t input[BLAKE2S_BLOCK_SIZE / 4]; uint32_t key[BLAKE2S_BLOCK_SIZE / 4] = { 0 };
	((uint816*)input)[0] = ((uint816*)s_data)[0];
	((uint48*)key)[0] = ((uint48*)salt)[0];
	uint32_t qbuf, rbuf, bitbuf;

#pragma nounroll
	for (int i = 0; i < 31; i++)
	{
#if __CUDA_ARCH__ < 500
		Blake2S(input, input, key);
#else
		Blake2S_v2(input, input, key);
#endif
		bufidx = 0;
		bufhelper = ((uchar4*)input)[0];
		for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
		qbuf = bufidx / 4;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R4(shifted, ((uint8*)input)[0], bitbuf);


		for (int k = 0; k < 9; k++) {
			temp[k] = __ldg(&((uint32_t*)B0)[(k + qbuf) % 64]);
		}

		((uint28*)temp)[0] ^= ((uint28*)shifted)[0];
		temp[8] ^= shifted[8];



		uint32_t a = s_data[qbuf % 64], b;
		//#pragma unroll
		for (int k = 0; k < 8; k++) {
			b = s_data[(qbuf + 2 * k + 1) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k]) : "r"(a), "r"(b), "r"(bitbuf));
			a = s_data[(qbuf + 2 * k + 2) % 64];
			asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[2 * k + 1]) : "r"(b), "r"(a), "r"(bitbuf));
		}


		uint32_t noncepos = 19 - qbuf % 20;
		if (noncepos <= 16 && qbuf < 60) {
			if (noncepos != 0)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos - 1]) : "r"(data18), "r"(nonce), "r"(bitbuf));
			if (noncepos != 16)	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(input[noncepos]) : "r"(nonce), "r"(data20), "r"(bitbuf));
		}

		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[0]) : "r"(temp[0]), "r"(temp[1]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[1]) : "r"(temp[1]), "r"(temp[2]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[2]) : "r"(temp[2]), "r"(temp[3]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[3]) : "r"(temp[3]), "r"(temp[4]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[4]) : "r"(temp[4]), "r"(temp[5]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[5]) : "r"(temp[5]), "r"(temp[6]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[6]) : "r"(temp[6]), "r"(temp[7]), "r"(bitbuf));
		asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(key[7]) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));

		for (int k = 0; k < 9; k++) {
			((uint32_t*)B0)[(k + qbuf) % 64] = temp[k];
		}

//		}
	}

#if __CUDA_ARCH__ < 500
	Blake2S(input, input, key);
#else
	Blake2S_v2(input, input, key);
#endif
	bufidx = 0;
	bufhelper = ((uchar4*)input)[0];
	for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x) { bufhelper += ((uchar4*)input)[x]; }
	bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
	qbuf = bufidx / 4;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;



	temp[7] = __ldg(&B0[(qbuf + 7) % 64]);
	temp[8] = __ldg(&B0[(qbuf + 8) % 64]);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(temp[7]), "r"(temp[8]), "r"(bitbuf));
	output ^= input[7];
	output ^= cdata7;

}




#if CUDART_VERSION >= 7000
#define SHIFT 128
#define TPB 128
#else
#define SHIFT 128
#define TPB 64
#endif
#define TPB2 128



__global__ __launch_bounds__(TPB2, 1) void neoscrypt_gpu_hash_start(int stratum, int threads, uint32_t startNonce)
{
	__shared__ uint32_t s_data[64];

	if (threadIdx.x < 64)
		s_data[threadIdx.x] = c_data[threadIdx.x];
//		for (int i = 0; i<2; i++) {
//	s_data[i+2*threadIdx.x] = c_data[i+2*threadIdx.x];

//}
	__syncthreads();
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;

	uint32_t ZNonce = (stratum) ? cuda_swab32(nonce) : nonce; //freaking morons !!!

#if __CUDA_ARCH__ < 500
	fastkdf256_v1(thread, ZNonce, s_data);
#else
	fastkdf256_v2(thread, ZNonce, s_data);
#endif

}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_chacha1_stream1(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	int shift = SHIFT * 8 * thread;
	unsigned int shiftTr = 8 * thread;


	vectypeS X[8];
	for (int i = 0; i < 8; i++)
		X[i] = __ldg4(&(Input + shiftTr)[i]);


#pragma nounroll
	for (int i = 0; i < 128; ++i)
	{
		uint32_t offset = shift + i * 8;
		for (int j = 0; j < 8; j++)
			(W + offset)[j] = X[j];
		neoscrypt_chacha((uint16*)X);

	}
	for (int i = 0; i < 8; i++)
		(Tr + shiftTr)[i] = X[i];

}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_chacha2_stream1(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	int shift = SHIFT * 8 * thread;
	int shiftTr = 8 * thread;

	vectypeS X[8];
	for (int i = 0; i < 8; i++)
		X[i] = __ldg4(&(Tr + shiftTr)[i]);

#pragma nounroll
	for (int t = 0; t < 128; t++)
	{
		int idx = (X[6].x.x & 0x7F) << 3;

		for (int j = 0; j < 8; j++)
			X[j] ^= __ldg4(&(W + shift + idx)[j]);
		neoscrypt_chacha((uint16*)X);

	}

	for (int i = 0; i < 8; i++)
		(Tr + shiftTr)[i] = X[i];  // best checked

}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_salsa1_stream1_orig(int threads, uint32_t startNonce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	int shift = SHIFT * 8 * thread;
	int shiftTr = 8 * thread;

	vectypeS Z[8];

	#pragma unroll
	for (int i = 0; i < 8; i++)
		Z[i] = __ldg4(&(Input + shiftTr)[i]);

// #pragma nounroll
	#pragma unroll
	for (int i = 0; i < 128; ++i)
	{
		for (int j = 0; j < 8; j++)
			(W2 + shift + i * 8)[j] = Z[j];
		neoscrypt_salsa((uint16*)Z);
	}

	#pragma unroll
	for (int i = 0; i < 8; i++)
		(Tr2 + shiftTr)[i] = Z[i];
}

static __device__ __inline__ void __copy16(uint4 *dest, const uint4 *src)
{
	// uint4 a = {1,2,3,4};
	// uint4 b;

	// asm("ld.local.cs.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(&(a.x)));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(&(a.x)));

	// asm volatile ("{\n\t"
	// 	".reg .u32 a,b,c,d; \n\t"
	// 	"ld.global.nc.v4.u32 {a,b,c,d}, [%1]; \n\t"
	// 	"mov.u32 [%0], a; \n\t"
	// 	"mov.u32 [%0+1], b; \n\t"
	// 	"mov.u32 [%0+2], c; \n\t"
	// 	"mov.u32 [%0+3], d; \n\t"
	// 	"}"
	// 	: "=l"((uint*)dest) : "l"(src));
	// uint* sptr = (uint*)src;
	// uint* dptr = (uint*)dest;
	uint* sptr = (uint*)src;
	uint64_t* dptr = (uint64_t*)dest;

	// asm volatile ("{\n\t"
	// 	".reg .u64 a,b; \n\t"
	// 	"ld.global.nc.v2.u64 {a,b}, [%1]; \n\t"
	// 	"st.local.cs.v2.u64 [%0], {a,b}; \n\t"
	// 	"}"
	// : "=l"(dptr) : "l"(sptr) );

	asm("ld.global.nc.v2.u64 {%0,%1}, [%2];"  : "=l"(dptr[0]), "=l"(dptr[1]) : "l"(src+0));
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2];"  : "=l"(dptr[2]), "=l"(dptr[3]) : "l"(src+1));

	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[0].x), "=r"(dest[0].y), "=r"(dest[0].z), "=r"(dest[0].w) : "l"(src+0));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[1].x), "=r"(dest[1].y), "=r"(dest[1].z), "=r"(dest[1].w) : "l"(src+1));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[2].x), "=r"(dest[2].y), "=r"(dest[2].z), "=r"(dest[2].w) : "l"(src+2));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[3].x), "=r"(dest[3].y), "=r"(dest[3].z), "=r"(dest[3].w) : "l"(src+3));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[4].x), "=r"(dest[4].y), "=r"(dest[4].z), "=r"(dest[4].w) : "l"(src+4));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[5].x), "=r"(dest[5].y), "=r"(dest[5].z), "=r"(dest[5].w) : "l"(src+5));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[6].x), "=r"(dest[6].y), "=r"(dest[6].z), "=r"(dest[6].w) : "l"(src+6));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[7].x), "=r"(dest[7].y), "=r"(dest[7].z), "=r"(dest[7].w) : "l"(src+7));

	dest+=2;
	src+=2;

	for(int i=2;i<16;++i)
		(*dest++) = (*src++);

	// dest[0] = src[0];
	// dest[1] = src[1];
	// dest[2] = src[2];
	// dest[3] = src[3];
	// dest[4] = src[4];
	// dest[5] = src[5];
	// dest[6] = src[6];
	// dest[7] = src[7];

	// uint28 ret;
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[0].x.x), "=r"(dest[0].x.y), "=r"(dest[0].y.x), "=r"(dest[0].y.y) : __LDG_PTR(src));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[0].z.x), "=r"(dest[0].z.y), "=r"(dest[0].w.x), "=r"(dest[0].w.y) : __LDG_PTR(src));
	// dest[0] = ret;

	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[1].x.x), "=r"(dest[1].x.y), "=r"(dest[1].y.x), "=r"(dest[1].y.y) : __LDG_PTR(src+1));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[1].z.x), "=r"(dest[1].z.y), "=r"(dest[1].w.x), "=r"(dest[1].w.y) : __LDG_PTR(src+1));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[2].x.x), "=r"(dest[2].x.y), "=r"(dest[2].y.x), "=r"(dest[2].y.y) : __LDG_PTR(src+2));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[2].z.x), "=r"(dest[2].z.y), "=r"(dest[2].w.x), "=r"(dest[2].w.y) : __LDG_PTR(src+2));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[3].x.x), "=r"(dest[3].x.y), "=r"(dest[3].y.x), "=r"(dest[3].y.y) : __LDG_PTR(src+3));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[3].z.x), "=r"(dest[3].z.y), "=r"(dest[3].w.x), "=r"(dest[3].w.y) : __LDG_PTR(src+3));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[4].x.x), "=r"(dest[4].x.y), "=r"(dest[4].y.x), "=r"(dest[4].y.y) : __LDG_PTR(src+4));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[4].z.x), "=r"(dest[4].z.y), "=r"(dest[4].w.x), "=r"(dest[4].w.y) : __LDG_PTR(src+4));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[5].x.x), "=r"(dest[5].x.y), "=r"(dest[5].y.x), "=r"(dest[5].y.y) : __LDG_PTR(src+5));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[5].z.x), "=r"(dest[5].z.y), "=r"(dest[5].w.x), "=r"(dest[5].w.y) : __LDG_PTR(src+5));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[6].x.x), "=r"(dest[6].x.y), "=r"(dest[6].y.x), "=r"(dest[6].y.y) : __LDG_PTR(src+6));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[6].z.x), "=r"(dest[6].z.y), "=r"(dest[6].w.x), "=r"(dest[6].w.y) : __LDG_PTR(src+6));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(dest[7].x.x), "=r"(dest[7].x.y), "=r"(dest[7].y.x), "=r"(dest[7].y.y) : __LDG_PTR(src+7));
	// asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(dest[7].z.x), "=r"(dest[7].z.y), "=r"(dest[7].w.x), "=r"(dest[7].w.y) : __LDG_PTR(src+7));

// 	return ret;
}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_salsa1_stream1(int threads, uint32_t startNonce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	// Now we try to process 2 of those blocks at the same time:
	int shift = SHIFT * 8 * thread;
	int shiftTr = 8 * thread;

	uint4 Z[16];

	__copy16((uint4*)Z,(uint4*)(Input+shiftTr));

	// #pragma unroll
	// for (int i = 0; i < 8; i++)
	// 	Z[i] = (Input + shiftTr)[i];

// #pragma nounroll
	uint4* ptr = (uint4*)(W2 + shift);

	#pragma unroll
	for (int i = 0; i < 128; ++i)
	{
		#pragma unroll
		for (int j = 0; j < 16; j++)
			(*ptr++) = Z[j];
		// __copy16(ptr,Z);
		// ptr += 8;

		neoscrypt_salsa((uint16*)Z);
	}

	// __copy16((uint4*)(Tr2+shiftTr),(uint4*)Z);

	ptr = (uint4*)(Tr2 + shiftTr);
	#pragma unroll
	for (int i = 0; i < 16; i++)
		(*ptr++) = Z[i];
}

#define BSIZE 32

// __launch_bounds__(TPB, 1)

__global__  void neoscrypt_gpu_hash_salsa1_stream1_opt(int threads, uint32_t startNonce, unsigned long long* time)
{
	// unsigned long long startTime = clock();

	int ioffset = BSIZE * 64 * blockIdx.x;
	int woffset = BSIZE * SHIFT * 64 * blockIdx.x;

	int x = threadIdx.x;

	// Input is provided as uint28 pointer, each uint28 is 2*uint4 = 8 uint
	// so if the stride is 8 uint28, then it is 8x8=64 uint
	// Same result for the shift: if it is 8 uint28, then this is 64 uint:

	// int shiftTr = 64 * offset;
	// int shift = SHIFT * 64 * offset;
	
	uint* iPtr = ((uint*)Input)+ioffset;

	// Prepare the buffer containing all the input rows:
	// Z rows contain 8 uint28, and thus 64 uint, to avoid memory bank conflits
	// We add 1 to this size:
	__shared__ uint Z[BSIZE][64+1];

	// Fill the input array:
	for(int j=0;j<BSIZE;++j)
	{
		Z[j][x] = iPtr[j*64 + x];
		Z[j][32+x] = iPtr[j*64 + 32 + x];
	}

	// Need to synchronize the threads:
	// __syncthreads();

	// #pragma nounroll
	uint* dPtr = ((uint*)W2) + woffset;

	#pragma unroll
	for (int i = 0; i < 128; ++i)
	{
		for(int j=0;j<BSIZE;++j)
		{
			dPtr[i*64 + j*SHIFT*64 + x] = Z[j][x];
			dPtr[i*64 + j*SHIFT*64 + 32 + x] = Z[j][32 + x];
		}
		// __syncthreads();

		// #pragma unroll
		// for (int j = 0; j < 16; j++)
		// 	(*ptr++) = Z[j];
		// __copy16(ptr,Z);
		// ptr += 8;

		neoscrypt_salsa((uint16*)Z[x]);
	}

	// Copy the final data in to the Tr2 buffer:
	dPtr = ((uint*)Tr2)+ioffset;
	for(int j=0;j<BSIZE;++j)
	{
		dPtr[j*64 + x] = Z[j][x];
		dPtr[j*64 + 32 + x] = Z[j][32 + x];
	}

	// No need to sync the threads here: we are done.
	// __syncthreads();

	// unsigned long long endTime = clock();
	// *time = (endTime - startTime);
}

__global__ __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_salsa2_stream1(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	int shift = SHIFT * 8 * thread;
	int shiftTr = 8 * thread;

	vectypeS X[8];
	for (int i = 0; i < 8; i++)
		X[i] = __ldg4(&(Tr2 + shiftTr)[i]);

#pragma nounroll
	for (int t = 0; t < 128; t++)
	{
		int idx = (X[6].x.x & 0x7F) << 3;

		for (int j = 0; j < 8; j++)
			X[j] ^= __ldg4(&(W2 + shift + idx)[j]);
		neoscrypt_salsa((uint16*)X);

	}
	for (int i = 0; i < 8; i++)
		(Tr2 + shiftTr)[i] = X[i];  // best checked

}

__global__  __launch_bounds__(TPB, 1) void neoscrypt_gpu_hash_salsa1_stream1_merge(int threads, uint32_t startNonce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	int shift = SHIFT * 8 * thread;
	int shiftTr = 8 * thread;
	int x = threadIdx.x;

	vectypeS Z[8];

	#pragma unroll
	for (int i = 0; i < 8; i++)
		Z[i] = __ldg4(&(Input + shiftTr)[i]);

// #pragma nounroll
	#pragma unroll
	for (int i = 0; i < 128; ++i)
	{
		for (int j = 0; j < 8; j++)
			(W2 + shift + i * 8)[j] = Z[j];
		neoscrypt_salsa((uint16*)Z);
	}

	#pragma unroll
	for (int t = 0; t < 128; t++)
	{
		int idx = (Z[6].x.x & 0x7F) << 3;

		for (int j = 0; j < 8; j++)
			Z[j] ^= __ldg4(&(W2 + shift + idx)[j]);
		neoscrypt_salsa((uint16*)Z);
	}

	for (int i = 0; i < 8; i++)
		(Tr2 + shiftTr)[i] = Z[i];  // best checked
}



__global__ __launch_bounds__(TPB2, 1) void neoscrypt_gpu_hash_ending(int stratum, int threads, uint32_t startNonce, uint32_t *nonceVector)
{
	__shared__ uint32_t s_data[64];
	/*
		if (threadIdx.x<40)
			for (int i = 0; i<2; i++)
	       s_data[i + 2 * threadIdx.x] = c_data[i + 2 * threadIdx.x];
	*/
	if (threadIdx.x < 64)
		s_data[threadIdx.x] = c_data[threadIdx.x];
	__syncthreads();
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;

	int shiftTr = 8 * thread;
	vectypeS Z[8];
	uint32_t outbuf;

	uint32_t ZNonce = (stratum) ? cuda_swab32(nonce) : nonce;

//		for (int i = 0; i<8; i++)
//		Z[i] = __ldg4(&(Tr + shiftTr)[i]);
	for (int i = 0; i < 8; i++)
		Z[i] = __ldg4(&(Tr2 + shiftTr)[i]) ^ __ldg4(&(Tr + shiftTr)[i]);
#if __CUDA_ARCH__ < 500
	fastkdf32_v1(thread, ZNonce, (uint32_t*)Z, s_data, outbuf);
#else
	fastkdf32_v3(thread, ZNonce, (uint32_t*)Z, s_data, outbuf);
#endif
	if (outbuf <= pTarget[7]) {
		uint32_t tmp = atomicExch(&nonceVector[0], nonce);
	}
}


void neoscrypt_cpu_init_2stream(int thr_id, int threads, uint32_t *hash, uint32_t *hash2, uint32_t *Trans1, uint32_t *Trans2, uint32_t *Trans3, uint32_t *Bhash)
{
	cudaMemcpyToSymbol(B2, &Bhash, sizeof(Bhash), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(W, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(W2, &hash2, sizeof(hash2), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Tr, &Trans1, sizeof(Trans1), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Tr2, &Trans2, sizeof(Trans2), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Input, &Trans3, sizeof(Trans3), 0, cudaMemcpyHostToDevice);

	cudaMalloc(&d_NNonce[thr_id], sizeof(uint32_t));
	cudaMalloc(&d_time[thr_id], sizeof(unsigned long long));

	// Create the streams:
	cudaStreamCreate(&g_stream[thr_id*2]);
	cudaStreamCreate(&g_stream[thr_id*2+1]);

}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__host__ uint32_t neoscrypt_cpu_hash_k4_2stream(int stratum, int thr_id, int threads, uint32_t startNounce, int order, unsigned long long &tres)
{
	uint32_t result[MAX_GPUS] = { 0xffffffff };
	cudaMemset(d_NNonce[thr_id], 0xffffffff, sizeof(uint32_t));


	const int threadsperblock = TPB;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	const int threadsperblock2 = TPB2;
	dim3 grid2((threads + threadsperblock2 - 1) / threadsperblock2);
	dim3 block2(threadsperblock2);

	const int threadsperblock3 = BSIZE;
	dim3 grid3((threads + threadsperblock3 - 1) / threadsperblock3);
	dim3 block3(threadsperblock3);

	//	neoscrypt_gpu_hash_orig << <grid, block >> >(threads, startNounce, d_NNonce[thr_id]);

	neoscrypt_gpu_hash_start << <grid2, block2, 0, g_stream[thr_id*2] >> >(stratum, threads, startNounce); //fastkdf

	cudaDeviceSynchronize();

	neoscrypt_gpu_hash_chacha1_stream1 << <grid, block, 0, g_stream[thr_id*2] >> >(threads, startNounce); //salsa
	gpuErrchk( cudaPeekAtLastError() );
	neoscrypt_gpu_hash_chacha2_stream1 << <grid, block, 0, g_stream[thr_id*2] >> >(threads, startNounce); //salsa
	gpuErrchk( cudaPeekAtLastError() );

	// neoscrypt_gpu_hash_salsa1_stream1_merge << <grid, block, 0, g_stream[thr_id*2+1] >> >(threads, startNounce); //chacha
	// neoscrypt_gpu_hash_salsa1_stream1 << <grid, block, 0, g_stream[thr_id*2+1] >> >(threads, startNounce); //chacha
	neoscrypt_gpu_hash_salsa1_stream1_orig << <grid, block, 0, g_stream[thr_id*2+1] >> >(threads, startNounce); //chacha
	gpuErrchk( cudaPeekAtLastError() );
	// neoscrypt_gpu_hash_salsa1_stream1_opt << <grid3, block3, 0, g_stream[thr_id*2+1] >> >(threads, startNounce, d_time[thr_id]); //chacha
	neoscrypt_gpu_hash_salsa2_stream1 << <grid, block, 0, g_stream[thr_id*2+1] >> >(threads, startNounce); //chacha
	gpuErrchk( cudaPeekAtLastError() );

	cudaDeviceSynchronize();
	// cudaStreamDestroy(g_stream[thr_id*2+1]); //will do the synchronization
	neoscrypt_gpu_hash_ending << <grid2, block2, 0, g_stream[thr_id*2] >> >(stratum, threads, startNounce, d_NNonce[thr_id]); //fastkdf+end


	MyStreamSynchronize(NULL, order, thr_id);
	cudaMemcpy(&result[thr_id], d_NNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaMemcpy(&tres, d_time[thr_id], sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	// cudaStreamDestroy(g_stream[thr_id*2]);


	return result[thr_id];
}

__host__ void neoscrypt_setBlockTarget(uint32_t* pdata, const void *target)
{

	unsigned int PaddedMessage[80]; //brings balance to the force
	uint32_t input[16], key[16] = {0};

	for (int i = 0; i < 20; i++) PaddedMessage[i     ] = pdata[i];
	for (int i = 0; i < 20; i++) PaddedMessage[i + 20] = pdata[i];
	for (int i = 0; i < 20; i++) PaddedMessage[i + 40] = pdata[i];
	for (int i = 0; i < 4; i++)  PaddedMessage[i + 60] = pdata[i];
	for (int i = 0; i < 16; i++) PaddedMessage[i + 64] = pdata[i];
	PaddedMessage[19] = 0;
	PaddedMessage[39] = 0;
	PaddedMessage[59] = 0;

	((uint16*)input)[0] = ((uint16*)pdata)[0];
	((uint8*)key)[0] = ((uint8*)pdata)[0];
//		for (int i = 0; i<10; i++) { printf(" pdata/input %d %08x %08x \n",i,pdata[2*i],pdata[2*i+1]); }


	Blake2Shost(input, key);


	cudaMemcpyToSymbol(pTarget, target, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(input_init, input, 16 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(key_init, key, 16 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_data, PaddedMessage, 40 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
//		cudaMemcpyToSymbol(c_data2, PaddedMessage, 40 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

