/* header -------------------------------------------------------------------*/
/**
  * 文件名称：Basic.h
  * 日    期：2022/09/07
  * 作    者：mrpotato
  * 简    述：工程内所有文件都要添加的声明
  */ 
#ifndef _Basic_h
#define _Basic_h

/* includes -----------------------------------------------------------------*/
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/file.h>
#include "oneapi/dnnl/dnnl.h"
#include "dnnl_debug.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.h"
#endif

/* define -------------------------------------------------------------------*/
/* Inputs&Outputs */
#define DATA_LENGTH 2000 // analogDataLength
#define PROCESS_WIDTH 8 // processWidth >= 8
//#define BLDCM_SIMU_OUTPUT
#define RL_NET_OUTPUT

/* ReinforcementLearning */
#define MAX_ITER 1

/* typedef ------------------------------------------------------------------*/
typedef struct
{
    double T;
    bool Da;
    bool Dd;
    double I1;
    double I2;
    double I3;
    double V1;
    double V2;
    double V3;
    bool H1;
    bool H2;
    bool H3;
    bool L1;
    bool L2;
    bool L3;
    double Itot;
    double Vzero;
} _data_line;

typedef struct
{
    double Time;
    double I1;
    double I2;
    double I3;
    double Itot;
    double ia;
    double ib;
    double ic;
    double V1;
    double V2;
    double V3;
    double Vzero;
    double H1;
    double H2;
    double H3;
    double L1;
    double L2;
    double L3;
    double theta;
    double etheta;
    double omega;
} _motor_line;

typedef struct
{
    float V;
    double La;
    double Lb;
    double Lc;
    double Mab;
    double Mbc;
    double Mca;
    double Ra;
    double Rb;
    double Rc;
    double K;
    double Tl;
    double J;
    double C;
    float r;
} _RL_trajectory;

typedef struct
{
    dnnl_engine_t engine;
    dnnl_stream_t stream;
    // CNN Forward
	float *conv1_src;
	float *conv1_weights;
	float *conv1_bias;
	float *pool1_dst;
	float *conv2_weights;
	float *conv2_bias;
	float *pool2_dst;
	float *conv3_weights;
	float *conv3_bias;
	float *pool3_dst;
    // ACTOR Forward
	float *dnseA1_src;
	float *dnseA1_weights;
	float *dnseA1_bias;
	float *dnseA2_weights;
	float *dnseA2_bias;
	float *dnseA3_weights;
	float *dnseA3_bias;
	float *dnseA4_weights;
	float *dnseA4_bias;
	float *dnseA5_weights;
	float *dnseA5_bias;
    // CRITIC Forward
	float *dnseC1_src;
	float *dnseC1_weights;
	float *dnseC1_bias;
	float *dnseC2_weights;
	float *dnseC2_bias;
	float *dnseC3_weights;
	float *dnseC3_bias;
	float *dnseC4_weights;
	float *dnseC4_bias;
	float *dnseC5_weights;
	float *dnseC5_bias;
    // ACTOR Backward
    float *dnseA5_diff_dst;
    float *dnseA5_diff_bias;
    float *dnseA5_diff_weights;
    float *dnseA4_diff_bias;
    float *dnseA4_diff_weights;
    float *dnseA3_diff_bias;
    float *dnseA3_diff_weights;
    float *dnseA2_diff_bias;
    float *dnseA2_diff_weights;
    float *dnseA1_diff_bias;
    float *dnseA1_diff_weights;
    // CRITIC Backward
    float *dnseC5_diff_dst;
    float *dnseC5_diff_bias;
    float *dnseC5_diff_weights;
    float *dnseC4_diff_bias;
    float *dnseC4_diff_weights;
    float *dnseC3_diff_bias;
    float *dnseC3_diff_weights;
    float *dnseC2_diff_bias;
    float *dnseC2_diff_weights;
    float *dnseC1_diff_bias;
    float *dnseC1_diff_weights;
    // CNN Backward
    float *pool3_diff_dst;
    float *conv3_diff_bias;
    float *conv3_diff_weights;
    float *pool2_diff_dst;
    float *conv2_diff_bias;
    float *conv2_diff_weights;
    float *pool1_diff_dst;
    float *conv1_diff_bias;
    float *conv1_diff_weights;
} _RL_worker;

typedef struct
{
	unsigned char threadIndex;
	_data_line DataImputBlock[1024];
	_motor_line MotorZeroState;
	_motor_line MotorSimuBlock[PROCESS_WIDTH];
	_RL_trajectory trajectories[DATA_LENGTH];
	_RL_worker Worker;
} _thread_args;

/* variables ----------------------------------------------------------------*/

/* function prototypes ------------------------------------------------------*/
extern bool appendLoad(FILE *fp1, _data_line *DataImputBlock);
extern bool itobool(int input);

#endif
/* end of file --------------------------------------------------------------*/

