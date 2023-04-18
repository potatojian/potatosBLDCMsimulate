/* header -------------------------------------------------------------------*/
/**
  * 文件名称：BLDCMotorModel.h
  * 日    期：2022/09/02
  * 作    者：mrpotato
  * 简    述：无控制无刷电机仿真模块+输出结果滤波器声明
  */ 
#ifndef _BLDCMotorModel_h
#define _BLDCMotorModel_h

/* includes -----------------------------------------------------------------*/
#include "Basic.h"

/* typedef ------------------------------------------------------------------*/

/* define -------------------------------------------------------------------*/
#define REVERSE

#define TICK_STEP 0.00000001
#define I_MIN 0.000001
#define POLES 5.0

/* variables ----------------------------------------------------------------*/

/* function prototypes ------------------------------------------------------*/
extern void BLDCMotorModel(unsigned char threadIndex, unsigned long RLIter,
        unsigned int *tjtryDataIndex, _data_line *DataImputBlock,
		_motor_line *MotorZeroState, _motor_line *MotorSimuBlock,
		_RL_trajectory *trajectories);

#endif
/* end of file --------------------------------------------------------------*/
