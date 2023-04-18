/* header -------------------------------------------------------------------*/
/**
  * 文件名称：BLDCMotorModel.c
  * 日    期：2022/08/31
  * 作    者：mrpotato
  * 简    述：无控制仿真模块+输出结果滤波器
  */ 

/* includes -----------------------------------------------------------------*/
#include "Basic.h"
#include "BLDCMotorModel.h"

/* typedef ------------------------------------------------------------------*/

/* define -------------------------------------------------------------------*/

/* variables ----------------------------------------------------------------*/

/* function prototypes ------------------------------------------------------*/
void BLDCMotorModel(unsigned char threadIndex, unsigned long RLIter,
        unsigned int *tjtryDataIndex, _data_line *DataImputBlock,
		_motor_line *MotorZeroState, _motor_line *MotorSimuBlock,
		_RL_trajectory *trajectories);

/* user code ----------------------------------------------------------------*/
/* user code begin 3*/
/**
  * 函数功能：按照模型运行数据
  * 输入参数：无
  * 返 回 值：无
  * 相关变量：
  * 说    明：
  */
void BLDCMotorModel(unsigned char threadIndex, unsigned long RLIter,
        unsigned int *tjtryDataIndex, _data_line *DataImputBlock,
		_motor_line *MotorZeroState, _motor_line *MotorSimuBlock,
		_RL_trajectory *trajectories)
{
/*--------------------仿真流程输出文档---------------------------------------*/
#ifdef BLDCM_SIMU_OUTPUT
	char outputMotorFile[128];
    sprintf(outputMotorFile, "BLDC_Worker%d_Iter%d_Line%d.tmp", threadIndex,
			RLIter, *tjtryDataIndex);
    FILE *fp2 = NULL;
    fp2 = fopen(outputMotorFile, "w");
    if(fp2 == NULL){printf("error opening outputFile!\r\n");}
#endif

/*--------------------初始化电动机数据---------------------------------------*/
    /* simulate */
    //double endTime = 1.0;// S 从第0.0 S开始，模拟运行至第endTime S
    //double t_step = 0.00000001;// S 每一步的时间(0.00000001s->10ns, 100MHz)
    double simu_time = MotorZeroState -> Time;// S 当前时间
    bool next_t = 1;

    /* BLDCmotor */
    // u->di + backEMF
    double ua = 0.0, ub = 0.0, uc = 0.0, um = 0.0;// V 如A相相电压为(ua - um)
    double ia = MotorZeroState -> ia,
            ib = MotorZeroState -> ib,
            ic = MotorZeroState -> ic;// A
	//double imin;//A  最小忽略电流
	//double dia, dib, dic;
    // |La  Mab Mac|  inv  |Lk11 Lk12 Lk13|相电压磁链逆矩阵
    // |Mab Lb  Mbc|------>|Lk21 Lk22 Lk23|
    // |Mac Mbc Lc |       |Lk31 Lk32 Lk33|
    // Lk_den = Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc
    // Lk11 = Lk_num1/Lk_den = (Mbc^2 - Lb*Lc)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    // Lk12 = Lk_num4/Lk_den = (Lc*Mab - Mac*Mbc)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    // Lk13 = Lk_num5/Lk_den = (Lb*Mac - Mab*Mbc)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    // Lk21 = Lk_num4/Lk_den = (Lc*Mab - Mac*Mbc)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    // Lk22 = Lk_num2/Lk_den = (Mac^2 - La*Lc)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    // Lk31 = Lk_num5/Lk_den = (Lb*Mac - Mab*Mbc)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    // Lk32 = Lk_num6/Lk_den = (La*Mbc - Mab*Mac)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    // Lk33 = Lk_num3/Lk_den = (Mab^2 - La*Lb)/(Lc*Mab^2 - 2*Mab*Mac*Mbc + Lb*Mac^2 + La*Mbc^2 - La*Lb*Lc)
    /**************************************/
    // |La-Mab  Mab-Lb  Mac-Mbc|  inv  |Lp11 Lp12 Lp13|线电压磁链逆矩阵
    // |Mab-Mac Lb-Mbc  Mbc-Lc |------>|Lp21 Lp22 Lp23|
    // |Mac-La  Mbc-Mab Lc-Mac |       |Lp31 Lp32 Lp33|abandoned plan
    double La = trajectories[*tjtryDataIndex].La,
            Lb = trajectories[*tjtryDataIndex].Lb,
            Lc = trajectories[*tjtryDataIndex].Lc;// H <----------------
	double Mab = trajectories[*tjtryDataIndex].Mab,
            Mbc = trajectories[*tjtryDataIndex].Mbc,
            Mca = trajectories[*tjtryDataIndex].Mca;// H <--------------
	double Lk_den = Lc*Mab*Mab + Lb*Mca*Mca + La*Mbc*Mbc - 2*Mab*Mca*Mbc
            - La*Lb*Lc;
	double Lk_num1 = Mbc*Mbc - Lb*Lc, Lk_num2 = Mca*Mca - La*Lc,
            Lk_num3 = Mab*Mab - La*Lb, Lk_num4 = Lc*Mab - Mca*Mbc,
            Lk_num5 = Lb*Mca - Mab*Mbc, Lk_num6 = La*Mbc - Mab*Mca;
	//double Lk11, Lk12, Lk13, Lk21, Lk22, Lk23, Lk31, Lk32, Lk33;
	//double Lp11, Lp12, Lp13, Lp21, Lp22, Lp23, Lp31, Lp32, Lp33;
	double Ea = 0.0, Eb = 0.0, Ec = 0.0;// V
    double Ra = trajectories[*tjtryDataIndex].Ra,
            Rb = trajectories[*tjtryDataIndex].Rb,
            Rc = trajectories[*tjtryDataIndex].Rc;// ohm <--------------
	//double poles;// 对(pairs) 极对数是整数
	double K = trajectories[*tjtryDataIndex].K;// V/(rad/s) <-----------
	double Kt = trajectories[*tjtryDataIndex].K;// N*m/A (Kv*Kt = 1;
                                                       // Kt = K)
	double omega = MotorZeroState -> omega;// rad/s
	//double domega;
	double theta = MotorZeroState -> theta;// rad <------------------
	double etheta = 0.0;// rad
	double Te = 0.0;// N*m

	double Tl = trajectories[*tjtryDataIndex].Tl;// N*m <---------------
	double J = trajectories[*tjtryDataIndex].J;// kg*m^2 <--------------
	double C = trajectories[*tjtryDataIndex].C;// N*m*s/rad <-----------

    /* driver */
    //gate->u
    bool gAH = MotorZeroState -> H1,
            gAL = MotorZeroState -> L1,
            gBH = MotorZeroState -> H2,
            gBL = MotorZeroState -> L2,
            gCH = MotorZeroState -> H3,
            gCL = MotorZeroState -> L3;
    double Vcc = 2*MotorZeroState -> Vzero;// V

    /* filter */
    double f_ia = 0.0, f_ib = 0.0, f_ic = 0.0;
    double fc_ia = 0.0, fc_ib = 0.0, fc_ic = 0.0;
    unsigned long f_stepCount = 0;

    /* countIndex */
    unsigned int countData = 0;
    unsigned int countNum = 0;

/*--------------------更新反电动势-------------------------------------------*/
    // 计算电角度
    etheta = fmod(theta*POLES, 2*M_PI);
    // 计算反电动势
#ifdef REVERSE
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
        Ec = -K*omega;
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
        Ec = K*omega;
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#else
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
        Ec = -K*omega;
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
        Ec = K*omega;
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#endif
    // 调试测试输出
//  printf("BackEMF: %lf, %lf, %lf\r\n", Ea, Eb, Ec);

/*--------------------更新输入电压-------------------------------------------*/
    /* 导通状态 */
    // 计算A相桥输出电压
    if((gAH == 1) && (gAL == 0)){ua = Vcc;}
    if((gAH == 0) && (gAL == 1)){ua = 0.0;}
    if((gAH == 1) && (gAL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算B相桥输出电压
    if((gBH == 1) && (gBL == 0)){ub = Vcc;}
    if((gBH == 0) && (gBL == 1)){ub = 0.0;}
    if((gBH == 1) && (gBL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算C相桥输出电压
    if((gCH == 1) && (gCL == 0)){uc = Vcc;}
    if((gCH == 0) && (gCL == 1)){uc = 0.0;}
    if((gCH == 1) && (gCL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    /* 截止状态 */
    // 计算A相桥输出电压
    if((gAH == 0) && (gAL == 0))
    {
        if(ia > I_MIN){ua = 0.0;}
        else if(ia < -I_MIN){ua = Vcc;}
        else
        {
            ua = -(Lk_num4*(Eb - ub + Rb*ib - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num5*(Ec - uc + Rc*ic - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num1*(Ea - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num1*((Lk_num1 + Lk_num4 + Lk_num5)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num5*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ua > Vcc){ua = Vcc;}
            if(ua < 0){ua = 0;}
        }
    }
    // 计算B相桥输出电压
    if((gBH == 0) && (gBL == 0))
    {
        if(ib > I_MIN){ub = 0.0;}
        else if(ib < -I_MIN){ub = Vcc;}
        else
        {
            ub = -(Lk_num4*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Ec - uc + Rc*ic - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num2*(Eb - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num2*((Lk_num2 + Lk_num4 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ub > Vcc){ub = Vcc;}
            if(ub < 0){ub = 0;}
        }
    }
    // 计算C相桥输出电压
    if((gCH == 0) && (gCL == 0))
    {
        if(ic > I_MIN){uc = 0.0;}
        else if(ic < -I_MIN){uc = Vcc;}
        else
        {
            uc = -(Lk_num5*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Eb - ub + Rb*ib - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num3*(Ec - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num3*((Lk_num3 + Lk_num5 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num5*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(uc > Vcc){uc = Vcc;}
            if(uc < 0){uc = 0;}
        }
    }
    /* 中性点电压 */
    um = -((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6);
//  printf("u:%lf, %lf, %lf\r\n", ua, ub, uc);
//  printf("%lf\r\n", um);

/*--------------------模拟运行循环-------------------------------------------*/
    while(next_t)// 模拟运行循环
    {
    /*--------------------处理数据-------------------------------------------*/
		if((simu_time >= (DataImputBlock[countNum].T - 0.5*TICK_STEP))
            &&(simu_time < (DataImputBlock[countNum].T + 0.5*TICK_STEP)))
		{
			if(DataImputBlock[countNum].Da)
			{
				Vcc = 2*DataImputBlock[countNum].Vzero;
                // filter
				if(simu_time != 0.0)
				{
					f_ia = fc_ia/f_stepCount; f_ib = fc_ib/f_stepCount;
                            f_ic = fc_ic/f_stepCount;
					fc_ia = 0.0; fc_ib = 0.0; fc_ic = 0.0;
					f_stepCount = 0;
				}
                // result output
                MotorSimuBlock[countData].Time = simu_time;
                MotorSimuBlock[countData].I1
                        = DataImputBlock[countNum].I1;
                MotorSimuBlock[countData].I2
                        = DataImputBlock[countNum].I2;
                MotorSimuBlock[countData].I3
                        = DataImputBlock[countNum].I3;
                MotorSimuBlock[countData].Itot
                        = DataImputBlock[countNum].Itot;
                MotorSimuBlock[countData].ia = ia;
                MotorSimuBlock[countData].ib = ib;
                MotorSimuBlock[countData].ic = ic;
                MotorSimuBlock[countData].V1
                        = DataImputBlock[countNum].V1;
                MotorSimuBlock[countData].V2
                        = DataImputBlock[countNum].V2;
                MotorSimuBlock[countData].V3
                        = DataImputBlock[countNum].V3;
                MotorSimuBlock[countData].Vzero
                        = DataImputBlock[countNum].Vzero;
                MotorSimuBlock[countData].H1 = (double)gAH;
                MotorSimuBlock[countData].H2 = (double)gBH;
                MotorSimuBlock[countData].H3 = (double)gCH;
                MotorSimuBlock[countData].L1 = (double)gAL;
                MotorSimuBlock[countData].L2 = (double)gBL;
                MotorSimuBlock[countData].L3 = (double)gCL;
                MotorSimuBlock[countData].theta = theta;
                MotorSimuBlock[countData].etheta = etheta;
                MotorSimuBlock[countData].omega = omega;

                countData ++;
			}
			if(DataImputBlock[countNum].Dd)
			{
				gAH = DataImputBlock[countNum].H1;
				gAL = DataImputBlock[countNum].L1;
			    gBH = DataImputBlock[countNum].H2;
                gBL = DataImputBlock[countNum].L2;
                gCH = DataImputBlock[countNum].H3;
                gCL = DataImputBlock[countNum].L3;
            }
			// count next DataLine
			if(countData < PROCESS_WIDTH){countNum ++;}
			else{next_t = 0;}
		}
		else if(simu_time >= (DataImputBlock[countNum].T + 0.5*TICK_STEP))
        {
            printf("Error simu_time counting!\r\n");
            break;
        }

    /*--------------------运行模型-------------------------------------------*/
        // 四阶龙格库塔法仿真计算
        // 统一计算缓存量
        double ula = 0.0, ulb = 0.0, ulc = 0.0;
        //double uda = 0.0, udb = 0.0, udc = 0.0;
        // 电流值缓存量
        double ki_1[3] = {0.0, 0.0, 0.0};// 存储顺序为A,B,C相
        double ki_2[3] = {0.0, 0.0, 0.0};
        double ki_3[3] = {0.0, 0.0, 0.0};
    	double ki_4[3] = {0.0, 0.0, 0.0};
    	double i_1[3] = {0.0, 0.0, 0.0};// 存储顺序为A,B,C相
    	double i_2[3] = {0.0, 0.0, 0.0};
    	double i_3[3] = {0.0, 0.0, 0.0};
    	double i_0[3]; i_0[0] = ia; i_0[1] = ib; i_0[2] = ic;// 储存零时刻电流
        // 转速（机械角速度）值/机械角度值缓存量
    	//double kdomega_1 = 0.0, kdomega_2 = 0.0, kdomega_3 = 0.0,
        //      kdomega_4 = 0.0;
    	double komega_1 = 0.0, komega_2 = 0.0, komega_3 = 0.0, komega_4 = 0.0;
    	double omega_1 = 0.0, omega_2 = 0.0, omega_3 = 0.0, omega_0 = omega;
    	double ktheta_1 = 0.0, ktheta_2 = 0.0, ktheta_3 = 0.0, ktheta_4 = 0.0;
    	double theta_1 = 0.0, theta_2 = 0.0, theta_3 = 0.0, theta_0 = theta;

        // 开始计算k1
        ula = ua - um - Ra*ia - Ea;
        ulb = ub - um - Rb*ib - Eb;
        ulc = uc - um - Rc*ic - Ec;
        ki_1[0] = (Lk_num1*ula + Lk_num4*ulb + Lk_num5*ulc)/Lk_den;
        ki_1[1] = (Lk_num4*ula + Lk_num2*ulb + Lk_num6*ulc)/Lk_den;
        ki_1[2] = (Lk_num5*ula + Lk_num6*ulb + Lk_num3*ulc)/Lk_den;
#ifdef REVERSE
        komega_1 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta - 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta - 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#else
        komega_1 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta + 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta + 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#endif
        ktheta_1 = omega;
        i_1[0] = i_0[0] + ki_1[0]*TICK_STEP/2; ia = i_1[0];
        i_1[1] = i_0[1] + ki_1[1]*TICK_STEP/2; ib = i_1[1];
        i_1[2] = i_0[2] + ki_1[2]*TICK_STEP/2; ic = i_1[2];
        omega_1 = omega_0 + komega_1*TICK_STEP/2; omega = omega_1;
        theta_1 = theta_0 + ktheta_1*TICK_STEP/2;
        if(theta_1 >= 0.0){theta = fmod(theta_1, 2*M_PI);}
        else{theta = theta_1 - 2*M_PI*(floor(theta_1/(2*M_PI)));}
/*--------------------更新反电动势-------------------------------------------*/
    // 计算电角度
    etheta = fmod(theta*POLES, 2*M_PI);
    // 计算反电动势
#ifdef REVERSE
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
        Ec = -K*omega;
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
        Ec = K*omega;
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#else
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
        Ec = -K*omega;
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
        Ec = K*omega;
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#endif
/*--------------------更新输入电压-------------------------------------------*/
    /* 导通状态 */
    // 计算A相桥输出电压
    if((gAH == 1) && (gAL == 0)){ua = Vcc;}
    if((gAH == 0) && (gAL == 1)){ua = 0.0;}
    if((gAH == 1) && (gAL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算B相桥输出电压
    if((gBH == 1) && (gBL == 0)){ub = Vcc;}
    if((gBH == 0) && (gBL == 1)){ub = 0.0;}
    if((gBH == 1) && (gBL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算C相桥输出电压
    if((gCH == 1) && (gCL == 0)){uc = Vcc;}
    if((gCH == 0) && (gCL == 1)){uc = 0.0;}
    if((gCH == 1) && (gCL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    /* 截止状态 */
    // 计算A相桥输出电压
    if((gAH == 0) && (gAL == 0))
    {
        if(ia > I_MIN){ua = 0.0;}
        else if(ia < -I_MIN){ua = Vcc;}
        else
        {
            ua = -(Lk_num4*(Eb - ub + Rb*ib - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num5*(Ec - uc + Rc*ic - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num1*(Ea - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num1*((Lk_num1 + Lk_num4 + Lk_num5)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num5*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ua > Vcc){ua = Vcc;}
            if(ua < 0){ua = 0;}
        }
    }
    // 计算B相桥输出电压
    if((gBH == 0) && (gBL == 0))
    {
        if(ib > I_MIN){ub = 0.0;}
        else if(ib < -I_MIN){ub = Vcc;}
        else
        {
            ub = -(Lk_num4*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Ec - uc + Rc*ic - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num2*(Eb - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num2*((Lk_num2 + Lk_num4 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ub > Vcc){ub = Vcc;}
            if(ub < 0){ub = 0;}
        }
    }
    // 计算C相桥输出电压
    if((gCH == 0) && (gCL == 0))
    {
        if(ic > I_MIN){uc = 0.0;}
        else if(ic < -I_MIN){uc = Vcc;}
        else
        {
            uc = -(Lk_num5*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Eb - ub + Rb*ib - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num3*(Ec - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num3*((Lk_num3 + Lk_num5 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num5*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(uc > Vcc){uc = Vcc;}
            if(uc < 0){uc = 0;}
        }
    }
    /* 中性点电压 */
    um = -((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6);
/*--------------------更新状态完毕-------------------------------------------*/        
        // 开始计算k2
        ula = ua - um - Ra*ia - Ea;
        ulb = ub - um - Rb*ib - Eb;
        ulc = uc - um - Rc*ic - Ec;
        ki_2[0] = (Lk_num1*ula + Lk_num4*ulb + Lk_num5*ulc)/Lk_den;
        ki_2[1] = (Lk_num4*ula + Lk_num2*ulb + Lk_num6*ulc)/Lk_den;
        ki_2[2] = (Lk_num5*ula + Lk_num6*ulb + Lk_num3*ulc)/Lk_den;
#ifdef REVERSE
        komega_2 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta - 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta - 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#else
        komega_2 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta + 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta + 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#endif
        ktheta_2 = omega;
        i_2[0] = i_0[0] + ki_2[0]*TICK_STEP/2; ia = i_2[0];
        i_2[1] = i_0[1] + ki_2[1]*TICK_STEP/2; ib = i_2[1];
        i_2[2] = i_0[2] + ki_2[2]*TICK_STEP/2; ic = i_2[2];
        omega_2 = omega_0 + komega_2*TICK_STEP/2; omega = omega_2;
        theta_2 = theta_0 + ktheta_2*TICK_STEP/2;
        if(theta_2 >= 0.0){theta = fmod(theta_2, 2*M_PI);}
        else{theta = theta_2 - 2*M_PI*(floor(theta_2/(2*M_PI)));}
/*--------------------更新反电动势-------------------------------------------*/
    // 计算电角度
    etheta = fmod(theta*POLES, 2*M_PI);
    // 计算反电动势
#ifdef REVERSE
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
        Ec = -K*omega;
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
        Ec = K*omega;
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#else
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
        Ec = -K*omega;
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
        Ec = K*omega;
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#endif
/*--------------------更新输入电压-------------------------------------------*/
    /* 导通状态 */
    // 计算A相桥输出电压
    if((gAH == 1) && (gAL == 0)){ua = Vcc;}
    if((gAH == 0) && (gAL == 1)){ua = 0.0;}
    if((gAH == 1) && (gAL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算B相桥输出电压
    if((gBH == 1) && (gBL == 0)){ub = Vcc;}
    if((gBH == 0) && (gBL == 1)){ub = 0.0;}
    if((gBH == 1) && (gBL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算C相桥输出电压
    if((gCH == 1) && (gCL == 0)){uc = Vcc;}
    if((gCH == 0) && (gCL == 1)){uc = 0.0;}
    if((gCH == 1) && (gCL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    /* 截止状态 */
    // 计算A相桥输出电压
    if((gAH == 0) && (gAL == 0))
    {
        if(ia > I_MIN){ua = 0.0;}
        else if(ia < -I_MIN){ua = Vcc;}
        else
        {
            ua = -(Lk_num4*(Eb - ub + Rb*ib - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num5*(Ec - uc + Rc*ic - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num1*(Ea - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num1*((Lk_num1 + Lk_num4 + Lk_num5)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num5*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ua > Vcc){ua = Vcc;}
            if(ua < 0){ua = 0;}
        }
    }
    // 计算B相桥输出电压
    if((gBH == 0) && (gBL == 0))
    {
        if(ib > I_MIN){ub = 0.0;}
        else if(ib < -I_MIN){ub = Vcc;}
        else
        {
            ub = -(Lk_num4*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Ec - uc + Rc*ic - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num2*(Eb - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num2*((Lk_num2 + Lk_num4 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ub > Vcc){ub = Vcc;}
            if(ub < 0){ub = 0;}
        }
    }
    // 计算C相桥输出电压
    if((gCH == 0) && (gCL == 0))
    {
        if(ic > I_MIN){uc = 0.0;}
        else if(ic < -I_MIN){uc = Vcc;}
        else
        {
            uc = -(Lk_num5*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Eb - ub + Rb*ib - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num3*(Ec - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num3*((Lk_num3 + Lk_num5 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num5*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(uc > Vcc){uc = Vcc;}
            if(uc < 0){uc = 0;}
        }
    }
    /* 中性点电压 */
    um = -((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6);
/*--------------------更新状态完毕-------------------------------------------*/
        // 开始计算k3
        ula = ua - um - Ra*ia - Ea;
        ulb = ub - um - Rb*ib - Eb;
        ulc = uc - um - Rc*ic - Ec;
        ki_3[0] = (Lk_num1*ula + Lk_num4*ulb + Lk_num5*ulc)/Lk_den;
        ki_3[1] = (Lk_num4*ula + Lk_num2*ulb + Lk_num6*ulc)/Lk_den;
        ki_3[2] = (Lk_num5*ula + Lk_num6*ulb + Lk_num3*ulc)/Lk_den;
#ifdef REVERSE
        komega_3 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta - 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta - 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#else
        komega_3 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta + 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta + 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#endif
        ktheta_3 = omega;
        i_3[0] = i_0[0] + ki_3[0]*TICK_STEP; ia = i_3[0];
        i_3[1] = i_0[1] + ki_3[1]*TICK_STEP; ib = i_3[1];
        i_3[2] = i_0[2] + ki_3[2]*TICK_STEP; ic = i_3[2];
        omega_3 = omega_0 + komega_3*TICK_STEP; omega = omega_3;
        theta_3 = theta_0 + ktheta_3*TICK_STEP;
        if(theta_3 >= 0.0){theta = fmod(theta_3, 2*M_PI);}
        else{theta = theta_3 - 2*M_PI*(floor(theta_3/(2*M_PI)));}
/*--------------------更新反电动势-------------------------------------------*/
    // 计算电角度
    etheta = fmod(theta*POLES, 2*M_PI);
    // 计算反电动势
#ifdef REVERSE
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
        Ec = -K*omega;
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
        Ec = K*omega;
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#else
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
        Ec = -K*omega;
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
        Ec = K*omega;
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#endif
/*--------------------更新输入电压-------------------------------------------*/
    /* 导通状态 */
    // 计算A相桥输出电压
    if((gAH == 1) && (gAL == 0)){ua = Vcc;}
    if((gAH == 0) && (gAL == 1)){ua = 0.0;}
    if((gAH == 1) && (gAL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算B相桥输出电压
    if((gBH == 1) && (gBL == 0)){ub = Vcc;}
    if((gBH == 0) && (gBL == 1)){ub = 0.0;}
    if((gBH == 1) && (gBL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算C相桥输出电压
    if((gCH == 1) && (gCL == 0)){uc = Vcc;}
    if((gCH == 0) && (gCL == 1)){uc = 0.0;}
    if((gCH == 1) && (gCL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    /* 截止状态 */
    // 计算A相桥输出电压
    if((gAH == 0) && (gAL == 0))
    {
        if(ia > I_MIN){ua = 0.0;}
        else if(ia < -I_MIN){ua = Vcc;}
        else
        {
            ua = -(Lk_num4*(Eb - ub + Rb*ib - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num5*(Ec - uc + Rc*ic - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num1*(Ea - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num1*((Lk_num1 + Lk_num4 + Lk_num5)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num5*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ua > Vcc){ua = Vcc;}
            if(ua < 0){ua = 0;}
        }
    }
    // 计算B相桥输出电压
    if((gBH == 0) && (gBL == 0))
    {
        if(ib > I_MIN){ub = 0.0;}
        else if(ib < -I_MIN){ub = Vcc;}
        else
        {
            ub = -(Lk_num4*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Ec - uc + Rc*ic - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num2*(Eb - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num2*((Lk_num2 + Lk_num4 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ub > Vcc){ub = Vcc;}
            if(ub < 0){ub = 0;}
        }
    }
    // 计算C相桥输出电压
    if((gCH == 0) && (gCL == 0))
    {
        if(ic > I_MIN){uc = 0.0;}
        else if(ic < -I_MIN){uc = Vcc;}
        else
        {
            uc = -(Lk_num5*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Eb - ub + Rb*ib - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num3*(Ec - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num3*((Lk_num3 + Lk_num5 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num5*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(uc > Vcc){uc = Vcc;}
            if(uc < 0){uc = 0;}
        }
    }
    /* 中性点电压 */
    um = -((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6);
/*--------------------更新状态完毕-------------------------------------------*/
        // 开始计算k4
        ula = ua - um - Ra*ia - Ea;
        ulb = ub - um - Rb*ib - Eb;
        ulc = uc - um - Rc*ic - Ec;
        ki_4[0] = (Lk_num1*ula + Lk_num4*ulb + Lk_num5*ulc)/Lk_den;
        ki_4[1] = (Lk_num4*ula + Lk_num2*ulb + Lk_num6*ulc)/Lk_den;
        ki_4[2] = (Lk_num5*ula + Lk_num6*ulb + Lk_num3*ulc)/Lk_den;
#ifdef REVERSE
        komega_4 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta - 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta - 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#else
        komega_4 = (Kt*(ia*sin(etheta - M_PI/6)
                + ib*sin(etheta + 2*M_PI/3 - M_PI/6)
                + ic*sin(etheta + 4*M_PI/3 - M_PI/6)) - Tl - C*omega)/J;
#endif
        ktheta_4 = omega;
        // 更新仿真结果
        // 更新电流值
        ia = i_0[0] + (ki_1[0] + 2*ki_2[0] + 2*ki_3[0] + ki_4[0])*TICK_STEP/6;
        ib = i_0[1] + (ki_1[1] + 2*ki_2[1] + 2*ki_3[1] + ki_4[1])*TICK_STEP/6;
        ic = i_0[2] + (ki_1[2] + 2*ki_2[2] + 2*ki_3[2] + ki_4[2])*TICK_STEP/6;
        // 更新转速（机械角速度）值
        omega = omega_0
                + (komega_1 + 2*komega_2 + 2*komega_3 + komega_4)*TICK_STEP/6;
        // 更新机械角度值
        theta = theta_0
                + (ktheta_1 + 2*ktheta_2 + 2*ktheta_3 + ktheta_4)*TICK_STEP/6;
        if(theta >= 0.0){theta = fmod(theta, 2*M_PI);}
        else{theta = theta - 2*M_PI*(floor(theta/(2*M_PI)));}
/*--------------------更新反电动势-------------------------------------------*/
    // 计算电角度
    etheta = fmod(theta*POLES, 2*M_PI);
    // 计算反电动势
#ifdef REVERSE
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
        Ec = -K*omega;
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
        Ec = K*omega;
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#else
    if((etheta >= 0.0) && (etheta < (M_PI/3)))
    {
        Ea = K*omega*((etheta*6/M_PI) - 1);
        Eb = K*omega;
        Ec = -K*omega;
    }
    else if((etheta >= (M_PI/3)) && (etheta < (2*M_PI/3)))
    {
        Ea = K*omega;
        Eb = K*omega*(1 - ((etheta - M_PI/3)*6/M_PI));
        Ec = -K*omega;
    }
    else if((etheta >= (2*M_PI/3)) && (etheta < M_PI))
    {
        Ea = K*omega;
        Eb = -K*omega;
        Ec = K*omega*(((etheta - 2*M_PI/3)*6/M_PI) - 1);
    }
    else if((etheta >= M_PI) && (etheta < (4*M_PI/3)))
    {
        Ea = K*omega*(1 - ((etheta - M_PI)*6/M_PI));
        Eb = -K*omega;
        Ec = K*omega;
    }
    else if((etheta >= (4*M_PI/3)) && (etheta < (5*M_PI/3)))
    {
        Ea = -K*omega;
        Eb = K*omega*(((etheta - 4*M_PI/3)*6/M_PI) - 1);
        Ec = K*omega;
    }
    else if((etheta >= (5*M_PI/3)) && (etheta < (2*M_PI)))
    {
        Ea = -K*omega;
        Eb = K*omega;
        Ec = K*omega*(1 - ((etheta - 5*M_PI/3)*6/M_PI));
    }
    else
    {
        printf("etheta error!\r\ntheta: %lf, etheta: %lf\r\n", theta, etheta);
        next_t = 0;
    }
#endif
    // 调试测试输出
//  printf("BackEMF: %lf, %lf, %lf\r\n", Ea, Eb, Ec);

/*--------------------更新输入电压-------------------------------------------*/
    /* 导通状态 */
    // 计算A相桥输出电压
    if((gAH == 1) && (gAL == 0)){ua = Vcc;}
    if((gAH == 0) && (gAL == 1)){ua = 0.0;}
    if((gAH == 1) && (gAL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算B相桥输出电压
    if((gBH == 1) && (gBL == 0)){ub = Vcc;}
    if((gBH == 0) && (gBL == 1)){ub = 0.0;}
    if((gBH == 1) && (gBL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    // 计算C相桥输出电压
    if((gCH == 1) && (gCL == 0)){uc = Vcc;}
    if((gCH == 0) && (gCL == 1)){uc = 0.0;}
    if((gCH == 1) && (gCL == 1))
    {printf("driver error!\r\n");next_t = 0;}
    /* 截止状态 */
    // 计算A相桥输出电压
    if((gAH == 0) && (gAL == 0))
    {
        if(ia > I_MIN){ua = 0.0;}
        else if(ia < -I_MIN){ua = Vcc;}
        else
        {
            ua = -(Lk_num4*(Eb - ub + Rb*ib - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num5*(Ec - uc + Rc*ic - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num1*(Ea - ((Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Ea*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num1*((Lk_num1 + Lk_num4 + Lk_num5)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num5*(Lk_num1 + Lk_num4 + Lk_num5))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ua > Vcc){ua = Vcc;}
            if(ua < 0){ua = 0;}
        }
    }
    // 计算B相桥输出电压
    if((gBH == 0) && (gBL == 0))
    {
        if(ib > I_MIN){ub = 0.0;}
        else if(ib < -I_MIN){ub = Vcc;}
        else
        {
            ub = -(Lk_num4*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Ec - uc + Rc*ic - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num2*(Eb - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic) + Eb*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num2*((Lk_num2 + Lk_num4 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num4*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num2 + Lk_num4 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(ub > Vcc){ub = Vcc;}
            if(ub < 0){ub = 0;}
        }
    }
    // 计算C相桥输出电压
    if((gCH == 0) && (gCL == 0))
    {
        if(ic > I_MIN){uc = 0.0;}
        else if(ic < -I_MIN){uc = Vcc;}
        else
        {
            uc = -(Lk_num5*(Ea - ua + Ra*ia - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num6*(Eb - ub + Rb*ib - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)) + Lk_num3*(Ec - ((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + Ec*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6)))/(Lk_num3*((Lk_num3 + Lk_num5 + Lk_num6)/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) - 1) + (Lk_num5*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6) + (Lk_num6*(Lk_num3 + Lk_num5 + Lk_num6))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6));
            if(uc > Vcc){uc = Vcc;}
            if(uc < 0){uc = 0;}
        }
    }
    /* 中性点电压 */
    um = -((Lk_num1 + Lk_num4 + Lk_num5)*(Ea - ua + Ra*ia) + (Lk_num2 + Lk_num4 + Lk_num6)*(Eb - ub + Rb*ib) + (Lk_num3 + Lk_num5 + Lk_num6)*(Ec - uc + Rc*ic))/(Lk_num1 + Lk_num2 + Lk_num3 + 2*Lk_num4 + 2*Lk_num5 + 2*Lk_num6);
/*--------------------更新状态完毕-------------------------------------------*/
        // 调试测试输出
        //printf("k1:\r\n%lf, %lf, %lf, \r\n", ki_1[0], ki_1[1], ki_1[2]);
        //printf("k2:\r\n%lf, %lf, %lf, \r\n", ki_2[0], ki_2[1], ki_2[2]);
        //printf("k3:\r\n%lf, %lf, %lf, \r\n", ki_3[0], ki_3[1], ki_3[2]);
        //printf("k4:\r\n%lf, %lf, %lf, \r\n", ki_4[0], ki_4[1], ki_4[2]);
        //printf("%.9lf, kdomega1-4:%lf, %lf, %lf, %lf  ", simu_time,
        //      kdomega_1, kdomega_2, kdomega_3, kdomega_4);
        //printf("%.9lf, komega1-4:%lf, %lf, %lf, %lf", simu_time,
        //      komega_1, komega_2, komega_3, komega_4);
        //printf("ktheta1-4:%lf, %lf, %lf, %lf\r\n", ktheta_1, ktheta_2,
        //      ktheta_3, ktheta_4);
        //printf("simu_time1/omega/theta:\r\n%lf, %lf, %lf/ %lf/ %lf\r\n",
        //      ic, ub, uc, omega, theea);
#ifdef BLDCM_SIMU_OUTPUT
        fprintf(fp2, "%.9lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf "
                "%.6lf %.6lf %.6lf %.6lf %.6lf\n", simu_time, ia, ib, ic,
                ua, ub, uc, Ea, Eb, Ec, um, omega, etheta, theta);
#endif
    /*--------------------滤波器---------------------------------------------*/
		f_stepCount ++;
		fc_ia = fc_ia + ia; fc_ib = fc_ib + ib; fc_ic = fc_ic + ic;
	/*--------------------按时间步进-----------------------------------------*/
		simu_time = simu_time + TICK_STEP;
    }
/*--------------------模拟循环结束-------------------------------------------*/
    
/*--------------------记录下次仿真的零状态值---------------------------------*/    
    MotorZeroState -> Time = MotorSimuBlock[0].Time;
    MotorZeroState -> ia = MotorSimuBlock[0].ia;
    MotorZeroState -> ib = MotorSimuBlock[0].ib;
    MotorZeroState -> ic = MotorSimuBlock[0].ic;
    MotorZeroState -> H1 = MotorSimuBlock[0].H1;
    MotorZeroState -> L1 = MotorSimuBlock[0].L1;
    MotorZeroState -> H2 = MotorSimuBlock[0].H2;
    MotorZeroState -> L2 = MotorSimuBlock[0].L2;
    MotorZeroState -> H3 = MotorSimuBlock[0].H3;
    MotorZeroState -> L3 = MotorSimuBlock[0].L3;
    MotorZeroState -> Vzero = MotorSimuBlock[0].Vzero;
    MotorZeroState -> theta = MotorSimuBlock[0].theta;
    MotorZeroState -> omega = MotorSimuBlock[0].omega;

/*--------------------关闭并转存仿真输出文档---------------------------------*/
#ifdef BLDCM_SIMU_OUTPUT
    int outputMotorFileClose;
    outputMotorFileClose = fclose(fp2);
    if(outputMotorFileClose == 0){printf("OutputMotorFile "
            "has been closed~\r\n");}
    else{printf("error closing outputMotorFile!\r\n");}
#endif
}

/* user code end 3*/

/* end of file --------------------------------------------------------------*/
