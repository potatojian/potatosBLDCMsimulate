/* header -------------------------------------------------------------------*/
/**
  * 文件名称：Main.c
  * 日    期：2023/04/04
  * 作    者：mrpotato
  * 简    述：用于实现强化学习回归参数
  */ 

/* includes -----------------------------------------------------------------*/
#include "Basic.h"
#include "BLDCMotorModel.h"
#include "RLModel.h"

/* typedef ------------------------------------------------------------------*/

/* define -------------------------------------------------------------------*/

/* MotorModelSet */

/* variables ----------------------------------------------------------------*/
/* filesForInputs&Outputs */
char *inputDataFile
		= "../20220419_BLDCData/BLDC_Bearing01_4/BLDCData_Bearing01_4.csv";
char *outputResultFile = "RLresult.csv";

/* function prototypes ------------------------------------------------------*/
void *threadFunc(void *args);
void firstFileLoadInit(FILE *fp1, _motor_line *MotorZeroState,
		_data_line *DataImputBlock);
bool appendLoad(FILE *fp1, _data_line *DataImputBlock);
bool itobool(int input);

/* user code ----------------------------------------------------------------*/
/* user code begin 0*/

/* user code end 0*/

/**
  * 函数功能：主函数
  * 输入参数：无
  * 返 回 值：int
  * 相关变量：
  * 说    明：
  */
int main(void)
{
    /* user code begin 2*/
	printf("\r\n");
	printf("__________________________________________________________________"
            "______________\r\n");
	printf("**Potatos_BLDC_A3C_Test Begin-->\r\n");
	printf("__________________________________________________________________"
            "______________\r\n");
	
	srand(time(NULL));
    //srand(100);
    
    pthread_t th1;
	pthread_t th2;
	
	_thread_args args1;
	args1.threadIndex = 1;
	_thread_args args2;
	args2.threadIndex = 2;
	
	pthread_create(&th1, NULL, threadFunc, &args1);
	pthread_create(&th2, NULL, threadFunc, &args2);

	pthread_join(th1, NULL);
	pthread_join(th2, NULL);

    /* user code end 2*/

    //scanf("%s");//或许需要暂停查看输出
    return 0;
}
/* user code begin 3*/
/**
  * 函数功能：运行一个一个一个线程程序
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
void *threadFunc(void *args)
{
	_thread_args *get_args = (_thread_args *) args;
	_data_line *DataImputBlock = get_args -> DataImputBlock;
	_motor_line *MotorZeroState = &(get_args -> MotorZeroState);
	_motor_line *MotorSimuBlock = get_args -> MotorSimuBlock;
	_RL_trajectory *trajectories = get_args -> trajectories;
	_RL_worker *Worker = &(get_args -> Worker);
	unsigned char threadIndex = get_args -> threadIndex;
	
	FILE *fp1 = NULL;
	FILE *fp4 = NULL;
	unsigned int tjtryDataIndexValue = 0;
	unsigned int *tjtryDataIndex = &(tjtryDataIndexValue);
	unsigned long RLIter;
    
	for(RLIter = 0; RLIter < MAX_ITER; RLIter ++)
    {
		// 打开IO文件和输出通道
		fp1 = fopen(inputDataFile, "r");
		if(fp1 == NULL){printf("error opening inputFile!\r\n");}
		fp4 = fopen(outputResultFile, "w");
		if(fp4 == NULL){printf("error opening resultFile!\r\n");}
		// 首次读入文档加载
		firstFileLoadInit(fp1, MotorZeroState, DataImputBlock);
		// 配置初始化参数
		trajectories[0].La = 0.00002; trajectories[0].Lb = 0.00002;
				trajectories[0].Lc = 0.00002;
		trajectories[0].Mab = 0.000001; trajectories[0].Mbc = 0.000001;
				trajectories[0].Mca = 0.000001;
		trajectories[0].Ra = 0.001; trajectories[0].Rb = 0.001;
				trajectories[0].Rc = 0.001;
		trajectories[0].K = 0.00002*30/M_PI;
		trajectories[0].Tl = 0.0;
		trajectories[0].J = 0.0000006;
		trajectories[0].C = 0.0;
		// 首次仿真获得第一组特征值
		BLDCMotorModel(threadIndex, RLIter, tjtryDataIndex, DataImputBlock,
				MotorZeroState, MotorSimuBlock, trajectories);
		// RL_Worker空间初始化
		RLInitWorker(Worker);
		// RL循环
		RLExecuteWorker(fp1, threadIndex, RLIter, tjtryDataIndex, DataImputBlock,
				MotorZeroState, MotorSimuBlock, trajectories, Worker);
		//appendLoad(fp1, DataImputBlock);
		//printf("thread%d append time: %lf\r\n", threadIndex,
		//		DataImputBlock[0].T);
		
		
		
	}
	
	// 关闭文件
    int inputDataFileClose;
    inputDataFileClose = fclose(fp1);
    if(inputDataFileClose == 0){printf("InputDataFile"
            " has been closed~\r\n");}
    else{printf("error closing inputDataFile!\r\n");}
    int outputResultFileClose;
    outputResultFileClose = fclose(fp4);
    if(outputResultFileClose == 0){printf("outputResultFile "
            "has been closed~\r\n");}
    else{printf("error closing outputResultFile!\r\n");}
	
	return NULL;
}

/**
  * 函数功能：首次读入文档及参数初始化
  * 输入参数：无
  * 返 回 值：无
  * 相关变量：
  * 说    明：
  */
void firstFileLoadInit(FILE *fp1, _motor_line *MotorZeroState,
		_data_line *DataImputBlock)
{
	char line[1024];
	char *sp;
	
    // 纪录起始数据
	while(flock(fileno(fp1), LOCK_EX) == -1){}
    if(fgets(line,1024,fp1) != NULL)// 读入零状态
    {
        sp = strtok(line, ",");
        MotorZeroState -> Time = atof(sp);
        sp = strtok(NULL, ",");// Da
        sp = strtok(NULL, ",");// Dd
        sp = strtok(NULL, ",");
        MotorZeroState -> I3 = atof(sp);
        sp = strtok(NULL, ",");
        MotorZeroState -> I1 = atof(sp);
        sp = strtok(NULL, ",");
        MotorZeroState -> I2 = atof(sp);
        sp = strtok(NULL, ",");
        MotorZeroState -> V3 = atof(sp);
        sp = strtok(NULL, ",");
        MotorZeroState -> V2 = atof(sp);
        sp = strtok(NULL, ",");
        MotorZeroState -> V1 = atof(sp);
        sp = strtok(NULL, ",");
        MotorZeroState -> H1 = itobool(atoi(sp));
        sp = strtok(NULL, ",");
        MotorZeroState -> H2 = itobool(atoi(sp));
        sp = strtok(NULL, ",");
        MotorZeroState -> H3 = itobool(atoi(sp));
        sp = strtok(NULL, ",");
        MotorZeroState -> L1 = itobool(atoi(sp));
        sp = strtok(NULL, ",");
        MotorZeroState -> L2 = itobool(atoi(sp));
        sp = strtok(NULL, ",");
        MotorZeroState -> L3 = itobool(atoi(sp));
        sp = strtok(NULL, ",");
        MotorZeroState -> Itot = atof(sp);
        sp = strtok(NULL, ",");
        MotorZeroState -> Vzero = atof(sp);
    }
    else
    {
        printf("error imputing initial data!\r\n");
    }
	flock(fileno(fp1), LOCK_UN);
    // 读入第一批数据
    unsigned int countWidth = 0; 
    unsigned int countLineNum = 0;
    while(countWidth < PROCESS_WIDTH)
    {
		while(flock(fileno(fp1), LOCK_EX) == -1){}
        if(fgets(line,1024,fp1) != NULL)
        {
            sp = strtok(line, ",");
            DataImputBlock[countLineNum].T = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Da = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Dd = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].I3 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].I1 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].I2 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].V3 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].V2 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].V1 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].H1 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].H2 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].H3 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].L1 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].L2 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].L3 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Itot = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Vzero = atof(sp);
            if(DataImputBlock[countLineNum].Da){countWidth ++;}
            countLineNum ++;
        }
        else
        {
            printf("error imputing initial data!\r\n");
            break;
        }
		flock(fileno(fp1), LOCK_UN);
    }
	// 配置初始状态
		MotorZeroState -> ia = 0.0; MotorZeroState -> ib = 0.0;
				MotorZeroState -> ic = 0.0;
		MotorZeroState -> theta = M_PI*9/(6*5);
		MotorZeroState -> omega = 0.0;
}

/**
  * 函数功能：读入下一步数据
  * 输入参数：无
  * 返 回 值：无
  * 相关变量：
  * 说    明：
  */
bool appendLoad(FILE *fp1, _data_line *DataImputBlock)
{
	char line[1024];
	char *sp;
    unsigned int countWidth = 1;
    unsigned int countLineNum = 0;
    unsigned int outSet = 0;
    bool outPut = 1;
    while(!DataImputBlock[countLineNum].Da){countLineNum ++;}
    outSet = countLineNum + 1;
    countLineNum = 0;
    while(countWidth < PROCESS_WIDTH)
    {
        unsigned int tar = countLineNum + outSet;
        DataImputBlock[countLineNum].T = DataImputBlock[tar].T;
        DataImputBlock[countLineNum].Da = DataImputBlock[tar].Da;
        DataImputBlock[countLineNum].Dd = DataImputBlock[tar].Dd;
        DataImputBlock[countLineNum].I3 = DataImputBlock[tar].I3;
        DataImputBlock[countLineNum].I1 = DataImputBlock[tar].I1;
        DataImputBlock[countLineNum].I2 = DataImputBlock[tar].I2;
        DataImputBlock[countLineNum].V3 = DataImputBlock[tar].V3;
        DataImputBlock[countLineNum].V2 = DataImputBlock[tar].V2;
        DataImputBlock[countLineNum].V1 = DataImputBlock[tar].V1;
        DataImputBlock[countLineNum].H1 = DataImputBlock[tar].H1;
        DataImputBlock[countLineNum].H2 = DataImputBlock[tar].H2;
        DataImputBlock[countLineNum].H3 = DataImputBlock[tar].H3;
        DataImputBlock[countLineNum].L1 = DataImputBlock[tar].L1;
        DataImputBlock[countLineNum].L2 = DataImputBlock[tar].L2;
        DataImputBlock[countLineNum].L3 = DataImputBlock[tar].L3;
        DataImputBlock[countLineNum].Itot = DataImputBlock[tar].Itot;
        DataImputBlock[countLineNum].Vzero = DataImputBlock[tar].Vzero;
        if(DataImputBlock[countLineNum].Da){countWidth ++;}
        countLineNum ++;
    }
    countWidth = 1;
    while(countWidth)
    {
		while(flock(fileno(fp1), LOCK_EX) == -1){}
        if(fgets(line,1024,fp1) != NULL)
        {
            sp = strtok(line, ",");
            DataImputBlock[countLineNum].T = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Da = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Dd = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].I3 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].I1 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].I2 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].V3 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].V2 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].V1 = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].H1 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].H2 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].H3 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].L1 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].L2 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].L3 = itobool(atoi(sp));
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Itot = atof(sp);
            sp = strtok(NULL, ",");
            DataImputBlock[countLineNum].Vzero = atof(sp);
            if(DataImputBlock[countLineNum].Da){countWidth = 0;}
            countLineNum ++;
        }
        else
        {
            printf("Successfully appended all data!\r\n");
            outPut = 0;
            break;
        }
		flock(fileno(fp1), LOCK_UN);
    }
    return outPut;
}

/**
  * 函数功能：整形转换为布尔类型
  * 输入参数：int input
  * 返 回 值：bool
  * 相关变量：
  * 说    明：
  */
bool itobool(int input)
{
    if(input == 0){return false;}
    else{return true;}
}
 
/* user code end 3*/

/* end of file --------------------------------------------------------------*/

