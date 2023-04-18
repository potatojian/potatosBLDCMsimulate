/* header -------------------------------------------------------------------*/
/**
  * 文件名称：RLModel.c
  * 日    期：2023/04/04
  * 作    者：mrpotato
  * 简    述：oneDNN库强化学习神经网络模型
  */ 
//// Required for posix_memalign
//#define _POSIX_C_SOURCE 200112L

/* includes -----------------------------------------------------------------*/
#include "Basic.h"
#include "BLDCMotorModel.h"
#include "RLModel.h"

/* typedef ------------------------------------------------------------------*/

/* define -------------------------------------------------------------------*/

// reference neural network model structure
/*
|-----------------------------------------------------------------------------|
|-->IN_CONV1: float32 1*20*PROCESS_WIDTH[b*c*w]
|
|convolution1: 40xkernel:20*3[c*w] stride:1 padding:1 dilation:0
|-->CONV1_RELU1: float32 1*40*PROCESS_WIDTH[b*c*w]
|
|relu1: negative_slop:0.0
|-->RELU1_POOL1: float32 1*40*PROCESS_WIDTH[b*c*w]
|
|pool1: max_pool kernel:40*2[c*w] stride:2 padding:0 dilation:0
|-->POOL1_CONV2: float32 1*40*(PROCESS_WIDTH/2)[b*c*w]
|
|convolution2: 80xkernel:40*3[c*w] stride:1 padding:1 dilation:0
|-->CONV2_RELU2: float32 1*80*(PROCESS_WIDTH/2)[b*c*w]
|
|relu2: negative_slop:0.0
|-->RELU2_POOL2: float32 1*80*(PROCESS_WIDTH/2)[b*c*w]
|
|pool2: max_pool kernel:80*2[c*w] stride:2 padding:0 dilation:0
|-->POOL2_CONV3: float32 1*80*(PROCESS_WIDTH/4)[b*c*w]
|
|convolution3: 160xkernel:80*3[c*w] stride:1 padding:1 dilation:0
|-->CONV3_RELU3: float32 1*160*(PROCESS_WIDTH/4)[b*c*w]
|
|relu3: negative_slop:0.0
|-->RELU3_POOL3: float32 1*160*(PROCESS_WIDTH/4)[b*c*w]
|
|pool3: max_pool kernel:160*2[c*w] stride:2 padding:0 dilation:0
|-->POOL3_RSHA: float32 1*160*(PROCESS_WIDTH/8)[b*c*w]
|
|reshape: [b*c*w]->[b*c]
|-->RSHA_DNSEA1_DNSEC1: float32 1*(20PROCESS_WIDTH)[b*c]
|-----------------------------------------------------------------------------|
|dense_actor1: 20PROCESS_WIDTH->1024[c]|dense_critic1: (same as dense_actor1) |
|-->DNSEA1_RELUA1: float32 1*1024[b*c] |-->DNSEC1_RELUC1: float32 1*1024[b*c] |
|
|relu_actor1: negative_slop:0.0        |relu_critic1: negative_slop:0.0       |
|-->RELUA1_DNSEA2: float32 1*1024[b*c] |-->RELUC1_DNSEC2: float32 1*1024[b*c] |
|
|dense_actor2: 1024->512[c]            |dense_critic2: 1024->512[c]           |
|-->DNSEA2_RELUA2: float32 1*512[b*c]  |-->DNSEC2_RELUC2: float32 1*512[b*c]  |
|
|relu_actor2: negative_slop:0.0        |relu_critic2: negative_slop:0.0       |
|-->RELUA2_DNSEA3: float32 1*512[b*c]  |-->RELUC2_DNSEC3: float32 1*512[b*c]  |
|
|dense_actor3: 512->256[c]             |dense_critic3: 512->256[c]            |
|-->DNSEA3_RELUA3: float32 1*256[b*c]  |-->DNSEC3_RELUC3: float32 1*256[b*c]  |
|
|relu_actor3: negative_slop:0.0        |relu_critic3: negative_slop:0.0       |
|-->RELUA3_DNSEA4: float32 1*256[b*c]  |-->RELUC3_DNSEC4: float32 1*256[b*c]  |
|
|dense_actor4: 256->128[c]             |dense_critic4: 256->128[c]            |
|-->DNSEA4_RELUA4: float32 1*128[b*c]  |-->DNSEC4_RELUC4: float32 1*128[b*c]  |
|
|relu_actor4: negative_slop:0.0        |relu_critic4: negative_slop:0.0       |
|-->RELUA4_DNSEA5: float32 1*256[b*c]  |-->RELUC4_DNSEC5: float32 1*256[b*c]  |
|
|dense_actor5: 128->13[c]              |dense_critic5: 128->1[c]              |
|-->DNSEA5_OUTA: float32 1*13[b*c]     |-->DNSEC5_OUTC: float32 1*1[b*c]      |
*/

// dimensions
#define CONV_DIMS 4
#define DNSE_DIMS 2

// cnn
#define BATCH 1
#define IN_CONV1_C 20
#define IN_CONV1_W PROCESS_WIDTH

#define CONV1_KERNEL_W 3
#define CONV1_STRIDE 1
#define CONV1_DILATION 0
#define CONV1_PADDING 1
#define CONV1_RELU1_C 40
#define CONV1_RELU1_W PROCESS_WIDTH

#define RELU1_NSLOP 0.0
#define RELU1_POOL1_C 40 
#define RELU1_POOL1_W PROCESS_WIDTH

#define POOL1_STRIDE 2
#define POOL1_DILATION 0
#define POOL1_PADDING 0
#define POOL1_CONV2_C 40 
#define POOL1_CONV2_W (PROCESS_WIDTH/2)

#define CONV2_KERNEL_W 3
#define CONV2_STRIDE 1
#define CONV2_DILATION 0
#define CONV2_PADDING 1
#define CONV2_RELU2_C 80
#define CONV2_RELU2_W (PROCESS_WIDTH/2)

#define RELU2_NSLOP 0.0
#define RELU2_POOL2_C 80 
#define RELU2_POOL2_W (PROCESS_WIDTH/2)

#define POOL2_STRIDE 2
#define POOL2_DILATION 0
#define POOL2_PADDING 0
#define POOL2_CONV3_C 80 
#define POOL2_CONV3_W (PROCESS_WIDTH/4)

#define CONV3_KERNEL_W 3
#define CONV3_STRIDE 1
#define CONV3_DILATION 0
#define CONV3_PADDING 1
#define CONV3_RELU3_C 160
#define CONV3_RELU3_W (PROCESS_WIDTH/4)

#define RELU3_NSLOP 0.0
#define RELU3_POOL3_C 160 
#define RELU3_POOL3_W (PROCESS_WIDTH/4)

#define POOL3_STRIDE 2
#define POOL3_DILATION 0
#define POOL3_PADDING 0
#define POOL3_RSHA_C 160 
#define POOL3_RSHA_W (PROCESS_WIDTH/8)

#define RSHA_DNSEA1_DNSEC1_C (PROCESS_WIDTH*20)

// actor
#define DNSEA1_RELUA1_C (PROCESS_WIDTH*32)

#define RELUA1_NSLOP 0.0
#define RELUA1_DNSEA2_C (PROCESS_WIDTH*32)

#define DNSEA2_RELUA2_C (PROCESS_WIDTH*32)

#define RELUA2_NSLOP 0.0
#define RELUA2_DNSEA3_C (PROCESS_WIDTH*32)

#define DNSEA3_RELUA3_C (PROCESS_WIDTH*32)

#define RELUA3_NSLOP 0.0
#define RELUA3_DNSEA4_C (PROCESS_WIDTH*32)

#define DNSEA4_RELUA4_C (PROCESS_WIDTH*32)

#define RELUA4_NSLOP 0.0
#define RELUA4_DNSEA5_C (PROCESS_WIDTH*32)

#define DNSEA5_OUTA_C 13

// critic
#define DNSEC1_RELUC1_C (PROCESS_WIDTH*32)

#define RELUC1_NSLOP 0.0
#define RELUC1_DNSEC2_C (PROCESS_WIDTH*32)

#define DNSEC2_RELUC2_C (PROCESS_WIDTH*32)

#define RELUC2_NSLOP 0.0
#define RELUC2_DNSEC3_C (PROCESS_WIDTH*32)

#define DNSEC3_RELUC3_C (PROCESS_WIDTH*32)

#define RELUC3_NSLOP 0.0
#define RELUC3_DNSEC4_C (PROCESS_WIDTH*32)

#define DNSEC4_RELUC4_C (PROCESS_WIDTH*32)

#define RELUC4_NSLOP 0.0
#define RELUC4_DNSEC5_C (PROCESS_WIDTH*32)

#define DNSEC5_OUTC_C 1

/* variables ----------------------------------------------------------------*/

/* function prototypes ------------------------------------------------------*/
/* basic */
double uniformDistribution_random(double leftEndPoint, double rightEndPoint);
double normalDistribution_random(double mu, double sigma);
/* DNNLhandle */
static size_t product(dnnl_dim_t *arr, uint32_t dim);
static void init_net_data(float *data, uint32_t dim, const dnnl_dim_t *dims);
static void prepare_arg_node(args_t *node, int nargs);
static void free_arg_node(args_t *node);
static void set_arg(dnnl_exec_arg_t *arg, int arg_idx, dnnl_memory_t memory);
static void init_data_memory(uint32_t dim, const dnnl_dim_t *dims,
        dnnl_format_tag_t user_tag, dnnl_engine_t engine, float *data,
        dnnl_memory_t *memory);
dnnl_status_t prepare_reorder(dnnl_memory_t *user_memory,
        const_dnnl_memory_desc_t prim_memory_md,
        dnnl_engine_t prim_engine, int dir_is_user_to_prim,
        dnnl_memory_t *prim_memory, dnnl_primitive_t *reorder,
        uint32_t *net_index, dnnl_primitive_t *net, args_t *net_args);

void printMatrix(float *data, uint32_t dim, uint32_t batch, uint32_t c,
        uint32_t h, uint32_t w);
void fprintMatrix(FILE *fp, float *data, uint32_t dim, uint32_t batch,
        uint32_t c, uint32_t h, uint32_t w);

/* public */
void RLInitWorker(_RL_worker *Worker);
void RLExecuteWorker(FILE *fp1, unsigned char threadIndex,
		unsigned long RLIter, unsigned int *tjtryDataIndex,
		_data_line *DataImputBlock, _motor_line *MotorZeroState,
		_motor_line *MotorSimuBlock, _RL_trajectory *trajectories,
		_RL_worker *Worker);
void RLUpdateActor();
void RLUpdateCritic();

/* user code ----------------------------------------------------------------*/
/* user code begin 3*/
/**
  * 函数功能：按照均匀分布生成一个随机数
  * 输入参数：double leftEndPoint,double rightEndPoint（分布域的左端点和右端点）
  * 返 回 值：double randomValue
  * 相关变量：<stdlib.h>
  * 说    明：无
  */
double uniformDistribution_random(double leftEndPoint, double rightEndPoint)
{
    if(rightEndPoint <= leftEndPoint)
    {printf("error uniform distribution random input!\r\n");return 0;}
    else
    {
        double randomValue;
        randomValue = (double)rand()/(RAND_MAX + 1.0);
        randomValue = randomValue*(rightEndPoint-leftEndPoint) + leftEndPoint;
        return randomValue;
    }
}

/**
  * 函数功能：按照正态分布生成一个随机数
  * 输入参数：double mu,double sigma（分布的平均值和方差）
  * 返 回 值：double randomValue
  * 相关变量：<stdlib.h>, <math.h>
  * 说    明：依照box_muller变换生成
  */
double normalDistribution_random(double mu, double sigma)
{
    if(sigma <= 0.0)
    {printf("error normal distribution random input!\r\n");return 0;}
    else
    {
        double u, v, randomValue;
        u = (double)rand()/(RAND_MAX + 1.0);
        v = (double)rand()/(RAND_MAX + 1.0);
        randomValue = sqrt(-2.0 * log(u)) * sin(2*M_PI * v);
        randomValue = randomValue*sigma + mu;
        return randomValue;
    }
}

/**
  * 函数功能：计算数据空间
  * 输入参数：dnnl_dim_t *arr 数据的维度特征；uint32_t dim 数据的维度
  * 返 回 值：size_t格式，返回这个多维数据所需的空间
  * 相关变量：
  * 说    明：
  */
static size_t product(dnnl_dim_t *arr, uint32_t dim)
{
    size_t prod = 1;
    for (size_t i = 0; i < dim; ++i)
        prod *= arr[i];
    return prod;
}

/**
  * 函数功能：初始化存储空间
  * 输入参数：float *data 要处理的空间指针；uint32_t dim 数据的维度；
  *             const dnnl_dim_t *dims 数据的维度特征
  * 返 回 值：
  * 相关变量：
  * 说    明：使用均匀分布初始化成[-1,1)范围随机数
  */
static void init_net_data(float *data, uint32_t dim, const dnnl_dim_t *dims)
{
    float band = sqrt(0.1875/PROCESS_WIDTH);
    if (dim == 1) {
        for (dnnl_dim_t i = 0; i < dims[0]; ++i) {
            data[i] = (float)uniformDistribution_random(-band,band);
        }
    }
    else if (dim == 2) {
        for(dnnl_dim_t in = 0; in < dims[0]; ++in)
            for(dnnl_dim_t iw = 0; iw < dims[1]; ++iw){
                dnnl_dim_t indx = in * dims[1] + iw;
                data[indx] = (float)uniformDistribution_random(-band,band);
            }
    }
    else if (dim == 3) {
        for(dnnl_dim_t in = 0; in < dims[0]; ++in)
            for(dnnl_dim_t ic = 0; ic < dims[1]; ++ic)
                for(dnnl_dim_t iw = 0; iw < dims[2]; ++iw){
                    dnnl_dim_t indx = in * dims[1] * dims[2]
                            + ic * dims[2] + iw;
                    data[indx] = (float)uniformDistribution_random(-band,band);
                }
    }
    else if (dim == 4) {
        for (dnnl_dim_t in = 0; in < dims[0]; ++in)
            for (dnnl_dim_t ic = 0; ic < dims[1]; ++ic)
                for (dnnl_dim_t ih = 0; ih < dims[2]; ++ih)
                    for (dnnl_dim_t iw = 0; iw < dims[3]; ++iw) {
                        dnnl_dim_t indx = in * dims[1] * dims[2] * dims[3]
                                + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx]
                                = (float)uniformDistribution_random(-band,band);
                    }
    }
}

/**
  * 函数功能：设置节点参数
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说     明：
  */
static void prepare_arg_node(args_t *node, int nargs)
{
    node->args = (dnnl_exec_arg_t *)malloc(sizeof(dnnl_exec_arg_t) * nargs);
    node->nargs = nargs;
}

/**
  * 函数功能：释放节点配置
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
static void free_arg_node(args_t *node)
{
    free(node->args);
}

/**
  * 函数功能：配置节点参数
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
static void set_arg(dnnl_exec_arg_t *arg, int arg_idx, dnnl_memory_t memory)
{
    arg->arg = arg_idx;
    arg->memory = memory;
}

/**
  * 函数功能：存储器创造及初始化函数
  * 输入参数：维度，维度特征，维度标签，计算引擎，存储空间，存储器
  * 返 回 值：
  * 相关变量：
  * 说    明：创造及初始化函数(init)内部流程：
  *           建立存储器描述符(dnnl_memory_desc_create_with_tag)；
  *           建立存储器(dnnl_memory_create)；
  *           删除存储器描述符(dnnl_memory_desc_destory)
  *           将存储空间内的数据写入存储器(分步)
  */
static void init_data_memory(uint32_t dim, const dnnl_dim_t *dims,
        dnnl_format_tag_t user_tag, dnnl_engine_t engine, float *data,
        dnnl_memory_t *memory)
{
    dnnl_memory_desc_t user_md;
    CHECK(dnnl_memory_desc_create_with_tag(
            &user_md, dim, dims, dnnl_f32, user_tag));
    CHECK(dnnl_memory_create(memory, user_md, engine, DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_desc_destroy(user_md));
    write_to_dnnl_memory(data, *memory);
}

/**
  * 函数功能：构建重定位环节
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
dnnl_status_t prepare_reorder(dnnl_memory_t *user_memory, // in
        const_dnnl_memory_desc_t prim_memory_md, // in
        dnnl_engine_t prim_engine, // in: primitive's engine
        int dir_is_user_to_prim, // in: user -> prim or prim -> user
        dnnl_memory_t *prim_memory, // out: primitive's memory created
        dnnl_primitive_t *reorder, // out: reorder primitive created
        uint32_t *net_index, // primitive index (inc if reorder created)
        dnnl_primitive_t *net, args_t *net_args)
{
    const_dnnl_memory_desc_t user_memory_md;
    dnnl_memory_get_memory_desc(*user_memory, &user_memory_md);

    dnnl_engine_t user_mem_engine;
    dnnl_memory_get_engine(*user_memory, &user_mem_engine);

    if (!dnnl_memory_desc_equal(user_memory_md, prim_memory_md)) {
        CHECK(dnnl_memory_create(prim_memory, prim_memory_md, prim_engine,
                DNNL_MEMORY_ALLOCATE));

        dnnl_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            CHECK(dnnl_reorder_primitive_desc_create(&reorder_pd,
                    user_memory_md, user_mem_engine, prim_memory_md,
                    prim_engine, NULL));
        } else {
            CHECK(dnnl_reorder_primitive_desc_create(&reorder_pd,
                    prim_memory_md, prim_engine, user_memory_md,
                    user_mem_engine, NULL));
        }
        CHECK(dnnl_primitive_create(reorder, reorder_pd));
        CHECK(dnnl_primitive_desc_destroy(reorder_pd));

        net[*net_index] = *reorder;
        prepare_arg_node(&net_args[*net_index], 2);
        set_arg(&net_args[*net_index].args[0], DNNL_ARG_FROM,
                dir_is_user_to_prim ? *user_memory : *prim_memory);
        set_arg(&net_args[*net_index].args[1], DNNL_ARG_TO,
                dir_is_user_to_prim ? *prim_memory : *user_memory);
        (*net_index)++;
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return dnnl_success;
}

/**
  * 函数功能：打印一个矩阵
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
void printMatrix(float *data, uint32_t dim, uint32_t batch, uint32_t c,
        uint32_t h, uint32_t w)
{
    printf("__________________________________________________________________"
    "______________\r\n");

    if(dim == 2)
    {
        for(uint32_t in = 0; in < batch; in++)
        {
            for(uint32_t ic = 0; ic < c; ic++)
            {
                uint32_t index = in*c+ ic;
                printf("%f //", data[index]);
            }
            printf("\r\n");
        }
    }
    else if(dim == 3)
    {
        for(uint32_t in = 0; in < batch; in++)
        {
            for(uint32_t ic = 0; ic < c; ic++)
            {
                for(uint32_t iw = 0; iw < w; iw++)
                {
                    uint32_t index = in*c*w+ ic*w + iw;
                    printf("%f //", data[index]);
                }
                printf("\r\n");
            }
            printf("____\r\n");
        }
    }
    else
    {
        printf("error printing matrix!\r\n");
    }

    printf("__________________________________________________________________"
    "______________\r\n");
}

/**
  * 函数功能：将一个矩阵输出到文档
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
void fprintMatrix(FILE *fp, float *data, uint32_t dim, uint32_t batch,
        uint32_t c, uint32_t h, uint32_t w)
{
    fprintf(fp, "_____________________________________________________________"
    "___________________\r\n");

    if(dim == 2)
    {
        for(uint32_t in = 0; in < batch; in++)
        {
            for(uint32_t ic = 0; ic < c; ic++)
            {
                uint32_t index = in*c+ ic;
                fprintf(fp, "%8.4f ", data[index]);
            }
            fprintf(fp, "\r\n");
        }
    }
    else if(dim == 3)
    {
        for(uint32_t in = 0; in < batch; in++)
        {
            for(uint32_t ic = 0; ic < c; ic++)
            {
                for(uint32_t iw = 0; iw < w; iw++)
                {
                    uint32_t index = in*c*w+ ic*w + iw;
                    fprintf(fp, "%8.4f ", data[index]);
                }
                fprintf(fp, "\r\n");
            }
            fprintf(fp, "____\r\n");
        }
    }
    else
    {
        fprintf(fp, "error printing matrix!\r\n");
    }

    fprintf(fp, "_____________________________________________________________"
    "___________________\r\n");
}

/**
  * 函数功能：随机初始化Worker空间
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
void RLInitWorker(_RL_worker *Worker)
{
    // 声明空间维度
    dnnl_dims_t conv1_src_sizes = {BATCH, IN_CONV1_C, 1, IN_CONV1_W};
    dnnl_dims_t conv1_weights_sizes = {CONV1_RELU1_C, IN_CONV1_C, 1,
            CONV1_KERNEL_W};
    dnnl_dims_t conv1_bias_sizes = {CONV1_RELU1_C};
    dnnl_dims_t pool1_dst_sizes = {BATCH, POOL1_CONV2_C, 1, POOL1_CONV2_W};
    dnnl_dims_t conv2_weights_sizes = {CONV2_RELU2_C, POOL1_CONV2_C, 1,
            CONV2_KERNEL_W};
    dnnl_dims_t conv2_bias_sizes = {CONV2_RELU2_C};
    dnnl_dims_t pool2_dst_sizes = {BATCH, POOL2_CONV3_C, 1, POOL2_CONV3_W};
    dnnl_dims_t conv3_weights_sizes = {CONV3_RELU3_C, POOL2_CONV3_C, 1,
            CONV3_KERNEL_W};
    dnnl_dims_t conv3_bias_sizes = {CONV3_RELU3_C};
    dnnl_dims_t pool3_dst_sizes = {BATCH, POOL3_RSHA_C, 1, POOL3_RSHA_W};

    dnnl_dims_t dnseA1_src_sizes = {BATCH, RSHA_DNSEA1_DNSEC1_C};
    dnnl_dims_t dnseA1_weights_sizes = {DNSEA1_RELUA1_C, RSHA_DNSEA1_DNSEC1_C};
    dnnl_dims_t dnseA1_bias_sizes = {DNSEA1_RELUA1_C};
    dnnl_dims_t dnseA2_weights_sizes = {DNSEA2_RELUA2_C, RELUA1_DNSEA2_C};
    dnnl_dims_t dnseA2_bias_sizes = {DNSEA2_RELUA2_C};
    dnnl_dims_t dnseA3_weights_sizes = {DNSEA3_RELUA3_C, RELUA2_DNSEA3_C};
    dnnl_dims_t dnseA3_bias_sizes = {DNSEA3_RELUA3_C};
    dnnl_dims_t dnseA4_weights_sizes = {DNSEA4_RELUA4_C, RELUA3_DNSEA4_C};
    dnnl_dims_t dnseA4_bias_sizes = {DNSEA4_RELUA4_C};
    dnnl_dims_t dnseA5_weights_sizes = {DNSEA5_OUTA_C, RELUA4_DNSEA5_C};
    dnnl_dims_t dnseA5_bias_sizes = {DNSEA5_OUTA_C};

    dnnl_dims_t dnseC1_src_sizes = {BATCH, RSHA_DNSEA1_DNSEC1_C};
    dnnl_dims_t dnseC1_weights_sizes = {DNSEC1_RELUC1_C, RSHA_DNSEA1_DNSEC1_C};
    dnnl_dims_t dnseC1_bias_sizes = {DNSEC1_RELUC1_C};
    dnnl_dims_t dnseC2_weights_sizes = {DNSEC2_RELUC2_C, RELUC1_DNSEC2_C};
    dnnl_dims_t dnseC2_bias_sizes = {DNSEC2_RELUC2_C};
    dnnl_dims_t dnseC3_weights_sizes = {DNSEC3_RELUC3_C, RELUC2_DNSEC3_C};
    dnnl_dims_t dnseC3_bias_sizes = {DNSEC3_RELUC3_C};
    dnnl_dims_t dnseC4_weights_sizes = {DNSEC4_RELUC4_C, RELUC3_DNSEC4_C};
    dnnl_dims_t dnseC4_bias_sizes = {DNSEC4_RELUC4_C};
    dnnl_dims_t dnseC5_weights_sizes = {DNSEC5_OUTC_C, RELUC4_DNSEC5_C};
    dnnl_dims_t dnseC5_bias_sizes = {DNSEC5_OUTC_C};

    dnnl_dims_t dnseA5_diff_dst_sizes = {BATCH, DNSEA5_OUTA_C};

    dnnl_dims_t dnseC5_diff_dst_sizes = {BATCH, DNSEC5_OUTC_C};
    // 分配空间
    Worker -> conv1_src
            = (float *)malloc(product(conv1_src_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv1_weights
            = (float *)malloc(product(conv1_weights_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv1_bias
            = (float *)malloc(product(conv1_bias_sizes, 1) * sizeof(float));
    Worker -> pool1_dst
            = (float *)malloc(product(pool1_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv2_weights
            = (float *)malloc(product(conv2_weights_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv2_bias
            = (float *)malloc(product(conv2_bias_sizes, 1) * sizeof(float));
    Worker -> pool2_dst
            = (float *)malloc(product(pool2_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv3_weights
            = (float *)malloc(product(conv3_weights_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv3_bias
            = (float *)malloc(product(conv3_bias_sizes, 1) * sizeof(float));
    Worker -> pool3_dst
            = (float *)malloc(product(pool3_dst_sizes, CONV_DIMS)
                    * sizeof(float));

    Worker -> dnseA1_src
            = (float *)malloc(product(dnseA1_src_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA1_weights
            = (float *)malloc(product(dnseA1_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA1_bias
            = (float *)malloc(product(dnseA1_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA2_weights
            = (float *)malloc(product(dnseA2_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA2_bias
            = (float *)malloc(product(dnseA2_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA3_weights
            = (float *)malloc(product(dnseA3_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA3_bias
            = (float *)malloc(product(dnseA3_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA4_weights
            = (float *)malloc(product(dnseA4_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA4_bias
            = (float *)malloc(product(dnseA4_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA5_weights
            = (float *)malloc(product(dnseA5_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA5_bias
            = (float *)malloc(product(dnseA5_bias_sizes, 1) * sizeof(float));

    Worker -> dnseC1_src
            = (float *)malloc(product(dnseC1_src_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC1_weights
            = (float *)malloc(product(dnseC1_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC1_bias
            = (float *)malloc(product(dnseC1_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC2_weights
            = (float *)malloc(product(dnseC2_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC2_bias
            = (float *)malloc(product(dnseC2_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC3_weights
            = (float *)malloc(product(dnseC3_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC3_bias
            = (float *)malloc(product(dnseC3_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC4_weights
            = (float *)malloc(product(dnseC4_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC4_bias
            = (float *)malloc(product(dnseC4_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC5_weights
            = (float *)malloc(product(dnseC5_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC5_bias
            = (float *)malloc(product(dnseC5_bias_sizes, 1) * sizeof(float));

    Worker -> dnseA5_diff_dst
            = (float *)malloc(product(dnseA5_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA5_diff_bias
            = (float *)malloc(product(dnseA5_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA5_diff_weights
            = (float *)malloc(product(dnseA5_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA4_diff_bias
            = (float *)malloc(product(dnseA4_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA4_diff_weights
            = (float *)malloc(product(dnseA4_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA3_diff_bias
            = (float *)malloc(product(dnseA3_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA3_diff_weights
            = (float *)malloc(product(dnseA3_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA2_diff_bias
            = (float *)malloc(product(dnseA2_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA2_diff_weights
            = (float *)malloc(product(dnseA2_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseA1_diff_bias
            = (float *)malloc(product(dnseA1_bias_sizes, 1) * sizeof(float));
    Worker -> dnseA1_diff_weights
            = (float *)malloc(product(dnseA1_weights_sizes, DNSE_DIMS)
                    * sizeof(float));

    Worker -> dnseC5_diff_dst
            = (float *)malloc(product(dnseC5_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC5_diff_bias
            = (float *)malloc(product(dnseC5_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC5_diff_weights
            = (float *)malloc(product(dnseC5_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC4_diff_bias
            = (float *)malloc(product(dnseC4_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC4_diff_weights
            = (float *)malloc(product(dnseC4_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC3_diff_bias
            = (float *)malloc(product(dnseC3_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC3_diff_weights
            = (float *)malloc(product(dnseC3_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC2_diff_bias
            = (float *)malloc(product(dnseC2_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC2_diff_weights
            = (float *)malloc(product(dnseC2_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    Worker -> dnseC1_diff_bias
            = (float *)malloc(product(dnseC1_bias_sizes, 1) * sizeof(float));
    Worker -> dnseC1_diff_weights
            = (float *)malloc(product(dnseC1_weights_sizes, DNSE_DIMS)
                    * sizeof(float));

    Worker -> pool3_diff_dst
            = (float *)malloc(product(pool3_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv3_diff_bias
            = (float *)malloc(product(conv3_bias_sizes, 1) * sizeof(float));
    Worker -> conv3_diff_weights
            = (float *)malloc(product(conv3_weights_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> pool2_diff_dst
            = (float *)malloc(product(pool2_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv2_diff_bias
            = (float *)malloc(product(conv2_bias_sizes, 1) * sizeof(float));
    Worker -> conv2_diff_weights
            = (float *)malloc(product(conv2_weights_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> pool1_diff_dst
            = (float *)malloc(product(pool1_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    Worker -> conv1_diff_bias
            = (float *)malloc(product(conv1_bias_sizes, 1) * sizeof(float));
    Worker -> conv1_diff_weights
            = (float *)malloc(product(conv1_weights_sizes, CONV_DIMS)
                    * sizeof(float));
    // 填入随机数
    memset(Worker -> conv1_src, 0, 
            product(conv1_src_sizes, CONV_DIMS) * sizeof(float));
    init_net_data(Worker -> conv1_weights, CONV_DIMS,
            conv1_weights_sizes);
    init_net_data(Worker -> conv1_bias, 1, conv1_bias_sizes);
    memset(Worker -> pool1_dst, 0,
            product(pool1_dst_sizes, CONV_DIMS) * sizeof(float));
    init_net_data(Worker -> conv2_weights, CONV_DIMS,
            conv2_weights_sizes);
    init_net_data(Worker -> conv2_bias, 1, conv2_bias_sizes);
    memset(Worker -> pool2_dst, 0,
            product(pool2_dst_sizes, CONV_DIMS) * sizeof(float));
    init_net_data(Worker -> conv3_weights, CONV_DIMS,
            conv3_weights_sizes);
    init_net_data(Worker -> conv3_bias, 1, conv3_bias_sizes);
    memset(Worker -> pool3_dst, 0,
            product(pool3_dst_sizes, CONV_DIMS) * sizeof(float));
    init_net_data(Worker -> pool3_dst, CONV_DIMS, pool3_dst_sizes);

    memset(Worker -> dnseA1_src, 0,
            product(dnseA1_src_sizes, DNSE_DIMS) * sizeof(float));
    init_net_data(Worker -> dnseA1_weights, DNSE_DIMS,
            dnseA1_weights_sizes);
    init_net_data(Worker -> dnseA1_bias, 1, dnseA1_bias_sizes);
    init_net_data(Worker -> dnseA2_weights, DNSE_DIMS,
            dnseA2_weights_sizes);
    init_net_data(Worker -> dnseA2_bias, 1, dnseA2_bias_sizes);
    init_net_data(Worker -> dnseA3_weights, DNSE_DIMS,
            dnseA3_weights_sizes);
    init_net_data(Worker -> dnseA3_bias, 1, dnseA3_bias_sizes);
    init_net_data(Worker -> dnseA4_weights, DNSE_DIMS,
            dnseA4_weights_sizes);
    init_net_data(Worker -> dnseA4_bias, 1, dnseA4_bias_sizes);
    init_net_data(Worker -> dnseA5_weights, DNSE_DIMS,
            dnseA5_weights_sizes);
    init_net_data(Worker -> dnseA5_bias, 1, dnseA5_bias_sizes);

    memset(Worker -> dnseC1_src, 0,
            product(dnseC1_src_sizes, DNSE_DIMS) * sizeof(float));
    init_net_data(Worker -> dnseC1_weights, DNSE_DIMS,
            dnseC1_weights_sizes);
    init_net_data(Worker -> dnseC1_bias, 1, dnseC1_bias_sizes);
    init_net_data(Worker -> dnseC2_weights, DNSE_DIMS,
            dnseC2_weights_sizes);
    init_net_data(Worker -> dnseC2_bias, 1, dnseC2_bias_sizes);
    init_net_data(Worker -> dnseC3_weights, DNSE_DIMS,
            dnseC3_weights_sizes);
    init_net_data(Worker -> dnseC3_bias, 1, dnseC3_bias_sizes);
    init_net_data(Worker -> dnseC4_weights, DNSE_DIMS,
            dnseC4_weights_sizes);
    init_net_data(Worker -> dnseC4_bias, 1, dnseC4_bias_sizes);
    init_net_data(Worker -> dnseC5_weights, DNSE_DIMS,
            dnseC5_weights_sizes);
    init_net_data(Worker -> dnseC5_bias, 1, dnseC5_bias_sizes);

    memset(Worker -> dnseA5_diff_dst, 0,
            product(dnseA5_diff_dst_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseA5_diff_bias, 0,
            product(dnseA5_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseA5_diff_weights, 0,
            product(dnseA5_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseA4_diff_bias, 0,
            product(dnseA4_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseA4_diff_weights, 0,
            product(dnseA4_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseA3_diff_bias, 0,
            product(dnseA3_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseA3_diff_weights, 0,
            product(dnseA3_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseA2_diff_bias, 0,
            product(dnseA2_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseA2_diff_weights, 0,
            product(dnseA2_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseA1_diff_bias, 0,
            product(dnseA1_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseA1_diff_weights, 0,
            product(dnseA1_weights_sizes, DNSE_DIMS) * sizeof(float));

    memset(Worker -> dnseC5_diff_dst, 0,
            product(dnseC5_diff_dst_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseC5_diff_bias, 0,
            product(dnseC5_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseC5_diff_weights, 0,
            product(dnseC5_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseC4_diff_bias, 0,
            product(dnseC4_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseC4_diff_weights, 0,
            product(dnseC4_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseC3_diff_bias, 0,
            product(dnseC3_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseC3_diff_weights, 0,
            product(dnseC3_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseC2_diff_bias, 0,
            product(dnseC2_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseC2_diff_weights, 0,
            product(dnseC2_weights_sizes, DNSE_DIMS) * sizeof(float));
    memset(Worker -> dnseC1_diff_bias, 0,
            product(dnseC1_bias_sizes, 1) * sizeof(float));
    memset(Worker -> dnseC1_diff_weights, 0,
            product(dnseC1_weights_sizes, DNSE_DIMS) * sizeof(float));

    memset(Worker -> pool3_diff_dst, 0,
            product(pool3_dst_sizes, CONV_DIMS) * sizeof(float));
    memset(Worker -> conv3_diff_bias, 0,
            product(conv3_bias_sizes, 1) * sizeof(float));
    memset(Worker -> conv3_diff_weights, 0,
            product(conv3_weights_sizes, CONV_DIMS) * sizeof(float));
    memset(Worker -> pool2_diff_dst, 0,
            product(pool2_dst_sizes, CONV_DIMS) * sizeof(float));
    memset(Worker -> conv2_diff_bias, 0,
            product(conv2_bias_sizes, 1) * sizeof(float));
    memset(Worker -> conv2_diff_weights, 0,
            product(conv2_weights_sizes, CONV_DIMS) * sizeof(float));
    memset(Worker -> pool1_diff_dst, 0,
            product(pool1_dst_sizes, CONV_DIMS) * sizeof(float));
    memset(Worker -> conv1_diff_bias, 0,
            product(conv1_bias_sizes, 1) * sizeof(float));
    memset(Worker -> conv1_diff_weights, 0,
            product(conv1_weights_sizes, CONV_DIMS) * sizeof(float));
}

/**
  * 函数功能：执行一次Worker
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
void RLExecuteWorker(FILE *fp1, unsigned char threadIndex,
		unsigned long RLIter, unsigned int *tjtryDataIndex,
		_data_line *DataImputBlock, _motor_line *MotorZeroState,
		_motor_line *MotorSimuBlock, _RL_trajectory *trajectories,
		_RL_worker *Worker)
{
/*--------------------神经网络运行文档---------------------------------------*/
#ifdef RL_NET_OUTPUT
	char outputNNFile[128];
	sprintf(outputNNFile, "RL_Worker%d_Iter%d_Line%d.tmp", threadIndex, RLIter,
			*tjtryDataIndex);
	FILE *fp3 = NULL;
    fp3 = fopen(outputNNFile, "w");
    if(fp3 == NULL){printf("error opening outputFile!\r\n");}
#endif

/*--------------------初始化神经网络数据-------------------------------------*/
    // CNN Forward
    uint32_t n_cnn_fwd;
    dnnl_primitive_t net_cnn_fwd[32];
    args_t net_cnn_fwd_args[32];
    dnnl_dims_t conv1_src_sizes = {BATCH, IN_CONV1_C, 1, IN_CONV1_W};// conv1
    dnnl_dims_t conv1_weights_sizes = {CONV1_RELU1_C, IN_CONV1_C, 1,
            CONV1_KERNEL_W};
    dnnl_dims_t conv1_bias_sizes = {CONV1_RELU1_C};
    dnnl_dims_t conv1_dst_sizes = {BATCH, CONV1_RELU1_C, 1, CONV1_RELU1_W};
    dnnl_dims_t conv1_strides = {1, CONV1_STRIDE};
    dnnl_dims_t conv1_dilation = {0, CONV1_DILATION};
    dnnl_dims_t conv1_padding = {0, CONV1_PADDING};
    dnnl_memory_t conv1_src_memory, conv1_weights_memory, conv1_bias_memory,
            conv1_dst_memory;
    dnnl_memory_t conv1_user_src_memory, conv1_user_weights_memory;
    dnnl_memory_t conv1_internal_src_memory, conv1_internal_weights_memory;
    dnnl_primitive_t conv1_reorder_src, conv1_reorder_weights;
    dnnl_primitive_desc_t conv1_pd;
    dnnl_primitive_t conv1;
    dnnl_memory_t relu1_dst_memory;// relu1
    dnnl_primitive_desc_t relu1_pd;
    dnnl_primitive_t relu1;
    dnnl_dims_t pool1_dst_sizes = {BATCH, POOL1_CONV2_C, 1,
            POOL1_CONV2_W};// pool1
    dnnl_dims_t pool1_kernel = {1, POOL1_STRIDE};
    dnnl_dims_t pool1_strides = {1, POOL1_STRIDE};
    dnnl_dims_t pool1_padding = {0, POOL1_PADDING};
    dnnl_dims_t pool1_dilation = {0, POOL1_DILATION};
    dnnl_memory_t pool1_dst_memory;
    dnnl_memory_t pool1_user_dst_memory;
    dnnl_memory_t pool1_ws_memory;
    dnnl_memory_t pool1_internal_dst_memory;
    dnnl_primitive_t pool1_reorder_dst;
    dnnl_primitive_desc_t pool1_pd;
    dnnl_primitive_t pool1;
    dnnl_dims_t conv2_weights_sizes = {CONV2_RELU2_C, POOL1_CONV2_C, 1,
            CONV2_KERNEL_W};// conv2
    dnnl_dims_t conv2_bias_sizes = {CONV2_RELU2_C};
    dnnl_dims_t conv2_dst_sizes = {BATCH, CONV2_RELU2_C, 1, CONV2_RELU2_W};
    dnnl_dims_t conv2_strides = {1, CONV2_STRIDE};
    dnnl_dims_t conv2_dilation = {0, CONV2_DILATION};
    dnnl_dims_t conv2_padding = {0, CONV2_PADDING};
    dnnl_memory_t conv2_weights_memory, conv2_bias_memory, conv2_dst_memory;
    dnnl_memory_t conv2_user_weights_memory;
    dnnl_memory_t conv2_internal_weights_memory;
    dnnl_primitive_t conv2_reorder_weights;
    dnnl_primitive_desc_t conv2_pd;
    dnnl_primitive_t conv2;
    dnnl_memory_t relu2_dst_memory;// relu2
    dnnl_primitive_desc_t relu2_pd;
    dnnl_primitive_t relu2;
    dnnl_dims_t pool2_dst_sizes = {BATCH, POOL2_CONV3_C, 1,
            POOL2_CONV3_W};// pool2
    dnnl_dims_t pool2_kernel = {1, POOL2_STRIDE};
    dnnl_dims_t pool2_strides = {1, POOL2_STRIDE};
    dnnl_dims_t pool2_padding = {0, POOL2_PADDING};
    dnnl_dims_t pool2_dilation = {0, POOL2_DILATION};
    dnnl_memory_t pool2_dst_memory;
    dnnl_memory_t pool2_user_dst_memory;
    dnnl_memory_t pool2_ws_memory;
    dnnl_memory_t pool2_internal_dst_memory;
    dnnl_primitive_t pool2_reorder_dst;
    dnnl_primitive_desc_t pool2_pd;
    dnnl_primitive_t pool2;
    dnnl_dims_t conv3_weights_sizes = {CONV3_RELU3_C, POOL2_CONV3_C, 1,
            CONV3_KERNEL_W};// conv3
    dnnl_dims_t conv3_bias_sizes = {CONV3_RELU3_C};
    dnnl_dims_t conv3_dst_sizes = {BATCH, CONV3_RELU3_C, 1, CONV3_RELU3_W};
    dnnl_dims_t conv3_strides = {1, CONV3_STRIDE};
    dnnl_dims_t conv3_dilation = {0, CONV3_DILATION};
    dnnl_dims_t conv3_padding = {0, CONV3_PADDING};
    dnnl_memory_t conv3_weights_memory, conv3_bias_memory, conv3_dst_memory;
    dnnl_memory_t conv3_user_weights_memory;
    dnnl_memory_t conv3_internal_weights_memory;
    dnnl_primitive_t conv3_reorder_weights;
    dnnl_primitive_desc_t conv3_pd;
    dnnl_primitive_t conv3;
    dnnl_memory_t relu3_dst_memory;// relu3
    dnnl_primitive_desc_t relu3_pd;
    dnnl_primitive_t relu3;
    dnnl_dims_t pool3_dst_sizes = {BATCH, POOL3_RSHA_C, 1,
            POOL3_RSHA_W};// pool3
    dnnl_dims_t pool3_kernel = {1, POOL3_STRIDE};
    dnnl_dims_t pool3_strides = {1, POOL3_STRIDE};
    dnnl_dims_t pool3_padding = {0, POOL3_PADDING};
    dnnl_dims_t pool3_dilation = {0, POOL3_DILATION};
    dnnl_memory_t pool3_dst_memory;
    dnnl_memory_t pool3_user_dst_memory;
    dnnl_memory_t pool3_ws_memory;
    dnnl_memory_t pool3_internal_dst_memory;
    dnnl_primitive_t pool3_reorder_dst;
    dnnl_primitive_desc_t pool3_pd;
    dnnl_primitive_t pool3;
    // Actor Forward
    uint32_t n_actor_fwd;
    dnnl_primitive_t net_actor_fwd[16];
    args_t net_actor_fwd_args[16];
    dnnl_dims_t dnseA1_src_sizes = {BATCH, RSHA_DNSEA1_DNSEC1_C};//dense_actor1
    dnnl_dims_t dnseA1_weights_sizes = {DNSEA1_RELUA1_C, RSHA_DNSEA1_DNSEC1_C};
    dnnl_dims_t dnseA1_bias_sizes = {DNSEA1_RELUA1_C};
    dnnl_dims_t dnseA1_dst_sizes = {BATCH, DNSEA1_RELUA1_C};
    dnnl_memory_desc_t dnseA1_src_md, dnseA1_weights_md, dnseA1_bias_md,
            dnseA1_dst_md;
    dnnl_memory_t dnseA1_src_memory, dnseA1_weights_memory, dnseA1_bias_memory,
            dnseA1_dst_memory;
    dnnl_primitive_desc_t dnseA1_pd;
    dnnl_primitive_t dnseA1;
    dnnl_dims_t reluA1_dst_sizes = {BATCH, RELUA1_DNSEA2_C};// relu_actor1
    dnnl_memory_desc_t reluA1_dst_md;
    dnnl_memory_t reluA1_dst_memory;
    dnnl_primitive_desc_t reluA1_pd;
    dnnl_primitive_t reluA1;
    dnnl_dims_t dnseA2_weights_sizes = {DNSEA2_RELUA2_C,
            RELUA1_DNSEA2_C};// dense_actor2
    dnnl_dims_t dnseA2_bias_sizes = {DNSEA2_RELUA2_C};
    dnnl_dims_t dnseA2_dst_sizes = {BATCH, DNSEA2_RELUA2_C};
    dnnl_memory_desc_t dnseA2_weights_md, dnseA2_bias_md, dnseA2_dst_md;
    dnnl_memory_t dnseA2_weights_memory, dnseA2_bias_memory, dnseA2_dst_memory;
    dnnl_primitive_desc_t dnseA2_pd;
    dnnl_primitive_t dnseA2;
    dnnl_dims_t reluA2_dst_sizes = {BATCH, RELUA2_DNSEA3_C};// relu_actor2
    dnnl_memory_desc_t reluA2_dst_md;
    dnnl_memory_t reluA2_dst_memory;
    dnnl_primitive_desc_t reluA2_pd;
    dnnl_primitive_t reluA2;
    dnnl_dims_t dnseA3_weights_sizes = {DNSEA3_RELUA3_C,
            RELUA2_DNSEA3_C};// dense_actor3
    dnnl_dims_t dnseA3_bias_sizes = {DNSEA3_RELUA3_C};
    dnnl_dims_t dnseA3_dst_sizes = {BATCH, DNSEA3_RELUA3_C};
    dnnl_memory_desc_t dnseA3_weights_md, dnseA3_bias_md, dnseA3_dst_md;
    dnnl_memory_t dnseA3_weights_memory, dnseA3_bias_memory, dnseA3_dst_memory;
    dnnl_primitive_desc_t dnseA3_pd;
    dnnl_primitive_t dnseA3;
    dnnl_dims_t reluA3_dst_sizes = {BATCH, RELUA3_DNSEA4_C};// relu_actor3
    dnnl_memory_desc_t reluA3_dst_md;
    dnnl_memory_t reluA3_dst_memory;
    dnnl_primitive_desc_t reluA3_pd;
    dnnl_primitive_t reluA3;
    dnnl_dims_t dnseA4_weights_sizes = {DNSEA4_RELUA4_C,
            RELUA3_DNSEA4_C};// dense_actor4
    dnnl_dims_t dnseA4_bias_sizes = {DNSEA4_RELUA4_C};
    dnnl_dims_t dnseA4_dst_sizes = {BATCH, DNSEA4_RELUA4_C};
    dnnl_memory_desc_t dnseA4_weights_md, dnseA4_bias_md, dnseA4_dst_md;
    dnnl_memory_t dnseA4_weights_memory, dnseA4_bias_memory, dnseA4_dst_memory;
    dnnl_primitive_desc_t dnseA4_pd;
    dnnl_primitive_t dnseA4;
    dnnl_dims_t reluA4_dst_sizes = {BATCH, RELUA4_DNSEA5_C};// relu_actor4
    dnnl_memory_desc_t reluA4_dst_md;
    dnnl_memory_t reluA4_dst_memory;
    dnnl_primitive_desc_t reluA4_pd;
    dnnl_primitive_t reluA4;
    dnnl_dims_t dnseA5_weights_sizes = {DNSEA5_OUTA_C,
            RELUA4_DNSEA5_C};// dense_actor5
    dnnl_dims_t dnseA5_bias_sizes = {DNSEA5_OUTA_C};
    dnnl_dims_t dnseA5_dst_sizes = {BATCH, DNSEA5_OUTA_C};
    dnnl_memory_desc_t dnseA5_weights_md, dnseA5_bias_md, dnseA5_dst_md;
    dnnl_memory_t dnseA5_weights_memory, dnseA5_bias_memory, dnseA5_dst_memory;
    dnnl_primitive_desc_t dnseA5_pd;
    dnnl_primitive_t dnseA5;
    // Critic Forward
    uint32_t n_critic_fwd;
    dnnl_primitive_t net_critic_fwd[16];
    args_t net_critic_fwd_args[16];
    dnnl_dims_t dnseC1_src_sizes = {BATCH,
            RSHA_DNSEA1_DNSEC1_C};// dense_critic1
    dnnl_dims_t dnseC1_weights_sizes = {DNSEC1_RELUC1_C, RSHA_DNSEA1_DNSEC1_C};
    dnnl_dims_t dnseC1_bias_sizes = {DNSEC1_RELUC1_C};
    dnnl_dims_t dnseC1_dst_sizes = {BATCH, DNSEC1_RELUC1_C};
    dnnl_memory_desc_t dnseC1_src_md, dnseC1_weights_md, dnseC1_bias_md,
            dnseC1_dst_md;
    dnnl_memory_t dnseC1_src_memory, dnseC1_weights_memory, dnseC1_bias_memory,
            dnseC1_dst_memory;
    dnnl_primitive_desc_t dnseC1_pd;
    dnnl_primitive_t dnseC1;
    dnnl_dims_t reluC1_dst_sizes = {BATCH, RELUC1_DNSEC2_C};// relu_critic1
    dnnl_memory_desc_t reluC1_dst_md;
    dnnl_memory_t reluC1_dst_memory;
    dnnl_primitive_desc_t reluC1_pd;
    dnnl_primitive_t reluC1;
    dnnl_dims_t dnseC2_weights_sizes = {DNSEC2_RELUC2_C,
            RELUC1_DNSEC2_C};// dense_critic2
    dnnl_dims_t dnseC2_bias_sizes = {DNSEC2_RELUC2_C};
    dnnl_dims_t dnseC2_dst_sizes = {BATCH, DNSEC2_RELUC2_C};
    dnnl_memory_desc_t dnseC2_weights_md, dnseC2_bias_md, dnseC2_dst_md;
    dnnl_memory_t dnseC2_weights_memory, dnseC2_bias_memory, dnseC2_dst_memory;
    dnnl_primitive_desc_t dnseC2_pd;
    dnnl_primitive_t dnseC2;
    dnnl_dims_t reluC2_dst_sizes = {BATCH, RELUC2_DNSEC3_C};// relu_critic2
    dnnl_memory_desc_t reluC2_dst_md;
    dnnl_memory_t reluC2_dst_memory;
    dnnl_primitive_desc_t reluC2_pd;
    dnnl_primitive_t reluC2;
    dnnl_dims_t dnseC3_weights_sizes = {DNSEC3_RELUC3_C,
            RELUC2_DNSEC3_C};// dense_critic3
    dnnl_dims_t dnseC3_bias_sizes = {DNSEC3_RELUC3_C};
    dnnl_dims_t dnseC3_dst_sizes = {BATCH, DNSEC3_RELUC3_C};
    dnnl_memory_desc_t dnseC3_weights_md, dnseC3_bias_md, dnseC3_dst_md;
    dnnl_memory_t dnseC3_weights_memory, dnseC3_bias_memory, dnseC3_dst_memory;
    dnnl_primitive_desc_t dnseC3_pd;
    dnnl_primitive_t dnseC3;
    dnnl_dims_t reluC3_dst_sizes = {BATCH, RELUC3_DNSEC4_C};// relu_critic3
    dnnl_memory_desc_t reluC3_dst_md;
    dnnl_memory_t reluC3_dst_memory;
    dnnl_primitive_desc_t reluC3_pd;
    dnnl_primitive_t reluC3;
    dnnl_dims_t dnseC4_weights_sizes = {DNSEC4_RELUC4_C,
            RELUC3_DNSEC4_C};// dense_critic4
    dnnl_dims_t dnseC4_bias_sizes = {DNSEC4_RELUC4_C};
    dnnl_dims_t dnseC4_dst_sizes = {BATCH, DNSEC4_RELUC4_C};
    dnnl_memory_desc_t dnseC4_weights_md, dnseC4_bias_md, dnseC4_dst_md;
    dnnl_memory_t dnseC4_weights_memory, dnseC4_bias_memory, dnseC4_dst_memory;
    dnnl_primitive_desc_t dnseC4_pd;
    dnnl_primitive_t dnseC4;
    dnnl_dims_t reluC4_dst_sizes = {BATCH, RELUC4_DNSEC5_C};// relu_critic4
    dnnl_memory_desc_t reluC4_dst_md;
    dnnl_memory_t reluC4_dst_memory;
    dnnl_primitive_desc_t reluC4_pd;
    dnnl_primitive_t reluC4;
    dnnl_dims_t dnseC5_weights_sizes = {DNSEC5_OUTC_C,
            RELUC4_DNSEC5_C};// dense_critic5
    dnnl_dims_t dnseC5_bias_sizes = {DNSEC5_OUTC_C};
    dnnl_dims_t dnseC5_dst_sizes = {BATCH, DNSEC5_OUTC_C};
    dnnl_memory_desc_t dnseC5_weights_md, dnseC5_bias_md, dnseC5_dst_md;
    dnnl_memory_t dnseC5_weights_memory, dnseC5_bias_memory, dnseC5_dst_memory;
    dnnl_primitive_desc_t dnseC5_pd;
    dnnl_primitive_t dnseC5;
    // Actor Backward
    uint32_t n_actor_bwd;
    dnnl_primitive_t net_actor_bwd[16];
    args_t net_actor_bwd_args[16];
    dnnl_memory_desc_t dnseA5_diff_dst_md, dnseA5_diff_weights_md,
            dnseA5_diff_bias_md, dnseA5_diff_src_md;// dense_actor5_backward
    dnnl_memory_t dnseA5_diff_dst_memory, dnseA5_diff_weights_memory,
            dnseA5_diff_bias_memory, dnseA5_diff_src_memory;
    dnnl_primitive_desc_t dnseA5_bwd_data_pd, dnseA5_bwd_weights_pd;
    dnnl_primitive_t dnseA5_bwd_data, dnseA5_bwd_weights;
    dnnl_memory_desc_t reluA4_diff_src_md;// relu_actor4_backward
    dnnl_memory_t reluA4_diff_src_memory;
    dnnl_primitive_desc_t reluA4_bwd_pd;
    dnnl_primitive_t reluA4_bwd;
    dnnl_memory_desc_t dnseA4_diff_weights_md, dnseA4_diff_bias_md,
            dnseA4_diff_src_md;// dense_actor4_backward
    dnnl_memory_t dnseA4_diff_weights_memory, dnseA4_diff_bias_memory,
            dnseA4_diff_src_memory;
    dnnl_primitive_desc_t dnseA4_bwd_data_pd, dnseA4_bwd_weights_pd;
    dnnl_primitive_t dnseA4_bwd_data, dnseA4_bwd_weights;
    dnnl_memory_desc_t reluA3_diff_src_md;// relu_actor3_backward
    dnnl_memory_t reluA3_diff_src_memory;
    dnnl_primitive_desc_t reluA3_bwd_pd;
    dnnl_primitive_t reluA3_bwd;
    dnnl_memory_desc_t dnseA3_diff_weights_md, dnseA3_diff_bias_md,
            dnseA3_diff_src_md;// dense_actor3_backward
    dnnl_memory_t dnseA3_diff_weights_memory, dnseA3_diff_bias_memory,
            dnseA3_diff_src_memory;
    dnnl_primitive_desc_t dnseA3_bwd_data_pd, dnseA3_bwd_weights_pd;
    dnnl_primitive_t dnseA3_bwd_data, dnseA3_bwd_weights;
    dnnl_memory_desc_t reluA2_diff_src_md;// relu_actor2_backward
    dnnl_memory_t reluA2_diff_src_memory;
    dnnl_primitive_desc_t reluA2_bwd_pd;
    dnnl_primitive_t reluA2_bwd;
    dnnl_memory_desc_t dnseA2_diff_weights_md, dnseA2_diff_bias_md,
            dnseA2_diff_src_md;// dense_actor2_backward
    dnnl_memory_t dnseA2_diff_weights_memory, dnseA2_diff_bias_memory,
            dnseA2_diff_src_memory;
    dnnl_primitive_desc_t dnseA2_bwd_data_pd, dnseA2_bwd_weights_pd;
    dnnl_primitive_t dnseA2_bwd_data, dnseA2_bwd_weights;
    dnnl_memory_desc_t reluA1_diff_src_md;// relu_actor1_backward
    dnnl_memory_t reluA1_diff_src_memory;
    dnnl_primitive_desc_t reluA1_bwd_pd;
    dnnl_primitive_t reluA1_bwd;
    dnnl_memory_desc_t dnseA1_diff_weights_md, dnseA1_diff_bias_md,
            dnseA1_diff_src_md;// dense_actor1_backward
    dnnl_memory_t dnseA1_diff_weights_memory, dnseA1_diff_bias_memory,
            dnseA1_diff_src_memory;
    dnnl_primitive_desc_t dnseA1_bwd_data_pd, dnseA1_bwd_weights_pd;
    dnnl_primitive_t dnseA1_bwd_data, dnseA1_bwd_weights;
    // Critic Backward
    uint32_t n_critic_bwd;
    dnnl_primitive_t net_critic_bwd[16];
    args_t net_critic_bwd_args[16];
    dnnl_memory_desc_t dnseC5_diff_dst_md, dnseC5_diff_weights_md,
            dnseC5_diff_bias_md, dnseC5_diff_src_md;// dense_critic5_backward
    dnnl_memory_t dnseC5_diff_dst_memory, dnseC5_diff_weights_memory,
            dnseC5_diff_bias_memory, dnseC5_diff_src_memory;
    dnnl_primitive_desc_t dnseC5_bwd_data_pd, dnseC5_bwd_weights_pd;
    dnnl_primitive_t dnseC5_bwd_data, dnseC5_bwd_weights;
    dnnl_memory_desc_t reluC4_diff_src_md;// relu_critic4_backward
    dnnl_memory_t reluC4_diff_src_memory;
    dnnl_primitive_desc_t reluC4_bwd_pd;
    dnnl_primitive_t reluC4_bwd;
    dnnl_memory_desc_t dnseC4_diff_weights_md, dnseC4_diff_bias_md,
            dnseC4_diff_src_md;// dense_critic4_backward
    dnnl_memory_t dnseC4_diff_weights_memory, dnseC4_diff_bias_memory,
            dnseC4_diff_src_memory;
    dnnl_primitive_desc_t dnseC4_bwd_data_pd, dnseC4_bwd_weights_pd;
    dnnl_primitive_t dnseC4_bwd_data, dnseC4_bwd_weights;
    dnnl_memory_desc_t reluC3_diff_src_md;// relu_critic3_backward
    dnnl_memory_t reluC3_diff_src_memory;
    dnnl_primitive_desc_t reluC3_bwd_pd;
    dnnl_primitive_t reluC3_bwd;
    dnnl_memory_desc_t dnseC3_diff_weights_md, dnseC3_diff_bias_md,
            dnseC3_diff_src_md;// dense_critic3_backward
    dnnl_memory_t dnseC3_diff_weights_memory, dnseC3_diff_bias_memory,
            dnseC3_diff_src_memory;
    dnnl_primitive_desc_t dnseC3_bwd_data_pd, dnseC3_bwd_weights_pd;
    dnnl_primitive_t dnseC3_bwd_data, dnseC3_bwd_weights;
    dnnl_memory_desc_t reluC2_diff_src_md;// relu_critic2_backward
    dnnl_memory_t reluC2_diff_src_memory;
    dnnl_primitive_desc_t reluC2_bwd_pd;
    dnnl_primitive_t reluC2_bwd;
    dnnl_memory_desc_t dnseC2_diff_weights_md, dnseC2_diff_bias_md,
            dnseC2_diff_src_md;// dense_critic2_backward
    dnnl_memory_t dnseC2_diff_weights_memory, dnseC2_diff_bias_memory,
            dnseC2_diff_src_memory;
    dnnl_primitive_desc_t dnseC2_bwd_data_pd, dnseC2_bwd_weights_pd;
    dnnl_primitive_t dnseC2_bwd_data, dnseC2_bwd_weights;
    dnnl_memory_desc_t reluC1_diff_src_md;// relu_critic1_backward
    dnnl_memory_t reluC1_diff_src_memory;
    dnnl_primitive_desc_t reluC1_bwd_pd;
    dnnl_primitive_t reluC1_bwd;
    dnnl_memory_desc_t dnseC1_diff_weights_md, dnseC1_diff_bias_md,
            dnseC1_diff_src_md;// dense_critic1_backward
    dnnl_memory_t dnseC1_diff_weights_memory, dnseC1_diff_bias_memory,
            dnseC1_diff_src_memory;
    dnnl_primitive_desc_t dnseC1_bwd_data_pd, dnseC1_bwd_weights_pd;
    dnnl_primitive_t dnseC1_bwd_data, dnseC1_bwd_weights;
    // CNN Backward
    uint32_t n_cnn_bwd;
    dnnl_primitive_t net_cnn_bwd[32];
    args_t net_cnn_bwd_args[32];
    dnnl_memory_t pool3_diff_dst_memory, pool3_diff_src_memory;//pool3_backward
    dnnl_memory_t pool3_user_diff_dst_memory;
    dnnl_memory_t pool3_internal_diff_dst_memory;
    dnnl_primitive_t pool3_reorder_diff_dst;
    dnnl_primitive_desc_t pool3_bwd_pd;
    dnnl_primitive_t pool3_bwd;
    dnnl_memory_t relu3_diff_src_memory;// relu3_backward
    dnnl_primitive_desc_t relu3_bwd_pd;
    dnnl_primitive_t relu3_bwd;
    dnnl_memory_t conv3_bwd_src_memory, conv3_bwd_weights_memory,
            conv3_diff_dst_memory, conv3_diff_weights_memory,
            conv3_diff_bias_memory, conv3_diff_src_memory;// conv3_backward
    dnnl_memory_t conv3_user_diff_weights_memory, conv3_user_diff_src_memory;
    dnnl_memory_t conv3_internal_bwd_src_memory,
            conv3_internal_bwd_weights_memory,
            conv3_internal_diff_dst_memory, conv3_internal_diff_weights_memory,
            conv3_internal_diff_src_memory;
    dnnl_primitive_t conv3_bwd_reorder_src, conv3_bwd_reorder_weights,
            conv3_reorder_diff_dst, conv3_reorder_diff_weights,
            conv3_reorder_diff_src;
    dnnl_primitive_desc_t conv3_bwd_data_pd, conv3_bwd_weights_pd;
    dnnl_primitive_t conv3_bwd_data, conv3_bwd_weights;
    dnnl_memory_t pool2_diff_dst_memory, pool2_diff_src_memory;//pool2_backward
    dnnl_memory_t pool2_internal_diff_dst_memory;
    dnnl_primitive_t pool2_reorder_diff_dst;
    dnnl_primitive_desc_t pool2_bwd_pd;
    dnnl_primitive_t pool2_bwd;
    dnnl_memory_t relu2_diff_src_memory;// relu2_backward
    dnnl_primitive_desc_t relu2_bwd_pd;
    dnnl_primitive_t relu2_bwd;
    dnnl_memory_t conv2_bwd_src_memory, conv2_bwd_weights_memory,
            conv2_diff_dst_memory, conv2_diff_weights_memory,
            conv2_diff_bias_memory, conv2_diff_src_memory;// conv2_backward
    dnnl_memory_t conv2_user_diff_weights_memory, conv2_user_diff_src_memory;
    dnnl_memory_t conv2_internal_bwd_src_memory,
            conv2_internal_bwd_weights_memory,
            conv2_internal_diff_dst_memory, conv2_internal_diff_weights_memory,
            conv2_internal_diff_src_memory;
    dnnl_primitive_t conv2_bwd_reorder_src, conv2_bwd_reorder_weights,
            conv2_reorder_diff_dst, conv2_reorder_diff_weights,
            conv2_reorder_diff_src;
    dnnl_primitive_desc_t conv2_bwd_data_pd, conv2_bwd_weights_pd;
    dnnl_primitive_t conv2_bwd_data, conv2_bwd_weights;
    dnnl_memory_t pool1_diff_dst_memory, pool1_diff_src_memory;//pool1_backward
    dnnl_memory_t pool1_internal_diff_dst_memory;
    dnnl_primitive_t pool1_reorder_diff_dst;
    dnnl_primitive_desc_t pool1_bwd_pd;
    dnnl_primitive_t pool1_bwd;
    dnnl_memory_t relu1_diff_src_memory;// relu1_backward
    dnnl_primitive_desc_t relu1_bwd_pd;
    dnnl_primitive_t relu1_bwd;
    dnnl_memory_t conv1_bwd_src_memory, conv1_bwd_weights_memory,
            conv1_diff_dst_memory, conv1_diff_weights_memory,
            conv1_diff_bias_memory, conv1_diff_src_memory;// conv1_backward
    dnnl_memory_t conv1_user_diff_weights_memory, conv1_user_diff_src_memory;
    dnnl_memory_t conv1_internal_bwd_src_memory,
            conv1_internal_bwd_weights_memory,
            conv1_internal_diff_dst_memory, conv1_internal_diff_weights_memory,
            conv1_internal_diff_src_memory;
    dnnl_primitive_t conv1_bwd_reorder_src, conv1_bwd_reorder_weights,
            conv1_reorder_diff_dst, conv1_reorder_diff_weights,
            conv1_reorder_diff_src;
    dnnl_primitive_desc_t conv1_bwd_data_pd, conv1_bwd_weights_pd;
    dnnl_primitive_t conv1_bwd_data, conv1_bwd_weights;
	
/*--------------------创建神经网络模型---------------------------------------*/
    CHECK(dnnl_engine_create(&(Worker -> engine), dnnl_cpu, 0));
    CHECK(dnnl_stream_create(&(Worker -> stream),
            Worker -> engine, dnnl_stream_default_flags));

    /*----------------- CNN Forward Stream ----------------------------------*/
    n_cnn_fwd = 0;

    // conv1
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, conv1_src_sizes, dnnl_nchw,
            Worker -> engine, Worker -> conv1_src,
            &conv1_user_src_memory);
    init_data_memory(CONV_DIMS, conv1_weights_sizes, dnnl_oihw,
            Worker -> engine, Worker -> conv1_weights,
            &conv1_user_weights_memory);
    init_data_memory(1, conv1_bias_sizes, dnnl_x, Worker -> engine,
            Worker -> conv1_bias, &conv1_bias_memory);
    
    // 建立conv1原型描述符
    {
        // create data descriptors for convolution w/ no specified format
        dnnl_memory_desc_t conv1_sys_src_md, conv1_sys_weights_md,
                conv1_sys_bias_md, conv1_sys_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_src_md, CONV_DIMS,
                conv1_src_sizes, dnnl_f32, dnnl_nchw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_weights_md,
                CONV_DIMS, conv1_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(
                &conv1_sys_bias_md, 1, conv1_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_dst_md, CONV_DIMS,
                conv1_dst_sizes, dnnl_f32, dnnl_nchw));

        CHECK(dnnl_convolution_forward_primitive_desc_create(&conv1_pd,
                Worker -> engine, dnnl_forward,
                dnnl_convolution_direct, conv1_sys_src_md,
                conv1_sys_weights_md, conv1_sys_bias_md, conv1_sys_dst_md,
                conv1_strides, conv1_dilation, conv1_padding, conv1_padding,
                NULL));

        CHECK(dnnl_memory_desc_destroy(conv1_sys_src_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_dst_md));
    }
    
    // 创建结果存储器
    const_dnnl_memory_desc_t conv1_dst_md
            = dnnl_primitive_desc_query_md(conv1_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv1_dst_memory, conv1_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 评估并创建重定位
    const_dnnl_memory_desc_t conv1_src_md
            = dnnl_primitive_desc_query_md(conv1_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&conv1_user_src_memory, conv1_src_md,
            Worker -> engine, 1, &conv1_internal_src_memory,
            &conv1_reorder_src, &n_cnn_fwd, net_cnn_fwd, net_cnn_fwd_args));

    const_dnnl_memory_desc_t conv1_weights_md
            = dnnl_primitive_desc_query_md(conv1_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv1_user_weights_memory, conv1_weights_md,
            Worker -> engine, 1, &conv1_internal_weights_memory,
            &conv1_reorder_weights, &n_cnn_fwd, net_cnn_fwd,
            net_cnn_fwd_args));

    conv1_src_memory = conv1_internal_src_memory
            ? conv1_internal_src_memory
            : conv1_user_src_memory;
    conv1_weights_memory = conv1_internal_weights_memory
            ? conv1_internal_weights_memory
            : conv1_user_weights_memory;
    
    // 建立conv1原型
    CHECK(dnnl_primitive_create(&conv1, conv1_pd));
    net_cnn_fwd[n_cnn_fwd] = conv1;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 4);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            conv1_src_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv1_weights_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[2], DNNL_ARG_BIAS,
            conv1_bias_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[3], DNNL_ARG_DST,
            conv1_dst_memory);
    n_cnn_fwd++;

    // relu1
    // 建立relu1原型描述符
    const_dnnl_memory_desc_t relu1_src_md = conv1_dst_md;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu1_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, relu1_src_md, RELU1_NSLOP, 0, NULL));
    
    // 创造数据存储器
    const_dnnl_memory_desc_t relu1_dst_md
            = dnnl_primitive_desc_query_md(relu1_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&relu1_dst_memory, relu1_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立relu1原型
    CHECK(dnnl_primitive_create(&relu1, relu1_pd));
    net_cnn_fwd[n_cnn_fwd] = relu1;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 2);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            conv1_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_DST,
            relu1_dst_memory);
    n_cnn_fwd++;

    // pool1
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, pool1_dst_sizes, dnnl_nchw,
            Worker -> engine, Worker -> pool1_dst,
            &pool1_user_dst_memory);

    // 建立pool1原型描述符
    {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive
        const_dnnl_memory_desc_t pool1_sys_src_md = relu1_dst_md;

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool1_sys_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool1_sys_dst_md, 4,
                pool1_dst_sizes, dnnl_f32, dnnl_nchw));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool1_pd,
                Worker -> engine, dnnl_forward, dnnl_pooling_max,
                pool1_sys_src_md, pool1_sys_dst_md, pool1_strides,
                pool1_kernel, pool1_dilation, pool1_padding, pool1_padding,
                NULL));
        CHECK(dnnl_memory_desc_destroy(pool1_sys_dst_md));
    }

    // 创造workSpace存储器
    const_dnnl_memory_desc_t pool1_ws_md = dnnl_primitive_desc_query_md(
            pool1_pd, dnnl_query_workspace_md, 0);
    CHECK(dnnl_memory_create(&pool1_ws_memory, pool1_ws_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    
    // 评估并创建重定位
    const_dnnl_memory_desc_t pool1_dst_md
            = dnnl_primitive_desc_query_md(pool1_pd, dnnl_query_dst_md, 0);
    n_cnn_fwd += 1; // tentative workaround: preserve space for pooling that
                    // should happen before the reorder
    CHECK(prepare_reorder(&pool1_user_dst_memory, pool1_dst_md,
            Worker -> engine, 0, &pool1_internal_dst_memory,
            &pool1_reorder_dst, &n_cnn_fwd, net_cnn_fwd, net_cnn_fwd_args));
    n_cnn_fwd -= pool1_reorder_dst ? 2 : 1;

    pool1_dst_memory = pool1_internal_dst_memory
            ? pool1_internal_dst_memory
            : pool1_user_dst_memory;

    // 建立pool1原型
    CHECK(dnnl_primitive_create(&pool1, pool1_pd));
    net_cnn_fwd[n_cnn_fwd] = pool1;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 3);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            relu1_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_DST,
            pool1_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[2], DNNL_ARG_WORKSPACE,
            pool1_ws_memory);
    n_cnn_fwd++;

    if (pool1_reorder_dst) n_cnn_fwd += 1;

    // conv2
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, conv2_weights_sizes, dnnl_oihw,
            Worker -> engine, Worker -> conv2_weights,
            &conv2_user_weights_memory);
    init_data_memory(1, conv2_bias_sizes, dnnl_x, Worker -> engine,
            Worker -> conv2_bias, &conv2_bias_memory);

    // 建立conv2原型描述符
    {
        // create data descriptors for convolution w/ no specified format
        const_dnnl_memory_desc_t conv2_sys_src_md = pool1_dst_md;

        dnnl_memory_desc_t conv2_sys_weights_md, conv2_sys_bias_md,
                conv2_sys_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_weights_md,
                CONV_DIMS, conv2_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(
                &conv2_sys_bias_md, 1, conv2_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_dst_md, CONV_DIMS,
                conv2_dst_sizes, dnnl_f32, dnnl_nchw));

        CHECK(dnnl_convolution_forward_primitive_desc_create(&conv2_pd,
                Worker -> engine, dnnl_forward,
                dnnl_convolution_direct, conv2_sys_src_md,
                conv2_sys_weights_md, conv2_sys_bias_md, conv2_sys_dst_md,
                conv2_strides, conv2_dilation, conv2_padding, conv2_padding,
                NULL));

        CHECK(dnnl_memory_desc_destroy(conv2_sys_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv2_sys_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv2_sys_dst_md));
    }

    // 创建结果存储器
    const_dnnl_memory_desc_t conv2_dst_md
            = dnnl_primitive_desc_query_md(conv2_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv2_dst_memory, conv2_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 评估并创建重定位
    const_dnnl_memory_desc_t conv2_weights_md
            = dnnl_primitive_desc_query_md(conv2_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv2_user_weights_memory, conv2_weights_md,
            Worker -> engine, 1, &conv2_internal_weights_memory,
            &conv2_reorder_weights, &n_cnn_fwd, net_cnn_fwd,
            net_cnn_fwd_args));

    conv2_weights_memory = conv2_internal_weights_memory
            ? conv2_internal_weights_memory
            : conv2_user_weights_memory;

    // 建立conv2原型
    CHECK(dnnl_primitive_create(&conv2, conv2_pd));
    net_cnn_fwd[n_cnn_fwd] = conv2;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 4);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            pool1_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv2_weights_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[2], DNNL_ARG_BIAS,
            conv2_bias_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[3], DNNL_ARG_DST,
            conv2_dst_memory);
    n_cnn_fwd++;

    // relu2
    // 建立relu2原型描述符
    const_dnnl_memory_desc_t relu2_src_md = conv2_dst_md;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu2_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, relu2_src_md, RELU2_NSLOP, 0, NULL));

    // 创造数据存储器
    const_dnnl_memory_desc_t relu2_dst_md
            = dnnl_primitive_desc_query_md(relu2_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&relu2_dst_memory, relu2_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立relu2原型
    CHECK(dnnl_primitive_create(&relu2, relu2_pd));
    net_cnn_fwd[n_cnn_fwd] = relu2;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 2);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            conv2_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_DST,
            relu2_dst_memory);
    n_cnn_fwd++;

    // pool2
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, pool2_dst_sizes, dnnl_nchw,
            Worker -> engine, Worker -> pool2_dst,
            &pool2_user_dst_memory);

    // 建立pool2原型描述符
    {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive
        const_dnnl_memory_desc_t pool2_sys_src_md = relu2_dst_md;

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool2_sys_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool2_sys_dst_md, 4,
                pool2_dst_sizes, dnnl_f32, dnnl_nchw));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool2_pd,
                Worker -> engine, dnnl_forward, dnnl_pooling_max,
                pool2_sys_src_md, pool2_sys_dst_md, pool2_strides,
                pool2_kernel, pool2_dilation, pool2_padding, pool2_padding,
                NULL));
        CHECK(dnnl_memory_desc_destroy(pool2_sys_dst_md));
    }

    // 创造workSpace存储器
    const_dnnl_memory_desc_t pool2_ws_md = dnnl_primitive_desc_query_md(
            pool2_pd, dnnl_query_workspace_md, 0);
    CHECK(dnnl_memory_create(&pool2_ws_memory, pool2_ws_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 评估并创建重定位
    const_dnnl_memory_desc_t pool2_dst_md
            = dnnl_primitive_desc_query_md(pool2_pd, dnnl_query_dst_md, 0);
    n_cnn_fwd += 1; // tentative workaround: preserve space for pooling that
                    // should happen before the reorder
    CHECK(prepare_reorder(&pool2_user_dst_memory, pool2_dst_md,
            Worker -> engine, 0, &pool2_internal_dst_memory,
            &pool2_reorder_dst, &n_cnn_fwd, net_cnn_fwd, net_cnn_fwd_args));
    n_cnn_fwd -= pool2_reorder_dst ? 2 : 1;

    pool2_dst_memory = pool2_internal_dst_memory
            ? pool2_internal_dst_memory
            : pool2_user_dst_memory;

    // 建立pool2原型
    CHECK(dnnl_primitive_create(&pool2, pool2_pd));
    net_cnn_fwd[n_cnn_fwd] = pool2;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 3);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            relu2_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_DST,
            pool2_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[2], DNNL_ARG_WORKSPACE,
            pool2_ws_memory);
    n_cnn_fwd++;

    if (pool2_reorder_dst) n_cnn_fwd += 1;

    // conv3
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, conv3_weights_sizes, dnnl_oihw,
            Worker -> engine, Worker -> conv3_weights,
            &conv3_user_weights_memory);
    init_data_memory(1, conv3_bias_sizes, dnnl_x, Worker -> engine,
            Worker -> conv3_bias, &conv3_bias_memory);

    // 建立conv3原型描述符
    {
        // create data descriptors for convolution w/ no specified format
        const_dnnl_memory_desc_t conv3_sys_src_md = pool2_dst_md;

        dnnl_memory_desc_t conv3_sys_weights_md, conv3_sys_bias_md,
                conv3_sys_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_weights_md,
                CONV_DIMS, conv3_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(
                &conv3_sys_bias_md, 1, conv3_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_dst_md, CONV_DIMS,
                conv3_dst_sizes, dnnl_f32, dnnl_nchw));

        CHECK(dnnl_convolution_forward_primitive_desc_create(&conv3_pd,
                Worker -> engine, dnnl_forward,
                dnnl_convolution_direct, conv3_sys_src_md,
                conv3_sys_weights_md, conv3_sys_bias_md, conv3_sys_dst_md,
                conv3_strides, conv3_dilation, conv3_padding, conv3_padding,
                NULL));

        CHECK(dnnl_memory_desc_destroy(conv3_sys_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv3_sys_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv3_sys_dst_md));
    }

    // 创建结果存储器
    const_dnnl_memory_desc_t conv3_dst_md
            = dnnl_primitive_desc_query_md(conv3_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv3_dst_memory, conv3_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 评估并创建重定位
    const_dnnl_memory_desc_t conv3_weights_md
            = dnnl_primitive_desc_query_md(conv3_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv3_user_weights_memory, conv3_weights_md,
            Worker -> engine, 1, &conv3_internal_weights_memory,
            &conv3_reorder_weights, &n_cnn_fwd, net_cnn_fwd,
            net_cnn_fwd_args));

    conv3_weights_memory = conv3_internal_weights_memory
            ? conv3_internal_weights_memory
            : conv3_user_weights_memory;

    // 建立conv3原型
    CHECK(dnnl_primitive_create(&conv3, conv3_pd));
    net_cnn_fwd[n_cnn_fwd] = conv3;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 4);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            pool2_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv3_weights_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[2], DNNL_ARG_BIAS,
            conv3_bias_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[3], DNNL_ARG_DST,
            conv3_dst_memory);
    n_cnn_fwd++;

    // relu3
    // 建立relu3原型描述符
    const_dnnl_memory_desc_t relu3_src_md = conv3_dst_md;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu3_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, relu3_src_md, RELU3_NSLOP, 0, NULL));

    // 创造数据存储器
    const_dnnl_memory_desc_t relu3_dst_md
            = dnnl_primitive_desc_query_md(relu3_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&relu3_dst_memory, relu3_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立relu3原型
    CHECK(dnnl_primitive_create(&relu3, relu3_pd));
    net_cnn_fwd[n_cnn_fwd] = relu3;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 2);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            conv3_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_DST,
            relu3_dst_memory);
    n_cnn_fwd++;

    // pool3
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, pool3_dst_sizes, dnnl_nchw,
            Worker -> engine, Worker -> pool3_dst,
            &pool3_user_dst_memory);

    // 建立pool3原型描述符
    {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive
        const_dnnl_memory_desc_t pool3_sys_src_md = relu3_dst_md;

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool3_sys_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool3_sys_dst_md, 4,
                pool3_dst_sizes, dnnl_f32, dnnl_nchw));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool3_pd,
                Worker -> engine, dnnl_forward, dnnl_pooling_max,
                pool3_sys_src_md, pool3_sys_dst_md, pool3_strides,
                pool3_kernel, pool3_dilation, pool3_padding, pool3_padding,
                NULL));
        CHECK(dnnl_memory_desc_destroy(pool3_sys_dst_md));
    }

    // 创造workSpace存储器
    const_dnnl_memory_desc_t pool3_ws_md = dnnl_primitive_desc_query_md(
            pool3_pd, dnnl_query_workspace_md, 0);
    CHECK(dnnl_memory_create(&pool3_ws_memory, pool3_ws_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 评估并创建重定位
    const_dnnl_memory_desc_t pool3_dst_md
            = dnnl_primitive_desc_query_md(pool3_pd, dnnl_query_dst_md, 0);
    n_cnn_fwd += 1; // tentative workaround: preserve space for pooling that
                    // should happen before the reorder
    CHECK(prepare_reorder(&pool3_user_dst_memory, pool3_dst_md,
            Worker -> engine, 0, &pool3_internal_dst_memory,
            &pool3_reorder_dst, &n_cnn_fwd, net_cnn_fwd, net_cnn_fwd_args));
    n_cnn_fwd -= pool3_reorder_dst ? 2 : 1;

    pool3_dst_memory = pool3_internal_dst_memory
            ? pool3_internal_dst_memory
            : pool3_user_dst_memory;

    // 建立pool3原型
    CHECK(dnnl_primitive_create(&pool3, pool3_pd));
    net_cnn_fwd[n_cnn_fwd] = pool3;
    prepare_arg_node(&net_cnn_fwd_args[n_cnn_fwd], 3);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[0], DNNL_ARG_SRC,
            relu3_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[1], DNNL_ARG_DST,
            pool3_dst_memory);
    set_arg(&net_cnn_fwd_args[n_cnn_fwd].args[2], DNNL_ARG_WORKSPACE,
            pool3_ws_memory);
    n_cnn_fwd++;

    if (pool3_reorder_dst) n_cnn_fwd += 1;
    
    /*----------------- Actor Forward Stream --------------------------------*/
    n_actor_fwd = 0;
    
    // dense_actor1
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA1_src_md, DNSE_DIMS,
            dnseA1_src_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA1_src_memory, dnseA1_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA1_src, dnseA1_src_memory);
    
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA1_weights_md, DNSE_DIMS,
            dnseA1_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseA1_weights_memory, dnseA1_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA1_weights,
            dnseA1_weights_memory);
    
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA1_bias_md, 1,
            dnseA1_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA1_bias_memory, dnseA1_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA1_bias, dnseA1_bias_memory);
    
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA1_dst_md, DNSE_DIMS,
            dnseA1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA1_dst_memory, dnseA1_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA1原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseA1_pd,
            Worker -> engine, dnnl_forward_training, dnseA1_src_md,
            dnseA1_weights_md, dnseA1_bias_md, dnseA1_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA1_bias_md));

    // 建立dnseA1原型
    CHECK(dnnl_primitive_create(&dnseA1, dnseA1_pd));
    net_actor_fwd[n_actor_fwd] = dnseA1;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 4);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            dnseA1_src_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseA1_weights_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[2], DNNL_ARG_BIAS,
            dnseA1_bias_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[3], DNNL_ARG_DST,
            dnseA1_dst_memory);
    n_actor_fwd++;
    
    // relu_actor1
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA1_dst_md, DNSE_DIMS,
            reluA1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA1_dst_memory, reluA1_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA1原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluA1_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseA1_dst_md, RELUA1_NSLOP, 0, NULL));

    // 建立reluA1原型
    CHECK(dnnl_primitive_create(&reluA1, reluA1_pd));
    net_actor_fwd[n_actor_fwd] = reluA1;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 2);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            dnseA1_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_DST,
            reluA1_dst_memory);
    n_actor_fwd++;
    
    // dense_actor2
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA2_weights_md, DNSE_DIMS,
            dnseA2_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseA2_weights_memory, dnseA2_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA2_weights,
            dnseA2_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA2_bias_md, 1,
            dnseA2_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA2_bias_memory, dnseA2_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA2_bias, dnseA2_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA2_dst_md, DNSE_DIMS,
            dnseA2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA2_dst_memory, dnseA2_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA2原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseA2_pd,
            Worker -> engine, dnnl_forward_training, reluA1_dst_md,
            dnseA2_weights_md, dnseA2_bias_md, dnseA2_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA2_bias_md));

    // 建立dnseA2原型
    CHECK(dnnl_primitive_create(&dnseA2, dnseA2_pd));
    net_actor_fwd[n_actor_fwd] = dnseA2;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 4);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            reluA1_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseA2_weights_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[2], DNNL_ARG_BIAS,
            dnseA2_bias_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[3], DNNL_ARG_DST,
            dnseA2_dst_memory);
    n_actor_fwd++;

    // relu_actor2
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA2_dst_md, DNSE_DIMS,
            reluA2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA2_dst_memory, reluA2_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA2原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluA2_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseA2_dst_md, RELUA2_NSLOP, 0, NULL));

    // 建立reluA2原型
    CHECK(dnnl_primitive_create(&reluA2, reluA2_pd));
    net_actor_fwd[n_actor_fwd] = reluA2;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 2);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            dnseA2_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_DST,
            reluA2_dst_memory);
    n_actor_fwd++;

    // dense_actor3
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA3_weights_md, DNSE_DIMS,
            dnseA3_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseA3_weights_memory, dnseA3_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA3_weights,
            dnseA3_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA3_bias_md, 1,
            dnseA3_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA3_bias_memory, dnseA3_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA3_bias, dnseA3_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA3_dst_md, DNSE_DIMS,
            dnseA3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA3_dst_memory, dnseA3_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA3原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseA3_pd,
            Worker -> engine, dnnl_forward_training, reluA2_dst_md,
            dnseA3_weights_md, dnseA3_bias_md, dnseA3_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA3_bias_md));

    // 建立dnseA3原型
    CHECK(dnnl_primitive_create(&dnseA3, dnseA3_pd));
    net_actor_fwd[n_actor_fwd] = dnseA3;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 4);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            reluA2_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseA3_weights_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[2], DNNL_ARG_BIAS,
            dnseA3_bias_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[3], DNNL_ARG_DST,
            dnseA3_dst_memory);
    n_actor_fwd++;

    // relu_actor3
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA3_dst_md, DNSE_DIMS,
            reluA3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA3_dst_memory, reluA3_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA3原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluA3_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseA3_dst_md, RELUA3_NSLOP, 0, NULL));

    // 建立reluA3原型
    CHECK(dnnl_primitive_create(&reluA3, reluA3_pd));
    net_actor_fwd[n_actor_fwd] = reluA3;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 2);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            dnseA3_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_DST,
            reluA3_dst_memory);
    n_actor_fwd++;

    // dense_actor4
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA4_weights_md, DNSE_DIMS,
            dnseA4_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseA4_weights_memory, dnseA4_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA4_weights,
            dnseA4_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA4_bias_md, 1,
            dnseA4_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA4_bias_memory, dnseA4_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA4_bias, dnseA4_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA4_dst_md, DNSE_DIMS,
            dnseA4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA4_dst_memory, dnseA4_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA4原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseA4_pd,
            Worker -> engine, dnnl_forward_training, reluA3_dst_md,
            dnseA4_weights_md, dnseA4_bias_md, dnseA4_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA4_bias_md));

    // 建立dnseA4原型
    CHECK(dnnl_primitive_create(&dnseA4, dnseA4_pd));
    net_actor_fwd[n_actor_fwd] = dnseA4;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 4);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            reluA3_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseA4_weights_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[2], DNNL_ARG_BIAS,
            dnseA4_bias_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[3], DNNL_ARG_DST,
            dnseA4_dst_memory);
    n_actor_fwd++;

    // relu_actor4
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA4_dst_md, DNSE_DIMS,
            reluA4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA4_dst_memory, reluA4_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA4原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluA4_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseA4_dst_md, RELUA4_NSLOP, 0, NULL));

    // 建立reluA4原型
    CHECK(dnnl_primitive_create(&reluA4, reluA4_pd));
    net_actor_fwd[n_actor_fwd] = reluA4;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 2);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            dnseA4_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_DST,
            reluA4_dst_memory);
    n_actor_fwd++;

    // dense_actor5
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA5_weights_md, DNSE_DIMS,
            dnseA5_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseA5_weights_memory, dnseA5_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA5_weights,
            dnseA5_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA5_bias_md, 1,
            dnseA5_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA5_bias_memory, dnseA5_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA5_bias, dnseA5_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA5_dst_md, DNSE_DIMS,
            dnseA5_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA5_dst_memory, dnseA5_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA5原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseA5_pd,
            Worker -> engine, dnnl_forward_training, reluA4_dst_md,
            dnseA5_weights_md, dnseA5_bias_md, dnseA5_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA5_bias_md));

    // 建立dnseA5原型
    CHECK(dnnl_primitive_create(&dnseA5, dnseA5_pd));
    net_actor_fwd[n_actor_fwd] = dnseA5;
    prepare_arg_node(&net_actor_fwd_args[n_actor_fwd], 4);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[0], DNNL_ARG_SRC,
            reluA4_dst_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseA5_weights_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[2], DNNL_ARG_BIAS,
            dnseA5_bias_memory);
    set_arg(&net_actor_fwd_args[n_actor_fwd].args[3], DNNL_ARG_DST,
            dnseA5_dst_memory);
    n_actor_fwd++;
	
    /*----------------- Critic Forward Stream -------------------------------*/
    n_critic_fwd = 0;

    // dense_critic1
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC1_src_md, DNSE_DIMS,
            dnseC1_src_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC1_src_memory, dnseC1_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC1_src, dnseC1_src_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC1_weights_md, DNSE_DIMS,
            dnseC1_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseC1_weights_memory, dnseC1_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC1_weights,
            dnseC1_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC1_bias_md, 1,
            dnseC1_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC1_bias_memory, dnseC1_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC1_bias, dnseC1_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC1_dst_md, DNSE_DIMS,
            dnseC1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC1_dst_memory, dnseC1_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC1原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseC1_pd,
            Worker -> engine, dnnl_forward_training, dnseC1_src_md,
            dnseC1_weights_md, dnseC1_bias_md, dnseC1_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC1_bias_md));

    // 建立dnseC1原型
    CHECK(dnnl_primitive_create(&dnseC1, dnseC1_pd));
    net_critic_fwd[n_critic_fwd] = dnseC1;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 4);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            dnseC1_src_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseC1_weights_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[2], DNNL_ARG_BIAS,
            dnseC1_bias_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[3], DNNL_ARG_DST,
            dnseC1_dst_memory);
    n_critic_fwd++;

    // relu_critic1
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC1_dst_md, DNSE_DIMS,
            reluC1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC1_dst_memory, reluC1_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC1原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluC1_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseC1_dst_md, RELUC1_NSLOP, 0, NULL));

    // 建立reluC1原型
    CHECK(dnnl_primitive_create(&reluC1, reluC1_pd));
    net_critic_fwd[n_critic_fwd] = reluC1;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 2);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            dnseC1_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_DST,
            reluC1_dst_memory);
    n_critic_fwd++;

    // dense_critic2
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC2_weights_md, DNSE_DIMS,
            dnseC2_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseC2_weights_memory, dnseC2_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC2_weights,
            dnseC2_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC2_bias_md, 1,
            dnseC2_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC2_bias_memory, dnseC2_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC2_bias, dnseC2_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC2_dst_md, DNSE_DIMS,
            dnseC2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC2_dst_memory, dnseC2_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC2原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseC2_pd,
            Worker -> engine, dnnl_forward_training, reluC1_dst_md,
            dnseC2_weights_md, dnseC2_bias_md, dnseC2_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC2_bias_md));

    // 建立dnseC2原型
    CHECK(dnnl_primitive_create(&dnseC2, dnseC2_pd));
    net_critic_fwd[n_critic_fwd] = dnseC2;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 4);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            reluC1_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseC2_weights_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[2], DNNL_ARG_BIAS,
            dnseC2_bias_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[3], DNNL_ARG_DST,
            dnseC2_dst_memory);
    n_critic_fwd++;

    // relu_critic2
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC2_dst_md, DNSE_DIMS,
            reluC2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC2_dst_memory, reluC2_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC2原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluC2_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseC2_dst_md, RELUC2_NSLOP, 0, NULL));

    // 建立reluC2原型
    CHECK(dnnl_primitive_create(&reluC2, reluC2_pd));
    net_critic_fwd[n_critic_fwd] = reluC2;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 2);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            dnseC2_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_DST,
            reluC2_dst_memory);
    n_critic_fwd++;

    // dense_critic3
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC3_weights_md, DNSE_DIMS,
            dnseC3_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseC3_weights_memory, dnseC3_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC3_weights,
            dnseC3_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC3_bias_md, 1,
            dnseC3_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC3_bias_memory, dnseC3_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC3_bias, dnseC3_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC3_dst_md, DNSE_DIMS,
            dnseC3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC3_dst_memory, dnseC3_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC3原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseC3_pd,
            Worker -> engine, dnnl_forward_training, reluC2_dst_md,
            dnseC3_weights_md, dnseC3_bias_md, dnseC3_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC3_bias_md));

    // 建立dnseC3原型
    CHECK(dnnl_primitive_create(&dnseC3, dnseC3_pd));
    net_critic_fwd[n_critic_fwd] = dnseC3;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 4);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            reluC2_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseC3_weights_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[2], DNNL_ARG_BIAS,
            dnseC3_bias_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[3], DNNL_ARG_DST,
            dnseC3_dst_memory);
    n_critic_fwd++;

    // relu_critic3
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC3_dst_md, DNSE_DIMS,
            reluC3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC3_dst_memory, reluC3_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC3原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluC3_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseC3_dst_md, RELUC3_NSLOP, 0, NULL));

    // 建立reluC3原型
    CHECK(dnnl_primitive_create(&reluC3, reluC3_pd));
    net_critic_fwd[n_critic_fwd] = reluC3;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 2);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            dnseC3_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_DST,
            reluC3_dst_memory);
    n_critic_fwd++;

    // dense_critic4
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC4_weights_md, DNSE_DIMS,
            dnseC4_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseC4_weights_memory, dnseC4_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC4_weights,
            dnseC4_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC4_bias_md, 1,
            dnseC4_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC4_bias_memory, dnseC4_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC4_bias, dnseC4_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC4_dst_md, DNSE_DIMS,
            dnseC4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC4_dst_memory, dnseC4_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC4原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseC4_pd,
            Worker -> engine, dnnl_forward_training, reluC3_dst_md,
            dnseC4_weights_md, dnseC4_bias_md, dnseC4_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC4_bias_md));

    // 建立dnseC4原型
    CHECK(dnnl_primitive_create(&dnseC4, dnseC4_pd));
    net_critic_fwd[n_critic_fwd] = dnseC4;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 4);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            reluC3_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseC4_weights_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[2], DNNL_ARG_BIAS,
            dnseC4_bias_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[3], DNNL_ARG_DST,
            dnseC4_dst_memory);
    n_critic_fwd++;

    // relu_critic4
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC4_dst_md, DNSE_DIMS,
            reluC4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC4_dst_memory, reluC4_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC4原型描述符
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&reluC4_pd,
            Worker -> engine, dnnl_forward_training,
            dnnl_eltwise_relu, dnseC4_dst_md, RELUC4_NSLOP, 0, NULL));

    // 建立reluC4原型
    CHECK(dnnl_primitive_create(&reluC4, reluC4_pd));
    net_critic_fwd[n_critic_fwd] = reluC4;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 2);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            dnseC4_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_DST,
            reluC4_dst_memory);
    n_critic_fwd++;

    // dense_critic5
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC5_weights_md, DNSE_DIMS,
            dnseC5_weights_sizes, dnnl_f32, dnnl_oi));
    CHECK(dnnl_memory_create(&dnseC5_weights_memory, dnseC5_weights_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC5_weights,
            dnseC5_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC5_bias_md, 1,
            dnseC5_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC5_bias_memory, dnseC5_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC5_bias, dnseC5_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC5_dst_md, DNSE_DIMS,
            dnseC5_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC5_dst_memory, dnseC5_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC5原型描述符
    CHECK(dnnl_inner_product_forward_primitive_desc_create(&dnseC5_pd,
            Worker -> engine, dnnl_forward_training, reluC4_dst_md,
            dnseC5_weights_md, dnseC5_bias_md, dnseC5_dst_md, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC5_bias_md));
    CHECK(dnnl_memory_desc_destroy(dnseC5_dst_md));

    // 建立dnseC5原型
    CHECK(dnnl_primitive_create(&dnseC5, dnseC5_pd));
    net_critic_fwd[n_critic_fwd] = dnseC5;
    prepare_arg_node(&net_critic_fwd_args[n_critic_fwd], 4);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[0], DNNL_ARG_SRC,
            reluC4_dst_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[1], DNNL_ARG_WEIGHTS,
            dnseC5_weights_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[2], DNNL_ARG_BIAS,
            dnseC5_bias_memory);
    set_arg(&net_critic_fwd_args[n_critic_fwd].args[3], DNNL_ARG_DST,
            dnseC5_dst_memory);
    n_critic_fwd++;
	
    /*----------------- Actor Backward Stream -------------------------------*/
    n_actor_bwd = 0;

    // dense_actor5_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA5_diff_dst_md, DNSE_DIMS,
            dnseA5_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA5_diff_dst_memory, dnseA5_diff_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA5_diff_dst,
            dnseA5_diff_dst_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA5_diff_weights_md, DNSE_DIMS,
            dnseA5_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA5_diff_weights_memory,
            dnseA5_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA5_diff_weights,
            dnseA5_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA5_diff_bias_md, 1,
            dnseA5_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA5_diff_bias_memory, dnseA5_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA5_diff_bias,
            dnseA5_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA5_diff_src_md, DNSE_DIMS,
            reluA4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA5_diff_src_memory, dnseA5_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA5_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseA5_bwd_data_pd, Worker -> engine,
            dnseA5_diff_src_md, dnseA5_weights_md, dnseA5_diff_dst_md,
            dnseA5_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA5_weights_md));

    // 建立dnseA5_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseA5_bwd_data, dnseA5_bwd_data_pd));
    net_actor_bwd[n_actor_bwd] = dnseA5_bwd_data;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseA5_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseA5_diff_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseA5_diff_src_memory);
    n_actor_bwd++;

    // 建立dnseA5_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseA5_bwd_weights_pd, Worker -> engine,
            reluA4_dst_md, dnseA5_diff_weights_md, dnseA5_diff_bias_md, 
            dnseA5_diff_dst_md, dnseA5_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluA4_dst_md));

    // 建立dnseA5_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseA5_bwd_weights, dnseA5_bwd_weights_pd));
    net_actor_bwd[n_actor_bwd] = dnseA5_bwd_weights;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 4);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            reluA4_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseA5_diff_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseA5_diff_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseA5_diff_bias_memory);
    n_actor_bwd++;

    // relu_actor4_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA4_diff_src_md, DNSE_DIMS,
            dnseA4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA4_diff_src_memory, reluA4_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA4_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluA4_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseA5_diff_src_md,
            dnseA4_dst_md, RELUC4_NSLOP, 0, reluA4_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA4_dst_md));

    // 建立reluA4_bwd原型
    CHECK(dnnl_primitive_create(&reluA4_bwd, reluA4_bwd_pd));
    net_actor_bwd[n_actor_bwd] = reluA4_bwd;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            dnseA4_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseA5_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluA4_diff_src_memory);
    n_actor_bwd++;

    // dense_actor4_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA4_diff_weights_md, DNSE_DIMS,
            dnseA4_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA4_diff_weights_memory,
            dnseA4_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA4_diff_weights,
            dnseA4_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA4_diff_bias_md, 1,
            dnseA4_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA4_diff_bias_memory, dnseA4_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA4_diff_bias,
            dnseA4_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA4_diff_src_md, DNSE_DIMS,
            reluA3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA4_diff_src_memory, dnseA4_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA4_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseA4_bwd_data_pd, Worker -> engine,
            dnseA4_diff_src_md, dnseA4_weights_md, reluA4_diff_src_md,
            dnseA4_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA4_weights_md));

    // 建立dnseA4_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseA4_bwd_data, dnseA4_bwd_data_pd));
    net_actor_bwd[n_actor_bwd] = dnseA4_bwd_data;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseA4_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA4_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseA4_diff_src_memory);
    n_actor_bwd++;

    // 建立dnseA4_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseA4_bwd_weights_pd, Worker -> engine,
            reluA3_dst_md, dnseA4_diff_weights_md, dnseA4_diff_bias_md, 
            reluA4_diff_src_md, dnseA4_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluA3_dst_md));

    // 建立dnseA4_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseA4_bwd_weights, dnseA4_bwd_weights_pd));
    net_actor_bwd[n_actor_bwd] = dnseA4_bwd_weights;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 4);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            reluA3_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA4_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseA4_diff_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseA4_diff_bias_memory);
    n_actor_bwd++;

    // relu_actor3_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA3_diff_src_md, DNSE_DIMS,
            dnseA3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA3_diff_src_memory, reluA3_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA3_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluA3_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseA4_diff_src_md,
            dnseA3_dst_md, RELUC3_NSLOP, 0, reluA3_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA3_dst_md));

    // 建立reluA3_bwd原型
    CHECK(dnnl_primitive_create(&reluA3_bwd, reluA3_bwd_pd));
    net_actor_bwd[n_actor_bwd] = reluA3_bwd;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            dnseA3_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseA4_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluA3_diff_src_memory);
    n_actor_bwd++;

    // dense_actor3_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA3_diff_weights_md, DNSE_DIMS,
            dnseA3_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA3_diff_weights_memory,
            dnseA3_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA3_diff_weights,
            dnseA3_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA3_diff_bias_md, 1,
            dnseA3_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA3_diff_bias_memory, dnseA3_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA3_diff_bias,
            dnseA3_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA3_diff_src_md, DNSE_DIMS,
            reluA2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA3_diff_src_memory, dnseA3_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA3_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseA3_bwd_data_pd, Worker -> engine,
            dnseA3_diff_src_md, dnseA3_weights_md, reluA3_diff_src_md,
            dnseA3_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA3_weights_md));

    // 建立dnseA3_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseA3_bwd_data, dnseA3_bwd_data_pd));
    net_actor_bwd[n_actor_bwd] = dnseA3_bwd_data;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseA3_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA4_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseA3_diff_src_memory);
    n_actor_bwd++;

    // 建立dnseA3_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseA3_bwd_weights_pd, Worker -> engine,
            reluA2_dst_md, dnseA3_diff_weights_md, dnseA3_diff_bias_md, 
            reluA3_diff_src_md, dnseA3_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluA2_dst_md));

    // 建立dnseA3_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseA3_bwd_weights, dnseA3_bwd_weights_pd));
    net_actor_bwd[n_actor_bwd] = dnseA3_bwd_weights;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 4);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            reluA2_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA4_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseA3_diff_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseA3_diff_bias_memory);
    n_actor_bwd++;

    // relu_critic2_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA2_diff_src_md, DNSE_DIMS,
            dnseA2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA2_diff_src_memory, reluA2_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA2_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluA2_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseA3_diff_src_md,
            dnseA2_dst_md, RELUC2_NSLOP, 0, reluA2_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA2_dst_md));

    // 建立reluA2_bwd原型
    CHECK(dnnl_primitive_create(&reluA2_bwd, reluA2_bwd_pd));
    net_actor_bwd[n_actor_bwd] = reluA2_bwd;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            dnseA2_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseA3_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluA2_diff_src_memory);
    n_actor_bwd++;

    // dense_actor2_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA2_diff_weights_md, DNSE_DIMS,
            dnseA2_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA2_diff_weights_memory,
            dnseA2_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA2_diff_weights,
            dnseA2_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA2_diff_bias_md, 1,
            dnseA2_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA2_diff_bias_memory, dnseA2_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA2_diff_bias,
            dnseA2_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA2_diff_src_md, DNSE_DIMS,
            reluA1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA2_diff_src_memory, dnseA2_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA2_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseA2_bwd_data_pd, Worker -> engine,
            dnseA2_diff_src_md, dnseA2_weights_md, reluA2_diff_src_md,
            dnseA2_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA2_weights_md));

    // 建立dnseA2_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseA2_bwd_data, dnseA2_bwd_data_pd));
    net_actor_bwd[n_actor_bwd] = dnseA2_bwd_data;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseA2_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA3_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseA2_diff_src_memory);
    n_actor_bwd++;

    // 建立dnseA2_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseA2_bwd_weights_pd, Worker -> engine,
            reluA1_dst_md, dnseA2_diff_weights_md, dnseA2_diff_bias_md, 
            reluA2_diff_src_md, dnseA2_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluA1_dst_md));

    // 建立dnseA2_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseA2_bwd_weights, dnseA2_bwd_weights_pd));
    net_actor_bwd[n_actor_bwd] = dnseA2_bwd_weights;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 4);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            reluA1_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA3_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseA2_diff_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseA2_diff_bias_memory);
    n_actor_bwd++;

    // relu_actor1_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluA1_diff_src_md, DNSE_DIMS,
            dnseA1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluA1_diff_src_memory, reluA1_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluA1_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluA1_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseA2_diff_src_md,
            dnseA1_dst_md, RELUC1_NSLOP, 0, reluA1_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA1_dst_md));

    // 建立reluA1_bwd原型
    CHECK(dnnl_primitive_create(&reluA1_bwd, reluA1_bwd_pd));
    net_actor_bwd[n_actor_bwd] = reluA1_bwd;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            dnseA1_dst_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseA2_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluA1_diff_src_memory);
    n_actor_bwd++;

    // dense_actor1_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseA1_diff_weights_md, DNSE_DIMS,
            dnseA1_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA1_diff_weights_memory,
            dnseA1_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA1_diff_weights,
            dnseA1_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA1_diff_bias_md, 1,
            dnseA1_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseA1_diff_bias_memory, dnseA1_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseA1_diff_bias,
            dnseA1_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseA1_diff_src_md, DNSE_DIMS,
            dnseA1_src_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseA1_diff_src_memory, dnseA1_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseA1_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseA1_bwd_data_pd, Worker -> engine,
            dnseA1_diff_src_md, dnseA1_weights_md, reluA1_diff_src_md,
            dnseA1_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA1_weights_md));

    // 建立dnseA1_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseA1_bwd_data, dnseA1_bwd_data_pd));
    net_actor_bwd[n_actor_bwd] = dnseA1_bwd_data;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 3);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseA1_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA2_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseA1_diff_src_memory);
    n_actor_bwd++;

    // 建立dnseA1_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseA1_bwd_weights_pd, Worker -> engine,
            dnseA1_src_md, dnseA1_diff_weights_md, dnseA1_diff_bias_md,
            reluA1_diff_src_md, dnseA1_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseA1_src_md));

    // 建立dnseA1_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseA1_bwd_weights, dnseA1_bwd_weights_pd));
    net_actor_bwd[n_actor_bwd] = dnseA1_bwd_weights;
    prepare_arg_node(&net_actor_bwd_args[n_actor_bwd], 4);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[0], DNNL_ARG_SRC,
            dnseA1_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluA2_diff_src_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseA1_diff_weights_memory);
    set_arg(&net_actor_bwd_args[n_actor_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseA1_diff_bias_memory);
    n_actor_bwd++;
	
    /*----------------- Critic Backward Stream ------------------------------*/
    n_critic_bwd = 0;

    // dense_critic5_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC5_diff_dst_md, DNSE_DIMS,
            dnseC5_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC5_diff_dst_memory, dnseC5_diff_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC5_diff_dst,
            dnseC5_diff_dst_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC5_diff_weights_md, DNSE_DIMS,
            dnseC5_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC5_diff_weights_memory,
            dnseC5_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC5_diff_weights,
            dnseC5_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC5_diff_bias_md, 1,
            dnseC5_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC5_diff_bias_memory, dnseC5_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC5_diff_bias,
            dnseC5_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC5_diff_src_md, DNSE_DIMS,
            reluC4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC5_diff_src_memory, dnseC5_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC5_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseC5_bwd_data_pd, Worker -> engine,
            dnseC5_diff_src_md, dnseC5_weights_md, dnseC5_diff_dst_md,
            dnseC5_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC5_weights_md));

    // 建立dnseC5_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseC5_bwd_data, dnseC5_bwd_data_pd));
    net_critic_bwd[n_critic_bwd] = dnseC5_bwd_data;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseC5_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseC5_diff_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseC5_diff_src_memory);
    n_critic_bwd++;

    // 建立dnseC5_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseC5_bwd_weights_pd, Worker -> engine,
            reluC4_dst_md, dnseC5_diff_weights_md, dnseC5_diff_bias_md, 
            dnseC5_diff_dst_md, dnseC5_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluC4_dst_md));

    // 建立dnseC5_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseC5_bwd_weights, dnseC5_bwd_weights_pd));
    net_critic_bwd[n_critic_bwd] = dnseC5_bwd_weights;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 4);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            reluC4_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseC5_diff_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseC5_diff_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseC5_diff_bias_memory);
    n_critic_bwd++;

    // relu_critic4_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC4_diff_src_md, DNSE_DIMS,
            dnseC4_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC4_diff_src_memory, reluC4_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC4_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluC4_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseC5_diff_src_md,
            dnseC4_dst_md, RELUC4_NSLOP, 0, reluC4_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC4_dst_md));

    // 建立reluC4_bwd原型
    CHECK(dnnl_primitive_create(&reluC4_bwd, reluC4_bwd_pd));
    net_critic_bwd[n_critic_bwd] = reluC4_bwd;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            dnseC4_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseC5_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluC4_diff_src_memory);
    n_critic_bwd++;

    // dense_critic4_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC4_diff_weights_md, DNSE_DIMS,
            dnseC4_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC4_diff_weights_memory,
            dnseC4_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC4_diff_weights,
            dnseC4_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC4_diff_bias_md, 1,
            dnseC4_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC4_diff_bias_memory, dnseC4_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC4_diff_bias,
            dnseC4_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC4_diff_src_md, DNSE_DIMS,
            reluC3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC4_diff_src_memory, dnseC4_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC4_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseC4_bwd_data_pd, Worker -> engine,
            dnseC4_diff_src_md, dnseC4_weights_md, reluC4_diff_src_md,
            dnseC4_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC4_weights_md));

    // 建立dnseC4_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseC4_bwd_data, dnseC4_bwd_data_pd));
    net_critic_bwd[n_critic_bwd] = dnseC4_bwd_data;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseC4_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC4_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseC4_diff_src_memory);
    n_critic_bwd++;

    // 建立dnseC4_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseC4_bwd_weights_pd, Worker -> engine,
            reluC3_dst_md, dnseC4_diff_weights_md, dnseC4_diff_bias_md, 
            reluC4_diff_src_md, dnseC4_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluC3_dst_md));

    // 建立dnseC4_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseC4_bwd_weights, dnseC4_bwd_weights_pd));
    net_critic_bwd[n_critic_bwd] = dnseC4_bwd_weights;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 4);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            reluC3_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC4_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseC4_diff_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseC4_diff_bias_memory);
    n_critic_bwd++;

    // relu_critic3_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC3_diff_src_md, DNSE_DIMS,
            dnseC3_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC3_diff_src_memory, reluC3_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC3_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluC3_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseC4_diff_src_md,
            dnseC3_dst_md, RELUC3_NSLOP, 0, reluC3_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC3_dst_md));

    // 建立reluC3_bwd原型
    CHECK(dnnl_primitive_create(&reluC3_bwd, reluC3_bwd_pd));
    net_critic_bwd[n_critic_bwd] = reluC3_bwd;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            dnseC3_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseC4_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluC3_diff_src_memory);
    n_critic_bwd++;

    // dense_critic3_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC3_diff_weights_md, DNSE_DIMS,
            dnseC3_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC3_diff_weights_memory,
            dnseC3_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC3_diff_weights,
            dnseC3_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC3_diff_bias_md, 1,
            dnseC3_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC3_diff_bias_memory, dnseC3_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC3_diff_bias,
            dnseC3_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC3_diff_src_md, DNSE_DIMS,
            reluC2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC3_diff_src_memory, dnseC3_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC3_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseC3_bwd_data_pd, Worker -> engine,
            dnseC3_diff_src_md, dnseC3_weights_md, reluC3_diff_src_md,
            dnseC3_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC3_weights_md));

    // 建立dnseC3_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseC3_bwd_data, dnseC3_bwd_data_pd));
    net_critic_bwd[n_critic_bwd] = dnseC3_bwd_data;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseC3_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC4_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseC3_diff_src_memory);
    n_critic_bwd++;

    // 建立dnseC3_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseC3_bwd_weights_pd, Worker -> engine,
            reluC2_dst_md, dnseC3_diff_weights_md, dnseC3_diff_bias_md, 
            reluC3_diff_src_md, dnseC3_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluC2_dst_md));

    // 建立dnseC3_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseC3_bwd_weights, dnseC3_bwd_weights_pd));
    net_critic_bwd[n_critic_bwd] = dnseC3_bwd_weights;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 4);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            reluC2_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC4_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseC3_diff_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseC3_diff_bias_memory);
    n_critic_bwd++;

    // relu_critic2_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC2_diff_src_md, DNSE_DIMS,
            dnseC2_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC2_diff_src_memory, reluC2_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC2_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluC2_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseC3_diff_src_md,
            dnseC2_dst_md, RELUC2_NSLOP, 0, reluC2_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC2_dst_md));

    // 建立reluC2_bwd原型
    CHECK(dnnl_primitive_create(&reluC2_bwd, reluC2_bwd_pd));
    net_critic_bwd[n_critic_bwd] = reluC2_bwd;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            dnseC2_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseC3_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluC2_diff_src_memory);
    n_critic_bwd++;

    // dense_critic2_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC2_diff_weights_md, DNSE_DIMS,
            dnseC2_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC2_diff_weights_memory,
            dnseC2_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC2_diff_weights,
            dnseC2_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC2_diff_bias_md, 1,
            dnseC2_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC2_diff_bias_memory, dnseC2_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC2_diff_bias,
            dnseC2_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC2_diff_src_md, DNSE_DIMS,
            reluC1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC2_diff_src_memory, dnseC2_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC2_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseC2_bwd_data_pd, Worker -> engine,
            dnseC2_diff_src_md, dnseC2_weights_md, reluC2_diff_src_md,
            dnseC2_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC2_weights_md));

    // 建立dnseC2_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseC2_bwd_data, dnseC2_bwd_data_pd));
    net_critic_bwd[n_critic_bwd] = dnseC2_bwd_data;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseC2_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC3_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseC2_diff_src_memory);
    n_critic_bwd++;

    // 建立dnseC2_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseC2_bwd_weights_pd, Worker -> engine,
            reluC1_dst_md, dnseC2_diff_weights_md, dnseC2_diff_bias_md, 
            reluC2_diff_src_md, dnseC2_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(reluC1_dst_md));

    // 建立dnseC2_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseC2_bwd_weights, dnseC2_bwd_weights_pd));
    net_critic_bwd[n_critic_bwd] = dnseC2_bwd_weights;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 4);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            reluC1_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC3_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseC2_diff_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseC2_diff_bias_memory);
    n_critic_bwd++;

    // relu_critic1_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&reluC1_diff_src_md, DNSE_DIMS,
            dnseC1_dst_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&reluC1_diff_src_memory, reluC1_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立reluC1_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&reluC1_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, dnseC2_diff_src_md,
            dnseC1_dst_md, RELUC1_NSLOP, 0, reluC1_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC1_dst_md));

    // 建立reluC1_bwd原型
    CHECK(dnnl_primitive_create(&reluC1_bwd, reluC1_bwd_pd));
    net_critic_bwd[n_critic_bwd] = reluC1_bwd;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            dnseC1_dst_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            dnseC2_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            reluC1_diff_src_memory);
    n_critic_bwd++;

    // dense_critic1_backward
    // 创造数据存储器
    CHECK(dnnl_memory_desc_create_with_tag(&dnseC1_diff_weights_md, DNSE_DIMS,
            dnseC1_weights_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC1_diff_weights_memory,
            dnseC1_diff_weights_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC1_diff_weights,
            dnseC1_diff_weights_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC1_diff_bias_md, 1,
            dnseC1_bias_sizes, dnnl_f32, dnnl_x));
    CHECK(dnnl_memory_create(&dnseC1_diff_bias_memory, dnseC1_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> dnseC1_diff_bias,
            dnseC1_diff_bias_memory);

    CHECK(dnnl_memory_desc_create_with_tag(&dnseC1_diff_src_md, DNSE_DIMS,
            dnseC1_src_sizes, dnnl_f32, dnnl_nc));
    CHECK(dnnl_memory_create(&dnseC1_diff_src_memory, dnseC1_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立dnseC1_bwd_data原型描述符
    CHECK(dnnl_inner_product_backward_data_primitive_desc_create(
            &dnseC1_bwd_data_pd, Worker -> engine,
            dnseC1_diff_src_md, dnseC1_weights_md, reluC1_diff_src_md,
            dnseC1_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC1_weights_md));

    // 建立dnseC1_bwd_data原型
    CHECK(dnnl_primitive_create(&dnseC1_bwd_data, dnseC1_bwd_data_pd));
    net_critic_bwd[n_critic_bwd] = dnseC1_bwd_data;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 3);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_WEIGHTS,
            dnseC1_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC2_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_SRC,
            dnseC1_diff_src_memory);
    n_critic_bwd++;

    // 建立dnseC1_bwd_weights原型描述符
    CHECK(dnnl_inner_product_backward_weights_primitive_desc_create(
            &dnseC1_bwd_weights_pd, Worker -> engine,
            dnseC1_src_md, dnseC1_diff_weights_md, dnseC1_diff_bias_md,
            reluC1_diff_src_md, dnseC1_pd, NULL));

    CHECK(dnnl_memory_desc_destroy(dnseC1_src_md));

    // 建立dnseC1_bwd_weights原型
    CHECK(dnnl_primitive_create(&dnseC1_bwd_weights, dnseC1_bwd_weights_pd));
    net_critic_bwd[n_critic_bwd] = dnseC1_bwd_weights;
    prepare_arg_node(&net_critic_bwd_args[n_critic_bwd], 4);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[0], DNNL_ARG_SRC,
            dnseC1_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[1], DNNL_ARG_DIFF_DST,
            reluC2_diff_src_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            dnseC1_diff_weights_memory);
    set_arg(&net_critic_bwd_args[n_critic_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            dnseC1_diff_bias_memory);
    n_critic_bwd++;
	
    /*----------------- CNN Backward Stream ---------------------------------*/
    n_cnn_bwd = 0;

    // pool3_backward
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, pool3_dst_sizes, dnnl_nchw,
            Worker -> engine, Worker -> pool3_diff_dst,
            &pool3_user_diff_dst_memory);

    // 建立pool3_bwd原型描述符
    CHECK(dnnl_pooling_backward_primitive_desc_create(&pool3_bwd_pd,
            Worker -> engine, dnnl_pooling_max, relu3_dst_md,
            pool3_dst_md, pool3_strides, pool3_kernel, pool3_dilation,
            pool3_padding, pool3_padding, pool3_pd, NULL));

    // 评估并创建重定位
    CHECK(prepare_reorder(&pool3_user_diff_dst_memory, pool3_dst_md,
            Worker -> engine, 1, &pool3_internal_diff_dst_memory,
            &pool3_reorder_diff_dst, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    pool3_diff_dst_memory = pool3_internal_diff_dst_memory
            ? pool3_internal_diff_dst_memory
            : pool3_user_diff_dst_memory;

    // 创建结果存储器
    CHECK(dnnl_memory_create(&pool3_diff_src_memory, relu3_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立pool3_bwd原型
    CHECK(dnnl_primitive_create(&pool3_bwd, pool3_bwd_pd));
    net_cnn_bwd[n_cnn_bwd] = pool3_bwd;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_DIFF_DST,
            pool3_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_WORKSPACE,
            pool3_ws_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            pool3_diff_src_memory);
    n_cnn_bwd++;

    // relu3_backward
    // 建立relu3_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&relu3_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, relu3_dst_md,
            conv3_dst_md, RELU3_NSLOP, 0, relu3_pd, NULL));

    // 创造数据存储器
    const_dnnl_memory_desc_t relu3_diff_src_md = dnnl_primitive_desc_query_md(
            relu3_bwd_pd, dnnl_query_diff_src_md, 0);
    CHECK(dnnl_memory_create(&relu3_diff_src_memory, relu3_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立relu3_bwd原型
    CHECK(dnnl_primitive_create(&relu3_bwd, relu3_bwd_pd));
    net_cnn_bwd[n_cnn_bwd] = relu3_bwd;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_SRC,
            conv3_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            pool3_diff_src_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            relu3_diff_src_memory);
    n_cnn_bwd++;

    // conv3_backward
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, conv3_weights_sizes, dnnl_oihw,
            Worker -> engine, Worker -> conv3_diff_weights,
            &conv3_user_diff_weights_memory);

    const_dnnl_memory_desc_t conv3_user_diff_src_md = pool2_dst_md;
    CHECK(dnnl_memory_create(&conv3_user_diff_src_memory,
            conv3_user_diff_src_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));

    // 建立conv3_bwd_data原型描述符
    {
        dnnl_memory_desc_t conv3_sys_bwd_weights_md, conv3_sys_diff_dst_md,
                conv3_sys_diff_src_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_bwd_weights_md,
                CONV_DIMS, conv3_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_diff_dst_md,
                CONV_DIMS, conv3_dst_sizes, dnnl_f32, dnnl_nchw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_diff_src_md,
                CONV_DIMS, pool2_dst_sizes, dnnl_f32, dnnl_nchw));

        // create backward convolution descriptor
        CHECK(dnnl_convolution_backward_data_primitive_desc_create(
                &conv3_bwd_data_pd, Worker -> engine,
                dnnl_convolution_direct, conv3_sys_diff_src_md,
                conv3_sys_bwd_weights_md, conv3_sys_diff_dst_md, conv3_strides,
                conv3_dilation, conv3_padding, conv3_padding, conv3_pd, NULL));

        CHECK(dnnl_memory_desc_destroy(conv3_sys_bwd_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv3_sys_diff_dst_md));
        CHECK(dnnl_memory_desc_destroy(conv3_sys_diff_src_md));
    }

    // 评估并创建存储器和重定位
    const_dnnl_memory_desc_t conv3_bwd_weights_md = dnnl_primitive_desc_query_md(
            conv3_bwd_data_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv3_weights_memory, conv3_bwd_weights_md,
            Worker -> engine, 1, &conv3_internal_bwd_weights_memory,
            &conv3_bwd_reorder_weights, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv3_bwd_weights_memory = conv3_internal_bwd_weights_memory
            ? conv3_internal_bwd_weights_memory
            : conv3_weights_memory;

    const_dnnl_memory_desc_t conv3_diff_dst_md = dnnl_primitive_desc_query_md(
            conv3_bwd_data_pd, dnnl_query_diff_dst_md, 0);
    CHECK(prepare_reorder(&relu3_diff_src_memory, conv3_diff_dst_md,
            Worker -> engine, 1, &conv3_internal_diff_dst_memory,
            &conv3_reorder_diff_dst, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv3_diff_dst_memory = conv3_internal_diff_dst_memory
            ? conv3_internal_diff_dst_memory
            : relu3_diff_src_memory;

    const_dnnl_memory_desc_t conv3_diff_src_md
            = dnnl_primitive_desc_query_md(conv3_bwd_data_pd,
                    dnnl_query_diff_src_md, 0);
    n_cnn_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
                    // that should happen before the reorder
    CHECK(prepare_reorder(&conv3_user_diff_src_memory, conv3_diff_src_md,
            Worker -> engine, 0, &conv3_internal_diff_src_memory,
            &conv3_reorder_diff_src, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));
    n_cnn_bwd -= conv3_reorder_diff_src ? 2 : 1;

    conv3_diff_src_memory = conv3_internal_diff_src_memory
            ? conv3_internal_diff_src_memory
            : conv3_user_diff_src_memory;

    // 建立conv3_bwd_data原型
    CHECK(dnnl_primitive_create(&conv3_bwd_data, conv3_bwd_data_pd));
    net_cnn_bwd[n_cnn_bwd] = conv3_bwd_data;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_WEIGHTS,
            conv3_bwd_weights_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            conv3_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            conv3_diff_src_memory);
    n_cnn_bwd++;

    if (conv3_reorder_diff_src) n_cnn_bwd += 1;

    // 建立conv3_bwd_weights原型描述符
    {
        // memory descriptors should be in format `any` to allow backward
        // convolution for
        // weights to chose the format it prefers for best performance
        dnnl_memory_desc_t conv3_sys_bwd_src_md, conv3_sys_diff_weights_md,
                conv3_sys_diff_bias_md, conv3_sys_diff_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_bwd_src_md,
                CONV_DIMS, pool2_dst_sizes, dnnl_f32, dnnl_nchw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_diff_weights_md,
                CONV_DIMS, conv3_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_diff_bias_md, 1,
                conv3_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv3_sys_diff_dst_md,
                CONV_DIMS, conv3_dst_sizes, dnnl_f32, dnnl_nchw));

        // create backward convolution descriptor
        CHECK(dnnl_convolution_backward_weights_primitive_desc_create(
                &conv3_bwd_weights_pd, Worker -> engine,
                dnnl_convolution_direct, conv3_sys_bwd_src_md,
                conv3_sys_diff_weights_md, conv3_sys_diff_bias_md,
                conv3_sys_diff_dst_md, conv3_strides, conv3_dilation,
                conv3_padding, conv3_padding, conv3_pd, NULL));

        CHECK(dnnl_memory_desc_destroy(conv3_sys_bwd_src_md));
        CHECK(dnnl_memory_desc_destroy(conv3_sys_diff_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv3_sys_diff_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv3_sys_diff_dst_md));
    }

    // 评估并创建存储器和重定位
    const_dnnl_memory_desc_t conv3_bwd_src_md = dnnl_primitive_desc_query_md(
            conv3_bwd_weights_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&pool2_dst_memory, conv3_bwd_src_md,
            Worker -> engine, 1, &conv3_internal_bwd_src_memory,
            &conv3_bwd_reorder_src, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv3_bwd_src_memory = conv3_internal_bwd_src_memory
            ? conv3_internal_bwd_src_memory
            : pool2_dst_memory;

    const_dnnl_memory_desc_t conv3_diff_weights_md
            = dnnl_primitive_desc_query_md(conv3_bwd_weights_pd,
                    dnnl_query_diff_weights_md, 0);
    n_cnn_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
                    // that should happen before the reorder
    CHECK(prepare_reorder(&conv3_user_diff_weights_memory, conv3_diff_weights_md,
            Worker -> engine, 0, &conv3_internal_diff_weights_memory,
            &conv3_reorder_diff_weights, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));
    n_cnn_bwd -= conv3_reorder_diff_weights ? 2 : 1;

    conv3_diff_weights_memory = conv3_internal_diff_weights_memory
            ? conv3_internal_diff_weights_memory
            : conv3_user_diff_weights_memory;

    // 创建diff_bias结果存储器
    const_dnnl_memory_desc_t conv3_diff_bias_md = dnnl_primitive_desc_query_md(
            conv3_bwd_weights_pd, dnnl_query_diff_weights_md, 1);
    CHECK(dnnl_memory_create(&conv3_diff_bias_memory, conv3_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> conv3_diff_bias,
            conv3_diff_bias_memory);

    // 建立conv3_bwd_weights原型
    CHECK(dnnl_primitive_create(&conv3_bwd_weights, conv3_bwd_weights_pd));
    net_cnn_bwd[n_cnn_bwd] = conv3_bwd_weights;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 4);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_SRC,
            conv3_bwd_src_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            conv3_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            conv3_diff_weights_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            conv3_diff_bias_memory);
    n_cnn_bwd++;

    if (conv3_reorder_diff_weights) n_cnn_bwd += 1;

    // pool2_backward
    // 建立pool2_bwd原型描述符
    CHECK(dnnl_pooling_backward_primitive_desc_create(&pool2_bwd_pd,
            Worker -> engine, dnnl_pooling_max, relu2_dst_md,
            pool2_dst_md, pool2_strides, pool2_kernel, pool2_dilation,
            pool2_padding, pool2_padding, pool2_pd, NULL));

    // 评估并创建重定位
    CHECK(prepare_reorder(&conv3_diff_src_memory, pool2_dst_md,
            Worker -> engine, 1, &pool2_internal_diff_dst_memory,
            &pool2_reorder_diff_dst, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    pool2_diff_dst_memory = pool2_internal_diff_dst_memory
            ? pool2_internal_diff_dst_memory
            : conv3_diff_src_memory;

    // 创建结果存储器
    CHECK(dnnl_memory_create(&pool2_diff_src_memory, relu2_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立pool2_bwd原型
    CHECK(dnnl_primitive_create(&pool2_bwd, pool2_bwd_pd));
    net_cnn_bwd[n_cnn_bwd] = pool2_bwd;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_DIFF_DST,
            pool2_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_WORKSPACE,
            pool2_ws_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            pool2_diff_src_memory);
    n_cnn_bwd++;

    // relu2_backward
    // 建立relu2_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&relu2_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, relu2_dst_md,
            conv2_dst_md, RELU2_NSLOP, 0, relu2_pd, NULL));

    // 创造数据存储器
    const_dnnl_memory_desc_t relu2_diff_src_md = dnnl_primitive_desc_query_md(
            relu2_bwd_pd, dnnl_query_diff_src_md, 0);
    CHECK(dnnl_memory_create(&relu2_diff_src_memory, relu2_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立relu2_bwd原型
    CHECK(dnnl_primitive_create(&relu2_bwd, relu2_bwd_pd));
    net_cnn_bwd[n_cnn_bwd] = relu2_bwd;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_SRC,
            conv2_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            pool2_diff_src_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            relu2_diff_src_memory);
    n_cnn_bwd++;

    // conv2_backward
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, conv2_weights_sizes, dnnl_oihw,
            Worker -> engine, Worker -> conv2_diff_weights,
            &conv2_user_diff_weights_memory);

    const_dnnl_memory_desc_t conv2_user_diff_src_md = pool1_dst_md;
    CHECK(dnnl_memory_create(&conv2_user_diff_src_memory,
            conv2_user_diff_src_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));

    // 建立conv2_bwd_data原型描述符
    {
        dnnl_memory_desc_t conv2_sys_bwd_weights_md, conv2_sys_diff_dst_md,
                conv2_sys_diff_src_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_bwd_weights_md,
                CONV_DIMS, conv2_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_diff_dst_md,
                CONV_DIMS, conv2_dst_sizes, dnnl_f32, dnnl_nchw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_diff_src_md,
                CONV_DIMS, pool1_dst_sizes, dnnl_f32, dnnl_nchw));

        // create backward convolution descriptor
        CHECK(dnnl_convolution_backward_data_primitive_desc_create(
                &conv2_bwd_data_pd, Worker -> engine,
                dnnl_convolution_direct, conv2_sys_diff_src_md,
                conv2_sys_bwd_weights_md, conv2_sys_diff_dst_md, conv2_strides,
                conv2_dilation, conv2_padding, conv2_padding, conv2_pd, NULL));

        CHECK(dnnl_memory_desc_destroy(conv2_sys_bwd_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv2_sys_diff_dst_md));
        CHECK(dnnl_memory_desc_destroy(conv2_sys_diff_src_md));
    }

    // 评估并创建存储器和重定位
    const_dnnl_memory_desc_t conv2_bwd_weights_md = dnnl_primitive_desc_query_md(
            conv2_bwd_data_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv2_weights_memory, conv2_bwd_weights_md,
            Worker -> engine, 1, &conv2_internal_bwd_weights_memory,
            &conv2_bwd_reorder_weights, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv2_bwd_weights_memory = conv2_internal_bwd_weights_memory
            ? conv2_internal_bwd_weights_memory
            : conv2_weights_memory;

    const_dnnl_memory_desc_t conv2_diff_dst_md = dnnl_primitive_desc_query_md(
            conv2_bwd_data_pd, dnnl_query_diff_dst_md, 0);
    CHECK(prepare_reorder(&relu2_diff_src_memory, conv2_diff_dst_md,
            Worker -> engine, 1, &conv2_internal_diff_dst_memory,
            &conv2_reorder_diff_dst, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv2_diff_dst_memory = conv2_internal_diff_dst_memory
            ? conv2_internal_diff_dst_memory
            : relu2_diff_src_memory;

    const_dnnl_memory_desc_t conv2_diff_src_md
            = dnnl_primitive_desc_query_md(conv2_bwd_data_pd,
                    dnnl_query_diff_src_md, 0);
    n_cnn_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
                    // that should happen before the reorder
    CHECK(prepare_reorder(&conv2_user_diff_src_memory, conv2_diff_src_md,
            Worker -> engine, 0, &conv2_internal_diff_src_memory,
            &conv2_reorder_diff_src, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));
    n_cnn_bwd -= conv2_reorder_diff_src ? 2 : 1;

    conv2_diff_src_memory = conv2_internal_diff_src_memory
            ? conv2_internal_diff_src_memory
            : conv2_user_diff_src_memory;

    // 建立conv2_bwd_data原型
    CHECK(dnnl_primitive_create(&conv2_bwd_data, conv2_bwd_data_pd));
    net_cnn_bwd[n_cnn_bwd] = conv2_bwd_data;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_WEIGHTS,
            conv2_bwd_weights_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            conv2_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            conv2_diff_src_memory);
    n_cnn_bwd++;

    if (conv2_reorder_diff_src) n_cnn_bwd += 1;

    // 建立conv2_bwd_weights原型描述符
    {
        // memory descriptors should be in format `any` to allow backward
        // convolution for
        // weights to chose the format it prefers for best performance
        dnnl_memory_desc_t conv2_sys_bwd_src_md, conv2_sys_diff_weights_md,
                conv2_sys_diff_bias_md, conv2_sys_diff_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_bwd_src_md,
                CONV_DIMS, pool1_dst_sizes, dnnl_f32, dnnl_nchw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_diff_weights_md,
                CONV_DIMS, conv2_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_diff_bias_md, 1,
                conv2_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv2_sys_diff_dst_md,
                CONV_DIMS, conv2_dst_sizes, dnnl_f32, dnnl_nchw));

        // create backward convolution descriptor
        CHECK(dnnl_convolution_backward_weights_primitive_desc_create(
                &conv2_bwd_weights_pd, Worker -> engine,
                dnnl_convolution_direct, conv2_sys_bwd_src_md,
                conv2_sys_diff_weights_md, conv2_sys_diff_bias_md,
                conv2_sys_diff_dst_md, conv2_strides, conv2_dilation,
                conv2_padding, conv2_padding, conv2_pd, NULL));

        CHECK(dnnl_memory_desc_destroy(conv2_sys_bwd_src_md));
        CHECK(dnnl_memory_desc_destroy(conv2_sys_diff_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv2_sys_diff_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv2_sys_diff_dst_md));
    }

    // 评估并创建存储器和重定位
    const_dnnl_memory_desc_t conv2_bwd_src_md = dnnl_primitive_desc_query_md(
            conv2_bwd_weights_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&pool1_dst_memory, conv2_bwd_src_md,
            Worker -> engine, 1, &conv2_internal_bwd_src_memory,
            &conv2_bwd_reorder_src, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv2_bwd_src_memory = conv2_internal_bwd_src_memory
            ? conv2_internal_bwd_src_memory
            : pool1_dst_memory;

    const_dnnl_memory_desc_t conv2_diff_weights_md
            = dnnl_primitive_desc_query_md(conv2_bwd_weights_pd,
                    dnnl_query_diff_weights_md, 0);
    n_cnn_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
                    // that should happen before the reorder
    CHECK(prepare_reorder(&conv2_user_diff_weights_memory, conv2_diff_weights_md,
            Worker -> engine, 0, &conv2_internal_diff_weights_memory,
            &conv2_reorder_diff_weights, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));
    n_cnn_bwd -= conv2_reorder_diff_weights ? 2 : 1;

    conv2_diff_weights_memory = conv2_internal_diff_weights_memory
            ? conv2_internal_diff_weights_memory
            : conv2_user_diff_weights_memory;

    // 创建diff_bias结果存储器
    const_dnnl_memory_desc_t conv2_diff_bias_md = dnnl_primitive_desc_query_md(
            conv2_bwd_weights_pd, dnnl_query_diff_weights_md, 1);
    CHECK(dnnl_memory_create(&conv2_diff_bias_memory, conv2_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> conv2_diff_bias,
            conv2_diff_bias_memory);

    // 建立conv2_bwd_weights原型
    CHECK(dnnl_primitive_create(&conv2_bwd_weights, conv2_bwd_weights_pd));
    net_cnn_bwd[n_cnn_bwd] = conv2_bwd_weights;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 4);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_SRC,
            conv2_bwd_src_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            conv2_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            conv2_diff_weights_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            conv2_diff_bias_memory);
    n_cnn_bwd++;

    if (conv2_reorder_diff_weights) n_cnn_bwd += 1;

    // pool1_backward
    // 建立pool1_bwd原型描述符
    CHECK(dnnl_pooling_backward_primitive_desc_create(&pool1_bwd_pd,
            Worker -> engine, dnnl_pooling_max, relu1_dst_md,
            pool1_dst_md, pool1_strides, pool1_kernel, pool1_dilation,
            pool1_padding, pool1_padding, pool1_pd, NULL));

    // 评估并创建重定位
    CHECK(prepare_reorder(&conv2_diff_src_memory, pool1_dst_md,
            Worker -> engine, 1, &pool1_internal_diff_dst_memory,
            &pool1_reorder_diff_dst, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    pool1_diff_dst_memory = pool1_internal_diff_dst_memory
            ? pool1_internal_diff_dst_memory
            : conv2_diff_src_memory;

    // 创建结果存储器
    CHECK(dnnl_memory_create(&pool1_diff_src_memory, relu1_dst_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立pool1_bwd原型
    CHECK(dnnl_primitive_create(&pool1_bwd, pool1_bwd_pd));
    net_cnn_bwd[n_cnn_bwd] = pool1_bwd;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_DIFF_DST,
            pool1_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_WORKSPACE,
            pool1_ws_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            pool1_diff_src_memory);
    n_cnn_bwd++;

    // relu1_backward
    // 建立relu1_bwd原型描述符
    CHECK(dnnl_eltwise_backward_primitive_desc_create(&relu1_bwd_pd,
            Worker -> engine, dnnl_eltwise_relu, relu1_dst_md,
            conv1_dst_md, RELU1_NSLOP, 0, relu1_pd, NULL));

    // 创造数据存储器
    const_dnnl_memory_desc_t relu1_diff_src_md = dnnl_primitive_desc_query_md(
            relu1_bwd_pd, dnnl_query_diff_src_md, 0);
    CHECK(dnnl_memory_create(&relu1_diff_src_memory, relu1_diff_src_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));

    // 建立relu1_bwd原型
    CHECK(dnnl_primitive_create(&relu1_bwd, relu1_bwd_pd));
    net_cnn_bwd[n_cnn_bwd] = relu1_bwd;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_SRC,
            conv1_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            pool1_diff_src_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            relu1_diff_src_memory);
    n_cnn_bwd++;

    // conv1_backward
    // 创造user数据存储器
    init_data_memory(CONV_DIMS, conv1_weights_sizes, dnnl_oihw,
            Worker -> engine, Worker -> conv1_diff_weights,
            &conv1_user_diff_weights_memory);

    const_dnnl_memory_desc_t conv1_user_diff_src_md = conv1_src_md;
    CHECK(dnnl_memory_create(&conv1_user_diff_src_memory,
            conv1_user_diff_src_md, Worker -> engine,
            DNNL_MEMORY_ALLOCATE));

    // 建立conv1_bwd_data原型描述符
    {
        dnnl_memory_desc_t conv1_sys_bwd_weights_md, conv1_sys_diff_dst_md,
                conv1_sys_diff_src_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_bwd_weights_md,
                CONV_DIMS, conv1_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_diff_dst_md,
                CONV_DIMS, conv1_dst_sizes, dnnl_f32, dnnl_nchw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_diff_src_md,
                CONV_DIMS, conv1_src_sizes, dnnl_f32, dnnl_nchw));

        // create backward convolution descriptor
        CHECK(dnnl_convolution_backward_data_primitive_desc_create(
                &conv1_bwd_data_pd, Worker -> engine,
                dnnl_convolution_direct, conv1_sys_diff_src_md,
                conv1_sys_bwd_weights_md, conv1_sys_diff_dst_md, conv1_strides,
                conv1_dilation, conv1_padding, conv1_padding, conv1_pd, NULL));

        CHECK(dnnl_memory_desc_destroy(conv1_sys_bwd_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_diff_dst_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_diff_src_md));
    }

    // 评估并创建存储器和重定位
    const_dnnl_memory_desc_t conv1_bwd_weights_md = dnnl_primitive_desc_query_md(
            conv1_bwd_data_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv1_weights_memory, conv1_bwd_weights_md,
            Worker -> engine, 1, &conv1_internal_bwd_weights_memory,
            &conv1_bwd_reorder_weights, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv1_bwd_weights_memory = conv1_internal_bwd_weights_memory
            ? conv1_internal_bwd_weights_memory
            : conv1_weights_memory;

    const_dnnl_memory_desc_t conv1_diff_dst_md = dnnl_primitive_desc_query_md(
            conv1_bwd_data_pd, dnnl_query_diff_dst_md, 0);
    CHECK(prepare_reorder(&relu1_diff_src_memory, conv1_diff_dst_md,
            Worker -> engine, 1, &conv1_internal_diff_dst_memory,
            &conv1_reorder_diff_dst, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv1_diff_dst_memory = conv1_internal_diff_dst_memory
            ? conv1_internal_diff_dst_memory
            : relu1_diff_src_memory;

    const_dnnl_memory_desc_t conv1_diff_src_md
            = dnnl_primitive_desc_query_md(conv1_bwd_data_pd,
                    dnnl_query_diff_src_md, 0);
    n_cnn_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
                    // that should happen before the reorder
    CHECK(prepare_reorder(&conv1_user_diff_src_memory, conv1_diff_src_md,
            Worker -> engine, 0, &conv1_internal_diff_src_memory,
            &conv1_reorder_diff_src, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));
    n_cnn_bwd -= conv1_reorder_diff_src ? 2 : 1;

    conv1_diff_src_memory = conv1_internal_diff_src_memory
            ? conv1_internal_diff_src_memory
            : conv1_user_diff_src_memory;

    // 建立conv1_bwd_data原型
    CHECK(dnnl_primitive_create(&conv1_bwd_data, conv1_bwd_data_pd));
    net_cnn_bwd[n_cnn_bwd] = conv1_bwd_data;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 3);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_WEIGHTS,
            conv1_bwd_weights_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            conv1_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_SRC,
            conv1_diff_src_memory);
    n_cnn_bwd++;

    if (conv1_reorder_diff_src) n_cnn_bwd += 1;

    // 建立conv1_bwd_weights原型描述符
    {
        // memory descriptors should be in format `any` to allow backward
        // convolution for
        // weights to chose the format it prefers for best performance
        dnnl_memory_desc_t conv1_sys_bwd_src_md, conv1_sys_diff_weights_md,
                conv1_sys_diff_bias_md, conv1_sys_diff_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_bwd_src_md,
                CONV_DIMS, conv1_src_sizes, dnnl_f32, dnnl_nchw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_diff_weights_md,
                CONV_DIMS, conv1_weights_sizes, dnnl_f32, dnnl_oihw));
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_diff_bias_md, 1,
                conv1_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv1_sys_diff_dst_md,
                CONV_DIMS, conv1_dst_sizes, dnnl_f32, dnnl_nchw));

        // create backward convolution descriptor
        CHECK(dnnl_convolution_backward_weights_primitive_desc_create(
                &conv1_bwd_weights_pd, Worker -> engine,
                dnnl_convolution_direct, conv1_sys_bwd_src_md,
                conv1_sys_diff_weights_md, conv1_sys_diff_bias_md,
                conv1_sys_diff_dst_md, conv1_strides, conv1_dilation,
                conv1_padding, conv2_padding, conv1_pd, NULL));

        CHECK(dnnl_memory_desc_destroy(conv1_sys_bwd_src_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_diff_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_diff_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv1_sys_diff_dst_md));
    }

    // 评估并创建存储器和重定位
    const_dnnl_memory_desc_t conv1_bwd_src_md = dnnl_primitive_desc_query_md(
            conv1_bwd_weights_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&conv1_src_memory, conv1_bwd_src_md,
            Worker -> engine, 1, &conv1_internal_bwd_src_memory,
            &conv1_bwd_reorder_src, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));

    conv1_bwd_src_memory = conv1_internal_bwd_src_memory
            ? conv1_internal_bwd_src_memory
            : conv1_src_memory;

    const_dnnl_memory_desc_t conv1_diff_weights_md
            = dnnl_primitive_desc_query_md(conv1_bwd_weights_pd,
                    dnnl_query_diff_weights_md, 0);
    n_cnn_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
                    // that should happen before the reorder
    CHECK(prepare_reorder(&conv1_user_diff_weights_memory, conv1_diff_weights_md,
            Worker -> engine, 0, &conv1_internal_diff_weights_memory,
            &conv1_reorder_diff_weights, &n_cnn_bwd, net_cnn_bwd,
            net_cnn_bwd_args));
    n_cnn_bwd -= conv1_reorder_diff_weights ? 2 : 1;

    conv1_diff_weights_memory = conv1_internal_diff_weights_memory
            ? conv1_internal_diff_weights_memory
            : conv1_user_diff_weights_memory;

    // 创建diff_bias结果存储器
    const_dnnl_memory_desc_t conv1_diff_bias_md = dnnl_primitive_desc_query_md(
            conv1_bwd_weights_pd, dnnl_query_diff_weights_md, 1);
    CHECK(dnnl_memory_create(&conv1_diff_bias_memory, conv1_diff_bias_md,
            Worker -> engine, DNNL_MEMORY_ALLOCATE));
    write_to_dnnl_memory(Worker -> conv1_diff_bias,
            conv1_diff_bias_memory);

    // 建立conv1_bwd_weights原型
    CHECK(dnnl_primitive_create(&conv1_bwd_weights, conv1_bwd_weights_pd));
    net_cnn_bwd[n_cnn_bwd] = conv1_bwd_weights;
    prepare_arg_node(&net_cnn_bwd_args[n_cnn_bwd], 4);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[0], DNNL_ARG_SRC,
            conv1_bwd_src_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[1], DNNL_ARG_DIFF_DST,
            conv1_diff_dst_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[2], DNNL_ARG_DIFF_WEIGHTS,
            conv1_diff_weights_memory);
    set_arg(&net_cnn_bwd_args[n_cnn_bwd].args[3], DNNL_ARG_DIFF_BIAS,
            conv1_diff_bias_memory);
    n_cnn_bwd++;

    if (conv1_reorder_diff_weights) n_cnn_bwd += 1;
	
/*--------------------执行神经网络模型---------------------------------------*/
    /*----------------- CNN Forward Stream ----------------------------------*/
    // 加载CNN数据源，reshape and load
    for(uint32_t in = 0; in < BATCH; in++)
    {
        for(uint32_t ic = 0; ic < IN_CONV1_C; ic++)
        {
            for(uint32_t iw = 0; iw < IN_CONV1_W; iw++)
            {
                uint32_t index = in*IN_CONV1_C*IN_CONV1_W + ic*IN_CONV1_W + iw;
                switch(ic)
                {
                    case 0:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].I1;
                        break;
                    case 1:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].I2;
                        break;
                    case 2:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].I3;
                        break;
                    case 3:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].Itot;
                        break;
                    case 4:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].ia;
                        break;
                    case 5:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].ib;
                        break;
                    case 6:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].ic;
                        break;
                    case 7:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].V1;
                        break;
                    case 8:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].V2;
                        break;
                    case 9:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].V3;
                        break;
                    case 10:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].Vzero;
                        break;
                    case 11:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].H1;
                        break;
                    case 12:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].H2;
                        break;
                    case 13:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].H3;
                        break;
                    case 14:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].L1;
                        break;
                    case 15:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].L2;
                        break;
                    case 16:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].L3;
                        break;
                    case 17:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].theta;
                        break;
                    case 18:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].etheta;
                        break;
                    case 19:
                        Worker -> conv1_src[index]
                                = (float) MotorSimuBlock[iw].omega;
                        break;
                    default:
                        printf("error while reshaping data!\r\n");
                        break;
                }
            }
        }
    }
    write_to_dnnl_memory(Worker -> conv1_src, conv1_src_memory);
    // 执行CNN_Forward
    for (uint32_t i = 0; i < n_cnn_fwd; ++i)
    {CHECK(dnnl_primitive_execute(net_cnn_fwd[i], Worker -> stream,
            net_cnn_fwd_args[i].nargs, net_cnn_fwd_args[i].args));}
    // 输出执行结果
    float *conv1_dst_get
            = (float *)malloc(product(conv1_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *conv1_dst_mem = conv1_dst_get;
    read_from_dnnl_memory(conv1_dst_mem, conv1_dst_memory);

    float *pool1_dst_get
            = (float *)malloc(product(pool1_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *pool1_dst_mem = pool1_dst_get;
    read_from_dnnl_memory(pool1_dst_mem, pool1_dst_memory);

    float *conv2_dst_get
            = (float *)malloc(product(conv2_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *conv2_dst_mem = conv2_dst_get;
    read_from_dnnl_memory(conv2_dst_mem, conv2_dst_memory);

    float *pool2_dst_get
            = (float *)malloc(product(pool2_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *pool2_dst_mem = pool2_dst_get;
    read_from_dnnl_memory(pool2_dst_mem, pool2_dst_memory);

    float *conv3_dst_get
            = (float *)malloc(product(conv3_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *conv3_dst_mem = conv3_dst_get;
    read_from_dnnl_memory(conv3_dst_mem, conv3_dst_memory);

    float *pool3_dst_get
            = (float *)malloc(product(pool3_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *pool3_dst_mem = pool3_dst_get;
    read_from_dnnl_memory(pool3_dst_mem, pool3_dst_memory);
/*
#ifdef RL_NET_OUTPUT
    fprintf(fp3, "conv1_src:\r\n");
    fprintMatrix(fp3, Worker -> conv1_src, 3, BATCH, IN_CONV1_C,
            1, IN_CONV1_W);
    fprintf(fp3, "conv1_weights:\r\n");
    fprintMatrix(fp3, Worker -> conv1_weights, 3, CONV1_RELU1_C,
            IN_CONV1_C, 1, CONV1_KERNEL_W);
    fprintf(fp3, "conv1_bias:\r\n");
    fprintMatrix(fp3, Worker -> conv1_bias, 2, BATCH,
            CONV1_RELU1_C, 1, 1);
    fprintf(fp3, "conv1_dst:\r\n");
    fprintMatrix(fp3, conv1_dst_get, 3, BATCH, CONV1_RELU1_C,
            1, CONV1_RELU1_W);
    fprintf(fp3, "pool1_dst:\r\n");
    fprintMatrix(fp3, pool1_dst_get, 3, BATCH, POOL1_CONV2_C,
            1, POOL1_CONV2_W);
    fprintf(fp3, "conv2_weights:\r\n");
    fprintMatrix(fp3, Worker -> conv2_weights, 3, CONV2_RELU2_C,
            POOL1_CONV2_C, 1, CONV2_KERNEL_W);
    fprintf(fp3, "conv2_bias:\r\n");
    fprintMatrix(fp3, Worker -> conv2_bias, 2, BATCH,
            CONV2_RELU2_C, 1, 1);
    fprintf(fp3, "conv2_dst:\r\n");
    fprintMatrix(fp3, conv2_dst_get, 3, BATCH, CONV2_RELU2_C,
            1, CONV2_RELU2_W);
    fprintf(fp3, "pool2_dst:\r\n");
    fprintMatrix(fp3, pool2_dst_get, 3, BATCH, POOL2_CONV3_C,
            1, POOL2_CONV3_W);
    fprintf(fp3, "conv3_weights:\r\n");
    fprintMatrix(fp3, Worker -> conv3_weights, 3, CONV3_RELU3_C,
            POOL2_CONV3_C, 1, CONV3_KERNEL_W);
    fprintf(fp3, "conv3_bias:\r\n");
    fprintMatrix(fp3, Worker -> conv3_bias, 2, BATCH,
            CONV3_RELU3_C, 1, 1);
    fprintf(fp3, "conv3_dst:\r\n");
    fprintMatrix(fp3, conv3_dst_get, 3, BATCH, CONV3_RELU3_C,
            1, CONV3_RELU3_W);
    fprintf(fp3, "pool3_dst:\r\n");
    fprintMatrix(fp3, pool3_dst_get, 3, BATCH, POOL3_RSHA_C,
            1, POOL3_RSHA_W);
#endif
*/
    /*----------------- Actor Forward Stream --------------------------------*/
    // 加载Actor数据源
    uint32_t cnn_result_index = BATCH*RSHA_DNSEA1_DNSEC1_C;
    for (uint32_t i = 0; i < cnn_result_index; i++)
    {
        Worker -> dnseA1_src[i] = pool3_dst_get[i];
    }
    write_to_dnnl_memory(Worker -> dnseA1_src, dnseA1_src_memory);
    // 执行Actor_Forward
    for (uint32_t i = 0; i < n_actor_fwd; ++i)
    {CHECK(dnnl_primitive_execute(net_actor_fwd[i], Worker -> stream,
            net_actor_fwd_args[i].nargs, net_actor_fwd_args[i].args));}
    // 输出执行结果
    float *dnseA1_src_get
            = (float *)malloc(product(dnseA1_src_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA1_src_mem = dnseA1_src_get;
    read_from_dnnl_memory(dnseA1_src_mem, dnseA1_src_memory);

    float *dnseA1_weights_get
            = (float *)malloc(product(dnseA1_weights_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA1_weights_mem = dnseA1_weights_get;
    read_from_dnnl_memory(dnseA1_weights_mem, dnseA1_weights_memory);

    float *dnseA1_bias_get
            = (float *)malloc(product(dnseA1_bias_sizes, 1) * sizeof(float));
    void *dnseA1_bias_mem = dnseA1_bias_get;
    read_from_dnnl_memory(dnseA1_bias_mem, dnseA1_bias_memory);

    float *dnseA1_dst_get
            = (float *)malloc(product(dnseA1_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA1_dst_mem = dnseA1_dst_get;
    read_from_dnnl_memory(dnseA1_dst_mem, dnseA1_dst_memory);

    float *reluA1_dst_get
            = (float *)malloc(product(reluA1_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *reluA1_dst_mem = reluA1_dst_get;
    read_from_dnnl_memory(reluA1_dst_mem, reluA1_dst_memory);

    float *dnseA5_dst_get
            = (float *)malloc(product(dnseA5_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA5_dst_mem = dnseA5_dst_get;
    read_from_dnnl_memory(dnseA5_dst_mem, dnseA5_dst_memory);
/*
#ifdef RL_NET_OUTPUT
    fprintf(fp3, "dnseA1_src:\r\n");
    fprintMatrix(fp3, dnseA1_src_get, 2, BATCH, RSHA_DNSEA1_DNSEC1_C, 1, 1);
    fprintf(fp3, "dnseA1_weights:\r\n");
    fprintMatrix(fp3, dnseA1_weights_get, 2, RSHA_DNSEA1_DNSEC1_C,
            DNSEA1_RELUA1_C, 1, 1);
    fprintf(fp3, "dnseA1_bias:\r\n");
    fprintMatrix(fp3, dnseA1_bias_get, 2, BATCH, DNSEA1_RELUA1_C, 1, 1);
    fprintf(fp3, "dnseA1_dst:\r\n");
    fprintMatrix(fp3, dnseA1_dst_get, 2, BATCH, DNSEA1_RELUA1_C, 1, 1);
    fprintf(fp3, "reluA1_dst:\r\n");
    fprintMatrix(fp3, reluA1_dst_get, 2, BATCH, RELUA1_DNSEA2_C, 1, 1);
    fprintf(fp3, "dnseA5_dst:\r\n");
    fprintMatrix(fp3, dnseA5_dst_get, 2, BATCH, DNSEA5_OUTA_C, 1, 1);
#endif
*/
    /*----------------- Critic Forward Stream -------------------------------*/
    // 加载Critic数据源
    for (uint32_t i = 0; i < cnn_result_index; i++)
    {
        Worker -> dnseC1_src[i] = pool3_dst_get[i];
    }
    write_to_dnnl_memory(Worker -> dnseC1_src, dnseC1_src_memory);
    // 执行Critic_Forward
    for (uint32_t i = 0; i < n_critic_fwd; ++i)
    {CHECK(dnnl_primitive_execute(net_critic_fwd[i],
            Worker -> stream, net_critic_fwd_args[i].nargs,
            net_critic_fwd_args[i].args));}
    // 输出执行结果
    float *dnseC1_src_get
            = (float *)malloc(product(dnseC1_src_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC1_src_mem = dnseC1_src_get;
    read_from_dnnl_memory(dnseC1_src_mem, dnseC1_src_memory);

    float *dnseC5_dst_get
            = (float *)malloc(product(dnseC5_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC5_dst_mem = dnseC5_dst_get;
    read_from_dnnl_memory(dnseC5_dst_mem, dnseC5_dst_memory);
/*
#ifdef RL_NET_OUTPUT
    fprintf(fp3, "dnseC1_src:\r\n");
    fprintMatrix(fp3, dnseC1_src_get, 2, BATCH, RSHA_DNSEA1_DNSEC1_C, 1, 1);
    fprintf(fp3, "dnseC5_dst:\r\n");
    fprintMatrix(fp3, dnseC5_dst_get, 2, BATCH, DNSEC5_OUTC_C, 1, 1);
#endif
*/
    /*----------------- Actor Backward Stream -------------------------------*/
    // 加载Actor_bwd数据源
    cnn_result_index = BATCH*DNSEA5_OUTA_C;
    for (uint32_t i = 0; i < cnn_result_index; i++)
    {
        Worker -> dnseA5_diff_dst[i] = dnseA5_dst_get[i];
    }
    write_to_dnnl_memory(Worker -> dnseA5_diff_dst,
            dnseA5_diff_dst_memory);
    // 执行Actor_Backward
    for (uint32_t i = 0; i < n_actor_bwd; ++i)
    {CHECK(dnnl_primitive_execute(net_actor_bwd[i],
            Worker -> stream, net_actor_bwd_args[i].nargs,
            net_actor_bwd_args[i].args));}
    // 输出执行结果
    float *dnseA5_diff_dst_get
            = (float *)malloc(product(dnseA5_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA5_diff_dst_mem = dnseA5_diff_dst_get;
    read_from_dnnl_memory(dnseA5_diff_dst_mem, dnseA5_diff_dst_memory);

    float *dnseA5_diff_src_get
            = (float *)malloc(product(reluA4_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA5_diff_src_mem = dnseA5_diff_src_get;
    read_from_dnnl_memory(dnseA5_diff_src_mem, dnseA5_diff_src_memory);

    float *dnseA4_diff_src_get
            = (float *)malloc(product(reluA3_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA4_diff_src_mem = dnseA4_diff_src_get;
    read_from_dnnl_memory(dnseA4_diff_src_mem, dnseA4_diff_src_memory);

    float *dnseA3_diff_src_get
            = (float *)malloc(product(reluA3_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA3_diff_src_mem = dnseA3_diff_src_get;
    read_from_dnnl_memory(dnseA3_diff_src_mem, dnseA3_diff_src_memory);

    float *dnseA2_diff_src_get
            = (float *)malloc(product(reluA1_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA2_diff_src_mem = dnseA2_diff_src_get;
    read_from_dnnl_memory(dnseA2_diff_src_mem, dnseA2_diff_src_memory);

    float *dnseA1_diff_src_get
            = (float *)malloc(product(dnseA1_src_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseA1_diff_src_mem = dnseA1_diff_src_get;
    read_from_dnnl_memory(dnseA1_diff_src_mem, dnseA1_diff_src_memory);
/*
#ifdef RL_NET_OUTPUT
    fprintf(fp3, "dnseA5_diff_dst:\r\n");
    fprintMatrix(fp3, dnseA5_diff_dst_get, 2, BATCH, DNSEA5_OUTA_C, 1, 1);
    fprintf(fp3, "dnseA5_diff_src:\r\n");
    fprintMatrix(fp3, dnseA5_diff_src_get, 2, BATCH, RELUA4_DNSEA5_C, 1, 1);
    fprintf(fp3, "dnseA4_diff_src:\r\n");
    fprintMatrix(fp3, dnseA4_diff_src_get, 2, BATCH, RELUA3_DNSEA4_C, 1, 1);
    fprintf(fp3, "dnseA3_diff_src:\r\n");
    fprintMatrix(fp3, dnseA3_diff_src_get, 2, BATCH, RELUA2_DNSEA3_C, 1, 1);
    fprintf(fp3, "dnseA2_diff_src:\r\n");
    fprintMatrix(fp3, dnseA2_diff_src_get, 2, BATCH, RELUA1_DNSEA2_C, 1, 1);
    fprintf(fp3, "dnseA1_diff_src:\r\n");
    fprintMatrix(fp3, dnseA1_diff_src_get, 2, BATCH,
            RSHA_DNSEA1_DNSEC1_C, 1, 1);
#endif
*/
    /*----------------- Critic Backward Stream ------------------------------*/
    // 加载Critic_bwd数据源
    Worker -> dnseC5_diff_dst[0] = dnseC5_dst_get[0];
    write_to_dnnl_memory(Worker -> dnseC5_diff_dst,
            dnseC5_diff_dst_memory);
    // 执行Critic_Backward
    for (uint32_t i = 0; i < n_critic_bwd; ++i)
    {CHECK(dnnl_primitive_execute(net_critic_bwd[i],
            Worker -> stream, net_critic_bwd_args[i].nargs,
            net_critic_bwd_args[i].args));}
    // 输出执行结果
    float *dnseC5_diff_dst_get
            = (float *)malloc(product(dnseC5_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC5_diff_dst_mem = dnseC5_diff_dst_get;
    read_from_dnnl_memory(dnseC5_diff_dst_mem, dnseC5_diff_dst_memory);

    float *dnseC5_diff_src_get
            = (float *)malloc(product(reluC4_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC5_diff_src_mem = dnseC5_diff_src_get;
    read_from_dnnl_memory(dnseC5_diff_src_mem, dnseC5_diff_src_memory);

    float *dnseC4_diff_src_get
            = (float *)malloc(product(reluC3_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC4_diff_src_mem = dnseC4_diff_src_get;
    read_from_dnnl_memory(dnseC4_diff_src_mem, dnseC4_diff_src_memory);

    float *dnseC3_diff_src_get
            = (float *)malloc(product(reluC2_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC3_diff_src_mem = dnseC3_diff_src_get;
    read_from_dnnl_memory(dnseC3_diff_src_mem, dnseC3_diff_src_memory);

    float *dnseC2_diff_src_get
            = (float *)malloc(product(reluC1_dst_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC2_diff_src_mem = dnseC2_diff_src_get;
    read_from_dnnl_memory(dnseC2_diff_src_mem, dnseC2_diff_src_memory);

    float *dnseC1_diff_src_get
            = (float *)malloc(product(dnseC1_src_sizes, DNSE_DIMS)
                    * sizeof(float));
    void *dnseC1_diff_src_mem = dnseC1_diff_src_get;
    read_from_dnnl_memory(dnseC1_diff_src_mem, dnseC1_diff_src_memory);
/*
#ifdef RL_NET_OUTPUT
    fprintf(fp3, "dnseC5_diff_dst:\r\n");
    fprintMatrix(fp3, dnseC5_diff_dst_get, 2, BATCH, DNSEC5_OUTC_C, 1, 1);
    fprintf(fp3, "dnseC5_diff_src:\r\n");
    fprintMatrix(fp3, dnseC5_diff_src_get, 2, BATCH, RELUC4_DNSEC5_C, 1, 1);
    fprintf(fp3, "dnseC4_diff_src:\r\n");
    fprintMatrix(fp3, dnseC4_diff_src_get, 2, BATCH, RELUC3_DNSEC4_C, 1, 1);
    fprintf(fp3, "dnseC3_diff_src:\r\n");
    fprintMatrix(fp3, dnseC3_diff_src_get, 2, BATCH, RELUC2_DNSEC3_C, 1, 1);
    fprintf(fp3, "dnseC2_diff_src:\r\n");
    fprintMatrix(fp3, dnseC2_diff_src_get, 2, BATCH, RELUC1_DNSEC2_C, 1, 1);
    fprintf(fp3, "dnseC1_diff_src:\r\n");
    fprintMatrix(fp3, dnseC1_diff_src_get, 2, BATCH,
            RSHA_DNSEA1_DNSEC1_C, 1, 1);
#endif
*/
    /*----------------- CNN Backward Stream ---------------------------------*/
    // 加载CNN_bwd数据源
    cnn_result_index = BATCH*RSHA_DNSEA1_DNSEC1_C;
    for (uint32_t i = 0; i < cnn_result_index; i++)
    {
        Worker -> pool3_diff_dst[i] = dnseC1_diff_src_get[i];
    }
    write_to_dnnl_memory(Worker -> pool3_diff_dst,
            pool3_diff_dst_memory);
    // 执行CNN_Backward
    for (uint32_t i = 0; i < n_cnn_bwd; ++i)
    {CHECK(dnnl_primitive_execute(net_cnn_bwd[i],
            Worker -> stream, net_cnn_bwd_args[i].nargs,
            net_cnn_bwd_args[i].args));}
    // 输出执行结果
    float *pool3_diff_dst_get
            = (float *)malloc(product(pool3_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *pool3_diff_dst_mem = pool3_diff_dst_get;
    read_from_dnnl_memory(pool3_diff_dst_mem, pool3_diff_dst_memory);

    float *pool3_diff_src_get
            = (float *)malloc(product(conv3_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *pool3_diff_src_mem = pool3_diff_src_get;
    read_from_dnnl_memory(pool3_diff_src_mem, pool3_diff_src_memory);

    float *relu3_diff_src_get
            = (float *)malloc(product(conv3_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *relu3_diff_src_mem = relu3_diff_src_get;
    read_from_dnnl_memory(relu3_diff_src_mem, relu3_diff_src_memory);

    float *conv3_diff_src_get
            = (float *)malloc(product(pool2_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *conv3_diff_src_mem = conv3_diff_src_get;
    read_from_dnnl_memory(conv3_diff_src_mem, conv3_diff_src_memory);

    float *pool2_diff_src_get
            = (float *)malloc(product(conv2_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *pool2_diff_src_mem = pool2_diff_src_get;
    read_from_dnnl_memory(pool2_diff_src_mem, pool2_diff_src_memory);

    float *relu2_diff_src_get
            = (float *)malloc(product(conv2_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *relu2_diff_src_mem = relu2_diff_src_get;
    read_from_dnnl_memory(relu2_diff_src_mem, relu2_diff_src_memory);

    float *conv2_diff_src_get
            = (float *)malloc(product(pool1_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *conv2_diff_src_mem = conv2_diff_src_get;
    read_from_dnnl_memory(conv2_diff_src_mem, conv2_diff_src_memory);

    float *pool1_diff_src_get
            = (float *)malloc(product(conv1_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *pool1_diff_src_mem = pool1_diff_src_get;
    read_from_dnnl_memory(pool1_diff_src_mem, pool1_diff_src_memory);

    float *relu1_diff_src_get
            = (float *)malloc(product(conv1_dst_sizes, CONV_DIMS)
                    * sizeof(float));
    void *relu1_diff_src_mem = relu1_diff_src_get;
    read_from_dnnl_memory(relu1_diff_src_mem, relu1_diff_src_memory);

    float *conv1_diff_src_get
            = (float *)malloc(product(conv1_src_sizes, CONV_DIMS)
                    * sizeof(float));
    void *conv1_diff_src_mem = conv1_diff_src_get;
    read_from_dnnl_memory(conv1_diff_src_mem, conv1_diff_src_memory);

#ifdef RL_NET_OUTPUT
    fprintf(fp3, "pool3_diff_dst:\r\n");
    fprintMatrix(fp3, pool3_diff_dst_get, 3, BATCH, POOL3_RSHA_C, 1,
            POOL3_RSHA_W);
    fprintf(fp3, "pool3_diff_src:\r\n");
    fprintMatrix(fp3, pool3_diff_src_get, 3, BATCH, RELU3_POOL3_C, 1,
            RELU3_POOL3_W);
    fprintf(fp3, "relu3_diff_src:\r\n");
    fprintMatrix(fp3, relu3_diff_src_get, 3, BATCH, CONV3_RELU3_C, 1,
            CONV3_RELU3_W);
    fprintf(fp3, "conv3_diff_src:\r\n");
    fprintMatrix(fp3, conv3_diff_src_get, 3, BATCH, POOL2_CONV3_C, 1,
            POOL2_CONV3_W);
    fprintf(fp3, "pool2_diff_src:\r\n");
    fprintMatrix(fp3, pool2_diff_src_get, 3, BATCH, RELU2_POOL2_C, 1,
            RELU2_POOL2_W);
    fprintf(fp3, "relu2_diff_src:\r\n");
    fprintMatrix(fp3, relu2_diff_src_get, 3, BATCH, CONV2_RELU2_C, 1,
            CONV2_RELU2_W);
    fprintf(fp3, "conv2_diff_src:\r\n");
    fprintMatrix(fp3, conv2_diff_src_get, 3, BATCH, POOL1_CONV2_C, 1,
            POOL1_CONV2_W);
    fprintf(fp3, "pool1_diff_src:\r\n");
    fprintMatrix(fp3, pool1_diff_src_get, 3, BATCH, RELU1_POOL1_C, 1,
            RELU1_POOL1_W);
    fprintf(fp3, "relu1_diff_src:\r\n");
    fprintMatrix(fp3, relu1_diff_src_get, 3, BATCH, CONV1_RELU1_C, 1,
            CONV1_RELU1_W);
    fprintf(fp3, "conv1_diff_src:\r\n");
    fprintMatrix(fp3, conv1_diff_src_get, 3, BATCH, IN_CONV1_C, 1, IN_CONV1_W);
#endif

	appendLoad(fp1, DataImputBlock);
	printf("thread%d append time: %lf\r\n", threadIndex,
			DataImputBlock[0].T);
	
/*--------------------解除神经网络模型---------------------------------------*/

    // 解除CNN Forward
    for (uint32_t i = 0; i < n_cnn_fwd; ++i)
        {free_arg_node(&net_cnn_fwd_args[i]);}
    
    CHECK(dnnl_primitive_desc_destroy(pool1_pd));
    CHECK(dnnl_primitive_desc_destroy(relu1_pd));
    CHECK(dnnl_primitive_desc_destroy(conv1_pd));

    dnnl_memory_destroy(conv1_user_src_memory);
    dnnl_memory_destroy(conv1_user_weights_memory);
    dnnl_memory_destroy(conv1_bias_memory);
    dnnl_memory_destroy(conv1_internal_src_memory);
    dnnl_memory_destroy(conv1_internal_weights_memory);
    dnnl_primitive_destroy(conv1_reorder_src);
    dnnl_primitive_destroy(conv1_reorder_weights);
    dnnl_primitive_destroy(conv1);

    dnnl_memory_destroy(relu1_dst_memory);
    dnnl_primitive_destroy(relu1);

    dnnl_memory_destroy(pool1_user_dst_memory);
    dnnl_memory_destroy(pool1_internal_dst_memory);
    dnnl_memory_destroy(pool1_ws_memory);
    dnnl_primitive_destroy(pool1_reorder_dst);
    dnnl_primitive_destroy(pool1);

    CHECK(dnnl_primitive_desc_destroy(pool2_pd));
    CHECK(dnnl_primitive_desc_destroy(relu2_pd));
    CHECK(dnnl_primitive_desc_destroy(conv2_pd));

    dnnl_memory_destroy(conv2_user_weights_memory);
    dnnl_memory_destroy(conv2_bias_memory);
    dnnl_memory_destroy(conv2_internal_weights_memory);
    dnnl_primitive_destroy(conv2_reorder_weights);
    dnnl_primitive_destroy(conv2);

    dnnl_memory_destroy(relu2_dst_memory);
    dnnl_primitive_destroy(relu2);

    dnnl_memory_destroy(pool2_user_dst_memory);
    dnnl_memory_destroy(pool2_internal_dst_memory);
    dnnl_memory_destroy(pool2_ws_memory);
    dnnl_primitive_destroy(pool2_reorder_dst);
    dnnl_primitive_destroy(pool2);

    CHECK(dnnl_primitive_desc_destroy(pool3_pd));
    CHECK(dnnl_primitive_desc_destroy(relu3_pd));
    CHECK(dnnl_primitive_desc_destroy(conv3_pd));

    dnnl_memory_destroy(conv3_user_weights_memory);
    dnnl_memory_destroy(conv3_bias_memory);
    dnnl_memory_destroy(conv3_internal_weights_memory);
    dnnl_primitive_destroy(conv3_reorder_weights);
    dnnl_primitive_destroy(conv3);

    dnnl_memory_destroy(relu3_dst_memory);
    dnnl_primitive_destroy(relu3);

    dnnl_memory_destroy(pool3_user_dst_memory);
    dnnl_memory_destroy(pool3_internal_dst_memory);
    dnnl_memory_destroy(pool3_ws_memory);
    dnnl_primitive_destroy(pool3_reorder_dst);
    dnnl_primitive_destroy(pool3);

    // 解除Actor Forward
    for (uint32_t i = 0; i < n_actor_fwd; ++i)
        {free_arg_node(&net_actor_fwd_args[i]);}

    CHECK(dnnl_primitive_desc_destroy(reluA1_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseA1_pd));

    CHECK(dnnl_primitive_desc_destroy(reluA2_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseA2_pd));

    CHECK(dnnl_primitive_desc_destroy(reluA3_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseA3_pd));

    CHECK(dnnl_primitive_desc_destroy(reluA4_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseA4_pd));

    CHECK(dnnl_primitive_desc_destroy(dnseA5_pd));

    // 解除Critic Forward
    for (uint32_t i = 0; i < n_critic_fwd; ++i)
        {free_arg_node(&net_critic_fwd_args[i]);}

    CHECK(dnnl_primitive_desc_destroy(reluC1_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseC1_pd));

    CHECK(dnnl_primitive_desc_destroy(reluC2_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseC2_pd));

    CHECK(dnnl_primitive_desc_destroy(reluC3_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseC3_pd));

    CHECK(dnnl_primitive_desc_destroy(reluC4_pd));
    CHECK(dnnl_primitive_desc_destroy(dnseC4_pd));

    CHECK(dnnl_primitive_desc_destroy(dnseC5_pd));

    dnnl_engine_destroy(Worker -> engine);

/*--------------------关闭并转存神经网络文档---------------------------------*/
#ifdef RL_NET_OUTPUT
    int outputNNFileClose;
    outputNNFileClose = fclose(fp3);
    if(outputNNFileClose == 0){printf("OutputNNFile "
            "has been closed~\r\n");}
    else{printf("error closing outputNNFile!\r\n");}
#endif
}

/**
  * 函数功能：更新Actor网络
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
void RLUpdateActor(unsigned int workerIndex)
{

}

/**
  * 函数功能：更新Critic网络
  * 输入参数：
  * 返 回 值：
  * 相关变量：
  * 说    明：
  */
void RLUpdateCritic(unsigned int workerIndex, unsigned int dataIndex)
{

}

/* user code end 3*/

/* end of file --------------------------------------------------------------*/

