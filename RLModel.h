/* header -------------------------------------------------------------------*/
/**
  * 文件名称：RLModel.h
  * 日    期：2023/03/26
  * 作    者：mrpotato
  * 简    述：oneDNN库强化学习神经网络模型
  */ 
#ifndef _RLModel_h
#define _RLModel_h

/* includes -----------------------------------------------------------------*/
#include "Basic.h"

/* typedef ------------------------------------------------------------------*/
typedef struct {
    int nargs;
    dnnl_exec_arg_t *args;
} args_t;

/* define -------------------------------------------------------------------*/
#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(complain_fmt, ...) \
    do { \
        printf("[%s:%d] Error in the example: " complain_fmt ".\n", __FILE__, \
                __LINE__, __VA_ARGS__); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

#define CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)

/* variables ----------------------------------------------------------------*/

/* function prototypes ------------------------------------------------------*/
extern void RLInitWorker(_RL_worker *Worker);
extern void RLExecuteWorker(FILE *fp1, unsigned char threadIndex,
		unsigned long RLIter, unsigned int *tjtryDataIndex,
		_data_line *DataImputBlock, _motor_line *MotorZeroState,
		_motor_line *MotorSimuBlock, _RL_trajectory *trajectories,
		_RL_worker *Worker);
extern void RLUpdateActor();
extern void RLUpdateCritic();

static dnnl_engine_kind_t validate_engine_kind(dnnl_engine_kind_t akind) {
	// Checking if a GPU exists on the machine
    if (akind == dnnl_gpu) {
        if (!dnnl_engine_get_count(dnnl_gpu)) {
            printf("Application couldn't find GPU, please run with CPU "
                   "instead.\n");
            exit(0);
        }
    }
    return akind;
}

static inline dnnl_engine_kind_t parse_engine_kind(int argc, char **argv) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return validate_engine_kind(dnnl_cpu);
    } else if (argc == 2) {
		// Checking the engine type, i.e. CPU or GPU
		char *engine_kind_str = argv[1];
        if (!strcmp(engine_kind_str, "cpu")) {
            return validate_engine_kind(dnnl_cpu);
        } else if (!strcmp(engine_kind_str, "gpu")) {
            return validate_engine_kind(dnnl_gpu);
        }
    }
	
	// If all above fails, the example should be run properly
	COMPLAIN_EXAMPLE_ERROR_AND_EXIT(
            "inappropriate engine kind.\n"
            "Please run the example like this: %s [cpu|gpu].",
            argv[0]);
}

static inline const char *engine_kind2str_upper(dnnl_engine_kind_t kind) {
    if (kind == dnnl_cpu) return "CPU";
    if (kind == dnnl_gpu) return "GPU";
    return "<Unknown engine>";
}

// Read from memory, write to handle
static inline void read_from_dnnl_memory(void *handle, dnnl_memory_t mem) {
    dnnl_engine_t eng;
    dnnl_engine_kind_t eng_kind;
    const_dnnl_memory_desc_t md;

    if (!handle) COMPLAIN_EXAMPLE_ERROR_AND_EXIT("%s", "handle is NULL.");

    CHECK(dnnl_memory_get_engine(mem, &eng));
    CHECK(dnnl_engine_get_kind(eng, &eng_kind));
    CHECK(dnnl_memory_get_memory_desc(mem, &md));
    size_t bytes = dnnl_memory_desc_get_size(md);

    bool is_cpu_sycl
            = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_cpu);

    if (eng_kind == dnnl_gpu || is_cpu_sycl) {
        void *mapped_ptr = NULL;
        CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
        if (mapped_ptr) memcpy(handle, mapped_ptr, bytes);
        CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
        return;
    }

    if (eng_kind == dnnl_cpu) {
        void *ptr = NULL;
        CHECK(dnnl_memory_get_data_handle(mem, &ptr));
        if (ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)handle)[i] = ((char *)ptr)[i];
            }
        }
        return;
    }

    assert(!"not expected");
}

// Read from handle, write to memory
static inline void write_to_dnnl_memory(void *handle, dnnl_memory_t mem) {
    dnnl_engine_t eng;
    dnnl_engine_kind_t eng_kind;
    const_dnnl_memory_desc_t md;

    if (!handle) COMPLAIN_EXAMPLE_ERROR_AND_EXIT("%s", "handle is NULL.");

    CHECK(dnnl_memory_get_engine(mem, &eng));
    CHECK(dnnl_engine_get_kind(eng, &eng_kind));
    CHECK(dnnl_memory_get_memory_desc(mem, &md));
    size_t bytes = dnnl_memory_desc_get_size(md);

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl
            = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_cpu);
    bool is_gpu_sycl
            = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL && eng_kind == dnnl_gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        void *mapped_ptr = NULL;
        CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
        if (mapped_ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)mapped_ptr)[i] = ((char *)handle)[i];
            }
        }
        CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
        return;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng_kind == dnnl_gpu) {
        void *mapped_ptr = NULL;
        CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
        if (mapped_ptr) memcpy(mapped_ptr, handle, bytes);
        CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
        return;
    }
#endif

    if (eng_kind == dnnl_cpu) {
        void *ptr = NULL;
        CHECK(dnnl_memory_get_data_handle(mem, &ptr));
        if (ptr) {
            for (size_t i = 0; i < bytes; ++i) {
                ((char *)ptr)[i] = ((char *)handle)[i];
            }
        }
        return;
    }

    assert(!"not expected");
}

#endif
/* end of file --------------------------------------------------------------*/
