#ifndef __HIP_NAIVE_CONV_DRIVER_H
#define __HIP_NAIVE_CONV_DRIVER_H
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static inline size_t hip_naive_conv_out_size(size_t in_size, size_t pad,
                                         size_t dilation, size_t ksize,
                                         size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

static struct {
    hipModule_t     module;
    hipFunction_t   kernel_func_naive_conv_fwd_nchw_fp32;
    hipFunction_t   kernel_func_naive_conv_bwd_nchw_fp32;
    hipFunction_t   kernel_func_naive_conv_wrw_nchw_fp32;
} the_hip_handle;

static inline void hip_naive_conv_init(){
    static int inited = 0;
    if(!inited){
        HIP_CALL(hipModuleLoad(&the_hip_handle.module, "hip_naive_conv.hsaco"));
        HIP_CALL(hipModuleGetFunction(&the_hip_handle.kernel_func_naive_conv_fwd_nchw_fp32, the_hip_handle.module, "hip_naive_conv_fwd_nchw_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_hip_handle.kernel_func_naive_conv_bwd_nchw_fp32, the_hip_handle.module, "hip_naive_conv_bwd_nchw_fp32"));
        HIP_CALL(hipModuleGetFunction(&the_hip_handle.kernel_func_naive_conv_wrw_nchw_fp32, the_hip_handle.module, "hip_naive_conv_wrw_nchw_fp32"));
        inited = 1;
    }
}

typedef struct {
    const float * p_in;
    const float * p_wei;
    float * p_out;
    int hi;
    int wi;
    int n;
    int k_per_group;
    int c_per_group;
    int ho;
    int wo;
    int sy;
    int sx;
    int dy;
    int dx;
    int py;
    int px;
    int fy;
    int fx;
    int group;
} __attribute__((packed)) hip_naive_conv_karg_t;

static inline void hip_naive_conv_fwd_nchw_fp32_driver(
    const float *src, const float *filter,
    float *dst, size_t n, size_t w, size_t h,
    size_t c, size_t k, size_t fx, size_t fy,
    size_t px, size_t py, size_t sx,
    size_t sy, size_t dx, size_t dy, size_t group)
{
    size_t ho = hip_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = hip_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
    int block_size = 256;
    int grid_size = n * k_per_group * group;

    // TODO: assume the input pointer are all host memory. now we allocate device memory inside
    float *device_input;
    float *device_weight;
    float *device_output;

    hip_naive_conv_init();

    HIP_CALL(hipMalloc(&device_input,  n * group * c_per_group * h * w * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, group * k_per_group * c_per_group * fy * fx * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, n * group * k_per_group * ho * wo * sizeof(float)));
    HIP_CALL(hipMemcpy(device_input, src,
                       n * group * c_per_group * h * w * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(device_weight, filter,
                       group * k_per_group * c_per_group * fy * fx * sizeof(float), hipMemcpyHostToDevice));

    hip_naive_conv_karg_t karg;
    karg.p_in           = device_input;
    karg.p_wei          = device_weight;
    karg.p_out          = device_output;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);
    
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipHccModuleLaunchKernel(the_hip_handle.kernel_func_naive_conv_fwd_nchw_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

    HIP_CALL(hipMemcpy(dst, device_output,
                       n * group * k_per_group * ho * wo * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);
}

static inline void hip_naive_conv_bwd_nchw_fp32_driver(
    float *src, const float *filter,
    const float *dst, size_t n, size_t w, size_t h,
    size_t c, size_t k, size_t fx, size_t fy,
    size_t px, size_t py, size_t sx,
    size_t sy, size_t dx, size_t dy, size_t group)
{
    size_t ho = hip_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = hip_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
    int block_size = 256;
    int grid_size = n * c_per_group * group;

    // TODO: assume the input pointer are all host memory. now we allocate device memory inside
    float *device_input;
    float *device_weight;
    float *device_output;

    hip_naive_conv_init();

    HIP_CALL(hipMalloc(&device_input,  n * group * c_per_group * h * w * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, group * k_per_group * c_per_group * fy * fx * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, n * group * k_per_group * ho * wo * sizeof(float)));

    HIP_CALL(hipMemcpy(device_output, dst,
                       n * group * k_per_group * ho * wo * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(device_weight, filter,
                       group * k_per_group * c_per_group * fy * fx * sizeof(float), hipMemcpyHostToDevice));

    hip_naive_conv_karg_t karg;
    karg.p_in           = device_input;
    karg.p_wei          = device_weight;
    karg.p_out          = device_output;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);
    
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipHccModuleLaunchKernel(the_hip_handle.kernel_func_naive_conv_bwd_nchw_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

    HIP_CALL(hipMemcpy(src, device_input,
                       n * group * c_per_group * h * w * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);
}

static inline void hip_naive_conv_wrw_nchw_fp32_driver(
    const float *src, float *filter,
    const float *dst, size_t n, size_t w, size_t h,
    size_t c, size_t k, size_t fx, size_t fy,
    size_t px, size_t py, size_t sx,
    size_t sy, size_t dx, size_t dy, size_t group)
{
    size_t ho = hip_naive_conv_out_size(h, py, dy, fy, sy);
    size_t wo = hip_naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
    int block_size = 256;
    int grid_size = group * k_per_group;

    // TODO: assume the input pointer are all host memory. now we allocate device memory inside
    float *device_input;
    float *device_weight;
    float *device_output;

    hip_naive_conv_init();

    HIP_CALL(hipMalloc(&device_input,  n * group * c_per_group * h * w * sizeof(float)));
    HIP_CALL(hipMalloc(&device_weight, group * k_per_group * c_per_group * fy * fx * sizeof(float)));
    HIP_CALL(hipMalloc(&device_output, n * group * k_per_group * ho * wo * sizeof(float)));

    HIP_CALL(hipMemcpy(device_output, dst,
                       n * group * k_per_group * ho * wo * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(device_input, src,
                       n * group * c_per_group * h * w * sizeof(float), hipMemcpyHostToDevice));

    hip_naive_conv_karg_t karg;
    karg.p_in           = device_input;
    karg.p_wei          = device_weight;
    karg.p_out          = device_output;
    karg.hi             = static_cast<int>(h);
    karg.wi             = static_cast<int>(w);
    karg.n              = static_cast<int>(n);
    karg.k_per_group    = static_cast<int>(k_per_group);
    karg.c_per_group    = static_cast<int>(c_per_group);
    karg.ho             = static_cast<int>(ho);
    karg.wo             = static_cast<int>(wo);
    karg.sy             = static_cast<int>(sy);
    karg.sx             = static_cast<int>(sx);
    karg.dy             = static_cast<int>(dy);
    karg.dx             = static_cast<int>(dx);
    karg.py             = static_cast<int>(py);
    karg.px             = static_cast<int>(px);
    karg.fy             = static_cast<int>(fy);
    karg.fx             = static_cast<int>(fx);
    karg.group          = static_cast<int>(group);
    size_t karg_size    = sizeof(karg);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
                        HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipHccModuleLaunchKernel(the_hip_handle.kernel_func_naive_conv_wrw_nchw_fp32, grid_size * block_size, 1, 1,
                                            block_size, 1, 1, 0, 0, NULL,
                                            (void **)&config, NULL, NULL));

    HIP_CALL(hipMemcpy(filter, device_weight,
                       group * k_per_group * c_per_group * fy * fx * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(device_input);
    hipFree(device_weight);
    hipFree(device_output);
}
#endif