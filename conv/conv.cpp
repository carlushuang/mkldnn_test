#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#include "onednn_conv.h"
#include "naive_conv.h"
#ifdef HIP_NAIVE_CONV
#include <half.hpp>
using half_float::half;
#include "hip_naive_conv_driver.h"
#endif
#include <math.h>
static size_t out_size(size_t in_size, size_t pad, size_t dilation, size_t ksize, size_t stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}
static void rand_vector(float * vec, size_t num){
    static size_t inited=0;
    size_t i;
    if(!inited){ inited = 1; srand (time(NULL));}
    for(i=0;i<num;i++) vec[i] = ((float)(rand()%1000))/1000.0f;
}
#ifdef HIP_NAIVE_CONV
static void convert_vector_half2float(float * dst, half * src, size_t num)
{
    for(size_t i=0;i<num;i++) dst[i] = half_float::half_cast<float>(src[i]);
}
static void convert_vector_float2half(half * dst, float * src, size_t num)
{
    for(size_t i=0;i<num;i++) dst[i] = half_float::half_cast<half_float::half>(src[i]);
}
#endif

static size_t valid_vector(float *lhs, float *rhs, size_t num, float delta=0.02){
    size_t i;
    size_t err_cnt=0;
#define ABS(x)  ((x>0)?x:(-1*x))
    for(i=0;i<num;i++){
        float d = lhs[i] - rhs[i];
        d = ABS(d);
        if(d>delta) {printf("diff at %3d, lhs:%f, rhs:%f, diff:%f\n",(int)i,lhs[i],rhs[i],d);err_cnt++;}
    }
    return err_cnt;
}
// normalized rms error
static size_t valid_vector_rms(float *lhs, float *rhs, size_t num, float threshold=1e-6){
    size_t i;
    double d=0;
    double sx=0;
    for(i=0;i<num;i++){
        double la = (double)lhs[i];
        double lb = (double)rhs[i];
        double delta = la-lb;
        sx+=la*la;
        d+=delta*delta;
    }
    double rms = sqrt(d/sx);
    printf("(%.12f)",rms);
    return rms<threshold?0:1;
    //return 0;
}

static void dump_vector_nchw(float * t, size_t n, size_t c, size_t h, size_t w){
    size_t in,ic,ih,iw;
    for(in=0;in<n;in++){
        for(ic=0;ic<c;ic++){
            for(ih=0;ih<h;ih++){
                for(iw=0;iw<w;iw++){
                    printf("%.3f ",t[in*c*h*w+ic*h*w+ih*w+iw]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("--------------------------------\n");
    }
}
static void dump_vector_cnhw(float * t, size_t n, size_t c, size_t h, size_t w){
    size_t in,ic,ih,iw;
    for(ic=0;ic<c;ic++){
        for(in=0;in<n;in++){
            for(ih=0;ih<h;ih++){
                for(iw=0;iw<w;iw++){
                    printf("%.3f ",t[ic*n*h*w+in*h*w+ih*w+iw]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("--------------------------------\n");
    }
}
typedef struct {
    size_t n;
    size_t w;
    size_t h;
    size_t d;
    size_t c;
    size_t k;
    size_t fx;
    size_t fy;
    size_t fz;
    size_t px;
    size_t py;
    size_t pz;
    size_t sx;
    size_t sy;
    size_t sz;
    size_t dx;
    size_t dy;
    size_t dz;
    size_t ng;
}shape_t;

#define ARR_LEN(arr)  (sizeof(arr)/sizeof(arr[0]))
#define ITR_ELEM(elem)  i##elem++; if (i##elem >=ARR_LEN(elem##_arr) ){ i##elem=0;

static size_t next_config(shape_t *shape){
#if 1
    size_t n_arr[] ={1, 8, 16};
    size_t c_arr[] ={3,8,24};
    size_t g_arr[] ={1,2,4};
    // size_t g_arr[] ={2,4};
    size_t w_arr[]={7,26,37};
    size_t h_arr[]={7,25,37};
    size_t k_arr[] ={4,8,16};
    size_t fy_arr[]={1,3,5,7};
    size_t fx_arr[]={1,3,5,7};
    size_t py_arr[]={0,1,2,3};
    size_t px_arr[]={0,1,2,3};
    size_t sy_arr[]={1,2,3};
    size_t sx_arr[]={1,2,3};
    size_t dy_arr[]={1,2,3};
    size_t dx_arr[] ={1,2,3};
#endif
#if 0
    size_t n_arr[] ={2};
    size_t c_arr[] ={128};
    size_t g_arr[] ={2};
    size_t wh_arr[]={17};
    size_t k_arr[] ={128};
    size_t fy_arr[]={1};
    size_t fx_arr[]={1};
    size_t py_arr[]={0};
    size_t px_arr[]={0};
    size_t uv_arr[]={1};
    size_t d_arr[] ={1};
#endif
    static size_t have_next=1;
    static size_t in=0;
    static size_t ic=0;
    static size_t ig=0;
    static size_t ih=0;
    static size_t iw=0;
    static size_t ik=0;
    static size_t ify=0;
    static size_t ifx=0;
    static size_t ipy=0;
    static size_t ipx=0;
    static size_t isy=0;
    static size_t isx=0;
    static size_t idy=0;
    static size_t idx=0;
    size_t need_restart = 0;

    if(!have_next)
        return 0;

restart:
    if(     ((int64_t)fy_arr[ify]>(int64_t)h_arr[ih])
        ||  ((int64_t)fx_arr[ifx]>(int64_t)w_arr[iw])
        ||  (((int64_t)fy_arr[ify]-1)<(int64_t)py_arr[ipy])
        ||  (((int64_t)fx_arr[ifx]-1)<(int64_t)px_arr[ipx])
        ||  (((int64_t)h_arr[ih] + 2*(int64_t)py_arr[ipy]- (int64_t)dy_arr[idy]*((int64_t)fy_arr[ify]-1) -1)<0)
        ||  (((int64_t)w_arr[iw] + 2*(int64_t)px_arr[ipx]- (int64_t)dx_arr[idx]*((int64_t)fx_arr[ifx]-1) -1)<0)
        ||  ((c_arr[ic] % g_arr[ig] != 0) || (k_arr[ik] % g_arr[ig] != 0) )
        ||  ((fy_arr[ify] == 5 && fx_arr[ifx] == 7) || (fy_arr[ify] == 7 && fx_arr[ifx] == 5))
    ){
        need_restart = 1;
        goto next_cfg;
    }
    need_restart= 0;
    shape->n=n_arr[in];
    shape->w=w_arr[iw];
    shape->h=h_arr[ih];
    shape->c=c_arr[ic];
    shape->k=k_arr[ik];
    shape->fx=fx_arr[ifx];
    shape->fy=fy_arr[ify];
    shape->px=px_arr[ipx];
    shape->py=py_arr[ipy];
    shape->sx=sx_arr[isx];
    shape->sy=sy_arr[isy];
    shape->dx=dx_arr[idx];
    shape->dy=dy_arr[idy];
    shape->ng=g_arr[ig];

next_cfg:
    ITR_ELEM(dy)
        ITR_ELEM(dx)
            ITR_ELEM(sy)
                ITR_ELEM(sx)
                    ITR_ELEM(py)
                        ITR_ELEM(px)
                            ITR_ELEM(fy)
                                ITR_ELEM(fx)
                                    ITR_ELEM(k)
                                        ITR_ELEM(h)
                                            ITR_ELEM(w)
                                                ITR_ELEM(g)
                                                    ITR_ELEM(c)
                                                        ITR_ELEM(n)
                                                            have_next=0;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(need_restart)
        goto restart;
    return 1;
}

static size_t next_config_conv3d(shape_t *shape){
#if 1
    size_t n_arr[] ={1,2,4};
    size_t c_arr[] ={3,16};
    size_t g_arr[] ={1,4};
    size_t wh_arr[]={7,20,32};
    size_t d_arr[]={23};
    size_t k_arr[] ={4,8};
    size_t fz_arr[]={1,3,5};
    size_t fy_arr[]={1,3,5};
    size_t fx_arr[]={1,3,5};
    size_t pz_arr[]={0,1,2};
    size_t py_arr[]={0,1,2};
    size_t px_arr[]={0,1,2};
    size_t sz_arr[]={1,2,3};
    size_t sy_arr[]={1,2,3};
    size_t sx_arr[]={1,2,3};
    size_t dz_arr[]={1,2,3};
    size_t dy_arr[]={1,2,3};
    size_t dx_arr[]={1,2,3};

#endif
    static size_t have_next=1;
    static size_t in=0;
    static size_t ic=0;
    static size_t ig=0;
    static size_t iwh=0;
    static size_t id=0;
    static size_t ik=0;
    static size_t ify=0;
    static size_t ifx=0;
    static size_t ifz=0;
    static size_t ipy=0;
    static size_t ipx=0;
    static size_t ipz=0;
    static size_t isy=0;
    static size_t isx=0;
    static size_t isz=0;
    static size_t idy=0;
    static size_t idx=0;
    static size_t idz=0;
    size_t need_restart = 0;

    if(!have_next)
        return 0;

restart:
    if(     ((int64_t)fz_arr[ifz]>(int64_t)d_arr[id])
        ||  ((int64_t)fy_arr[ify]>(int64_t)wh_arr[iwh])
        ||  ((int64_t)fx_arr[ifx]>(int64_t)wh_arr[iwh])
        ||  (((int64_t)fz_arr[ifz]-1)<(int64_t)pz_arr[ipz])
        ||  (((int64_t)fy_arr[ify]-1)<(int64_t)py_arr[ipy])
        ||  (((int64_t)fx_arr[ifx]-1)<(int64_t)px_arr[ipx])
        ||  (((int64_t)d_arr[id] + 2*(int64_t)pz_arr[ipz]- (int64_t)dz_arr[idz]*((int64_t)fz_arr[ifz]-1) -1)<0)
        ||  (((int64_t)wh_arr[iwh] + 2*(int64_t)py_arr[ipy]- (int64_t)dy_arr[idy]*((int64_t)fy_arr[ify]-1) -1)<0)
        ||  (((int64_t)wh_arr[iwh] + 2*(int64_t)px_arr[ipx]- (int64_t)dx_arr[idx]*((int64_t)fx_arr[ifx]-1) -1)<0)
        ||  ((c_arr[ic] % g_arr[ig] != 0) || (k_arr[ik] % g_arr[ig] != 0) )
    ){
        need_restart = 1;
        goto next_cfg;
    }
    need_restart= 0;
    shape->n=n_arr[in];
    shape->w=wh_arr[iwh];
    shape->h=wh_arr[iwh];
    shape->d=d_arr[id];
    shape->c=c_arr[ic];
    shape->k=k_arr[ik];
    shape->fx=fx_arr[ifx];
    shape->fy=fy_arr[ify];
    shape->fz=fy_arr[ifz];
    shape->px=px_arr[ipx];
    shape->py=py_arr[ipy];
    shape->pz=pz_arr[ipz];
    shape->sx=sx_arr[isx];
    shape->sy=sy_arr[isy];
    shape->sz=sz_arr[isz];
    shape->dx=dx_arr[idx];
    shape->dy=dy_arr[idy];
    shape->dz=dz_arr[idz];
    shape->ng=g_arr[ig];

next_cfg:
    ITR_ELEM(dz)
        ITR_ELEM(dy)
            ITR_ELEM(dx)
                ITR_ELEM(sz)
                    ITR_ELEM(sy)
                        ITR_ELEM(sx)
                            ITR_ELEM(pz)
                                ITR_ELEM(py)
                                    ITR_ELEM(px)
                                        ITR_ELEM(fz)
                                            ITR_ELEM(fy)
                                                ITR_ELEM(fx)
                                                    ITR_ELEM(k)
                                                        ITR_ELEM(wh)
                                                            ITR_ELEM(d)
                                                                ITR_ELEM(g)
                                                                    ITR_ELEM(c)
                                                                        ITR_ELEM(n)
                                                                            have_next=0;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(need_restart)
        goto restart;
    return 1;
}

#define DIRECTION_FWD   (0 << 4)
#define DIRECTION_BWD   (1 << 4)
#define DIRECTION_WRW   (2 << 4)

#define DATA_TYPE_FP32 0
#define DATA_TYPE_FP16 1

static inline float get_tolerence(int direction, int data_type)
{
    if(data_type == DATA_TYPE_FP32){
        return 1e-4;
    }else if(data_type == DATA_TYPE_FP16){
        return 8.2e-3;
    }
    else
        assert(0);
    
}

int main(){
    auto test_conv_3d = [&](int data_type){
        shape_t shape;
        printf(" n  w  h  d  c  k  fx fy fz px py pz sx sy sz dx dy dz ow oh od ng| fwd     bwd       wrw\n");
        while(next_config_conv3d(&shape)){
            size_t err_cnt,od,oh,ow;
            od = out_size(shape.d, shape.pz, shape.dz, shape.fz, shape.sz);
            oh = out_size(shape.h, shape.py, shape.dy, shape.fy, shape.sy);
            ow = out_size(shape.w, shape.px, shape.dx, shape.fx, shape.sx);
            printf("%2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu",
                shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,
                shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,ow,oh,od,shape.ng);
            fflush(stdout);
            float * t_input = new float[shape.n*shape.c*shape.d*shape.h*shape.w];
            float * t_out = new float[shape.n*shape.k*od*oh*ow];
            float * t_filter = new float[shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng];

            float * t_ref = new float[shape.n*shape.k*od*oh*ow];
            rand_vector(t_input, shape.n*shape.c*shape.d*shape.h*shape.w);
            rand_vector(t_filter, shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng);
            onednn_conv_fwd_ncdhw(t_input, t_filter, t_out, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#ifdef HIP_NAIVE_CONV
            if(data_type == DATA_TYPE_FP16){
                half * t_input_fp16 = new half[shape.n*shape.c*shape.d*shape.h*shape.w];
                half * t_filter_fp16 = new half[shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng];
                half * t_ref_fp16 = new half[shape.n*shape.k*od*oh*ow];
                convert_vector_float2half(t_input_fp16, t_input, shape.n*shape.c*shape.d*shape.h*shape.w);
                convert_vector_float2half(t_filter_fp16, t_filter, shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng);
                hip_naive_conv_fwd_ncdhw_fp16_driver(t_input_fp16, t_filter_fp16, t_ref_fp16, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
                convert_vector_half2float(t_ref, t_ref_fp16, shape.n*shape.k*od*oh*ow);
                delete [] t_input_fp16;
                delete [] t_filter_fp16;
                delete [] t_ref_fp16;
            }else
                hip_naive_conv_fwd_ncdhw_fp32_driver(t_input, t_filter, t_ref, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#else
            naive_conv_fwd_ncdhw(t_input, t_filter, t_ref, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#endif
            err_cnt = valid_vector_rms(t_out, t_ref, shape.n*shape.k*od*oh*ow, get_tolerence(DIRECTION_FWD, data_type));
            printf("%s ",(err_cnt==0)?"y":"n");
            fflush(stdout);
            assert(err_cnt==0 && "fail to validate fwd");
            delete [] t_ref;

            t_ref = new float[shape.n*shape.c*shape.d*shape.h*shape.w];
            rand_vector(t_out, shape.n*shape.k*od*oh*ow);
            rand_vector(t_filter, shape.k*shape.c*shape.fz*shape.fy*shape.fx/shape.ng);
            onednn_conv_bwd_ncdhw(t_input, t_filter, t_out, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#ifdef HIP_NAIVE_CONV
            if(data_type == DATA_TYPE_FP16){
                half * t_ref_fp16 = new half[shape.n*shape.c*shape.d*shape.h*shape.w];
                half * t_filter_fp16 = new half[shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng];
                half * t_out_fp16 = new half[shape.n*shape.k*od*oh*ow];
                convert_vector_float2half(t_filter_fp16, t_filter, shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng);
                convert_vector_float2half(t_out_fp16, t_out, shape.n*shape.k*od*oh*ow);
                hip_naive_conv_bwd_ncdhw_fp16_driver(t_ref_fp16, t_filter_fp16, t_out_fp16, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
                convert_vector_half2float(t_ref, t_ref_fp16, shape.n*shape.c*shape.d*shape.h*shape.w);
                delete [] t_ref_fp16;
                delete [] t_filter_fp16;
                delete [] t_out_fp16;
            }else
                hip_naive_conv_bwd_ncdhw_fp32_driver(t_ref, t_filter, t_out, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#else
            naive_conv_bwd_ncdhw(t_ref, t_filter, t_out, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#endif
            err_cnt = valid_vector_rms(t_input, t_ref, shape.n*shape.c*shape.d*shape.h*shape.w, get_tolerence(DIRECTION_BWD, data_type));
            printf("%s ",(err_cnt==0)?"y":"n");
            fflush(stdout);
            assert(err_cnt==0 && "fail to validate bwd");
            delete [] t_ref;

            t_ref = new float[shape.k*shape.c*shape.fz*shape.fy*shape.fx/shape.ng];
            rand_vector(t_input, shape.n*shape.c*shape.d*shape.h*shape.w);
            rand_vector(t_out, shape.n*shape.k*od*oh*ow);
            onednn_conv_wrw_ncdhw(t_input, t_filter, t_out, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#ifdef HIP_NAIVE_CONV
            if(data_type == DATA_TYPE_FP16){
                half * t_input_fp16 = new half[shape.n*shape.c*shape.d*shape.h*shape.w];
                half * t_ref_fp16 = new half[shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng];
                half * t_out_fp16 = new half[shape.n*shape.k*od*oh*ow];
                convert_vector_float2half(t_input_fp16, t_input, shape.n*shape.c*shape.d*shape.h*shape.w);
                convert_vector_float2half(t_out_fp16, t_out, shape.n*shape.k*od*oh*ow);
                hip_naive_conv_wrw_ncdhw_fp16_driver(t_input_fp16, t_ref_fp16, t_out_fp16, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
                convert_vector_half2float(t_ref, t_ref_fp16, shape.k*shape.c*shape.fz*shape.fy*shape.fx / shape.ng);
                delete [] t_input_fp16;
                delete [] t_ref_fp16;
                delete [] t_out_fp16;
            }else
                hip_naive_conv_wrw_ncdhw_fp32_driver(t_input, t_ref, t_out, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#else
            naive_conv_wrw_ncdhw(t_input, t_ref, t_out, shape.n,shape.w,shape.h,shape.d,shape.c,shape.k,shape.fx,shape.fy,shape.fz,shape.px,shape.py,shape.pz,shape.sx,shape.sy,shape.sz,shape.dx,shape.dy,shape.dz,shape.ng);
#endif
            err_cnt = valid_vector_rms(t_filter, t_ref, shape.k*shape.c*shape.fz*shape.fy*shape.fx/shape.ng, get_tolerence(DIRECTION_WRW, data_type));
            printf("%s ",(err_cnt==0)?"y":"n");
            fflush(stdout);
            assert(err_cnt==0 && "fail to validate wrw");
            delete [] t_ref;

            delete [] t_input;
            delete [] t_filter;
            delete [] t_out;
            printf("\n");
            fflush(stdout);
        }
    };

    auto test_conv_2d = [&](int data_type){
        shape_t shape;
        printf(" n  w  h  c  k  fx fy px py sx sy dx dy ow oh ng| fwd     bwd       wrw\n");
        while(next_config(&shape)){
            size_t err_cnt,oh,ow;
            oh = out_size(shape.h, shape.py, shape.dy, shape.fy, shape.sy);
            ow = out_size(shape.w, shape.px, shape.dx, shape.fx, shape.sx);
            printf("%2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu",
                shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,ow,oh,shape.ng);
            fflush(stdout);
            float * t_input = new float[shape.n*shape.c*shape.h*shape.w];
            float * t_out = new float[shape.n*shape.k*oh*ow];
            float * t_filter = new float[shape.k*shape.c*shape.fy*shape.fx / shape.ng];

            float * t_ref = new float[shape.n*shape.k*oh*ow];
            rand_vector(t_input, shape.n*shape.c*shape.h*shape.w);
            rand_vector(t_filter, shape.k*shape.c*shape.fy*shape.fx / shape.ng);
            onednn_conv_fwd_nchw(t_input, t_filter, t_out, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#ifdef HIP_NAIVE_CONV
            if(data_type == DATA_TYPE_FP16){
                half * t_input_fp16 = new half[shape.n*shape.c*shape.h*shape.w];
                half * t_filter_fp16 = new half[shape.k*shape.c*shape.fy*shape.fx / shape.ng];
                half * t_ref_fp16 = new half[shape.n*shape.k*oh*ow];
                convert_vector_float2half(t_input_fp16, t_input, shape.n*shape.c*shape.h*shape.w);
                convert_vector_float2half(t_filter_fp16, t_filter, shape.k*shape.c*shape.fy*shape.fx / shape.ng);
                hip_naive_conv_fwd_nchw_fp16_driver(t_input_fp16, t_filter_fp16, t_ref_fp16, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
                convert_vector_half2float(t_ref, t_ref_fp16, shape.n*shape.k*oh*ow);
                delete [] t_input_fp16;
                delete [] t_filter_fp16;
                delete [] t_ref_fp16;
            }else
                hip_naive_conv_fwd_nchw_fp32_driver(t_input, t_filter, t_ref, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#else
            naive_conv_fwd_nchw(t_input, t_filter, t_ref, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#endif
            err_cnt = valid_vector_rms(t_out, t_ref, shape.n*shape.k*oh*ow, get_tolerence(DIRECTION_FWD, data_type));
            printf("%s ",(err_cnt==0)?"y":"n");
            fflush(stdout);
            assert(err_cnt==0 && "fail to validate fwd");
            delete [] t_ref;

            t_ref = new float[shape.n*shape.c*shape.h*shape.w];
            rand_vector(t_out, shape.n*shape.k*oh*ow);
            rand_vector(t_filter, shape.k*shape.c*shape.fy*shape.fx/shape.ng);
            onednn_conv_bwd_nchw(t_input, t_filter, t_out, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#ifdef HIP_NAIVE_CONV
            if(data_type == DATA_TYPE_FP16){
                half * t_ref_fp16 = new half[shape.n*shape.c*shape.h*shape.w];
                half * t_filter_fp16 = new half[shape.k*shape.c*shape.fy*shape.fx / shape.ng];
                half * t_out_fp16 = new half[shape.n*shape.k*oh*ow];
                convert_vector_float2half(t_out_fp16, t_out, shape.n*shape.k*oh*ow);
                convert_vector_float2half(t_filter_fp16, t_filter, shape.k*shape.c*shape.fy*shape.fx / shape.ng);
                hip_naive_conv_bwd_nchw_fp16_driver(t_ref_fp16, t_filter_fp16, t_out_fp16, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
                convert_vector_half2float(t_ref, t_ref_fp16, shape.n*shape.c*shape.h*shape.w);
                delete [] t_ref_fp16;
                delete [] t_filter_fp16;
                delete [] t_out_fp16;
            }else
                hip_naive_conv_bwd_nchw_fp32_driver(t_ref, t_filter, t_out, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#else
            naive_conv_bwd_nchw(t_ref, t_filter, t_out, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#endif
            err_cnt = valid_vector_rms(t_input, t_ref, shape.n*shape.c*shape.h*shape.w, get_tolerence(DIRECTION_BWD, data_type));
            printf("%s ",(err_cnt==0)?"y":"n");
            fflush(stdout);
            assert(err_cnt==0 && "fail to validate bwd");
            delete [] t_ref;

            t_ref = new float[shape.k*shape.c*shape.fy*shape.fx/shape.ng];
            rand_vector(t_input, shape.n*shape.c*shape.h*shape.w);
            rand_vector(t_out, shape.n*shape.k*oh*ow);
            onednn_conv_wrw_nchw(t_input, t_filter, t_out, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#ifdef HIP_NAIVE_CONV
            if(data_type == DATA_TYPE_FP16){
                half * t_input_fp16 = new half[shape.n*shape.c*shape.h*shape.w];
                half * t_ref_fp16 = new half[shape.k*shape.c*shape.fy*shape.fx / shape.ng];
                half * t_out_fp16 = new half[shape.n*shape.k*oh*ow];
                convert_vector_float2half(t_input_fp16, t_input, shape.n*shape.c*shape.h*shape.w);
                convert_vector_float2half(t_out_fp16, t_out, shape.n*shape.k*oh*ow);
                hip_naive_conv_wrw_nchw_fp16_driver(t_input_fp16, t_ref_fp16, t_out_fp16, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
                convert_vector_half2float(t_ref, t_ref_fp16, shape.k*shape.c*shape.fy*shape.fx / shape.ng);
                delete [] t_input_fp16;
                delete [] t_ref_fp16;
                delete [] t_out_fp16;
            }else
                hip_naive_conv_wrw_nchw_fp32_driver(t_input, t_ref, t_out, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#else
            naive_conv_wrw_nchw(t_input, t_ref, t_out, shape.n,shape.w,shape.h,shape.c,shape.k,shape.fx,shape.fy,shape.px,shape.py,shape.sx,shape.sy,shape.dx,shape.dy,shape.ng);
#endif
            err_cnt = valid_vector_rms(t_filter, t_ref, shape.k*shape.c*shape.fy*shape.fx/shape.ng, get_tolerence(DIRECTION_WRW, data_type));
            printf("%s ",(err_cnt==0)?"y":"n");
            fflush(stdout);
            assert(err_cnt==0 && "fail to validate wrw");
            delete [] t_ref;

            delete [] t_input;
            delete [] t_filter;
            delete [] t_out;
            printf("\n");
            fflush(stdout);
        }
    };

    test_conv_2d(DATA_TYPE_FP32);
    test_conv_3d(DATA_TYPE_FP32);
    test_conv_2d(DATA_TYPE_FP16);
    test_conv_3d(DATA_TYPE_FP16);
}
