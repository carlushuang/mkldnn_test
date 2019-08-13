#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#include "mkldnn_conv.h"
#include "naive_conv.h"
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
    //return rms<threshold?0:1;
    return 0;
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

static size_t next_config(size_t *n, size_t *c, size_t *h, size_t *w, size_t *k, size_t *fy, size_t *fx, size_t *py, size_t *px, size_t *sy, size_t *sx, size_t *dh, size_t *dw){
#if 1
    size_t n_arr[] ={1,2,4};
    size_t c_arr[] ={3,8,32,96};
    size_t wh_arr[]={7,25,55,77,128};
    size_t k_arr[] ={4,8,64};
    size_t fy_arr[]={1,3,5,7,11};
    size_t fx_arr[]={1,3,5,7,11};
    size_t py_arr[]={0,1,2,3};
    size_t px_arr[]={0,1,2,3};
    size_t uv_arr[]={1,2,3};
    size_t d_arr[] ={1,2,3};
#endif
#if 0
    size_t n_arr[] ={2};
    size_t c_arr[] ={128};
    size_t wh_arr[]={17};
    size_t k_arr[] ={128};
    size_t rs_arr[]={7};
    size_t pq_arr[]={3};
    size_t uv_arr[]={1};
    size_t d_arr[] ={1};
#endif
    static size_t have_next=1;
    static size_t in=0;
    static size_t ic=0;
    static size_t iwh=0;
    static size_t ik=0;
    static size_t ify=0;
    static size_t ifx=0;
    static size_t ipy=0;
    static size_t ipx=0;
    static size_t iuv=0;
    static size_t id=0;
    size_t need_restart = 0;

    if(!have_next)
        return 0;

restart:
    if(     fy_arr[ify]>wh_arr[iwh]
        ||  fx_arr[ifx]>wh_arr[iwh]
        ||  (fy_arr[ify]-1>py_arr[ipy])
        ||  (fx_arr[ify]-1>fx_arr[ipy])
        ||  (((int64_t)wh_arr[iwh] + 2*(int64_t)py_arr[ipy]- (int64_t)d_arr[id]*((int64_t)fy_arr[ify]-1) -1)<0)
        ||  (((int64_t)wh_arr[iwh] + 2*(int64_t)px_arr[ipx]- (int64_t)d_arr[id]*((int64_t)fx_arr[ifx]-1) -1)<0)
    ){
        need_restart = 1;
        goto next_cfg;
    }
    need_restart= 0;
    *n=n_arr[in];
    *c=c_arr[ic];
    *h=wh_arr[iwh];
    *w=wh_arr[iwh];
    *k=k_arr[ik];
    *fy=fy_arr[ify];
    *fx=fx_arr[ifx];
    *py=py_arr[ipy];
    *px=px_arr[ipx];
    *sy=uv_arr[iuv];
    *sx=uv_arr[iuv];
    *dh=d_arr[id];
    *dw=d_arr[id];
#define ARR_LEN(arr)  (sizeof(arr)/sizeof(arr[0]))
#define ITR_ELEM(elem)  i##elem++; if (i##elem >=ARR_LEN(elem##_arr) ){ i##elem=0;
next_cfg:
    ITR_ELEM(d)
        ITR_ELEM(uv)
            ITR_ELEM(py)
                ITR_ELEM(px)
                    ITR_ELEM(fy)
                        ITR_ELEM(fx)
                            ITR_ELEM(k)
                                ITR_ELEM(wh)
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
    if(need_restart)
        goto restart;
    return 1;
}

int main(){
    size_t n;
    size_t c;
    size_t h;
    size_t w;
    size_t k;
    size_t fy;
    size_t fx;
    size_t py;
    size_t px;
    size_t sy;
    size_t sx;
    size_t dh ;
    size_t dw ;
    size_t oh;
    size_t ow;
    
    printf(" n  c  h  w  k  fy fx py px sy sx dh dw oh ow| fwd     bwd_d     bwd_f\n");
    while(next_config(&n, &c, &h, &w, &k, &fy, &fx, &py, &px, &sy, &sx, &dh, &dw)){
        size_t err_cnt;
        oh = out_size(h, py, dh, fy, sy);
        ow = out_size(w, px, dw, fx, sx);
        printf("%2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu ",
            n,c,h,w,k,fy,fx,py,px,sy,sx,dh,dw,oh,ow);
            
        float * t_input = new float[n*c*h*w];
        float * t_out = new float[n*k*oh*ow];
        float * t_filter = new float[k*c*fy*fx];
        
        float * t_ref = new float[n*k*oh*ow];
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_filter, k*c*fy*fx);
        mkldnn_conv_fwd_cnhw(t_input, t_filter, t_out, n,c,h,w,k,fy,fx,py,px,sy,sx,dh,dw);
        naive_conv_fwd_cnhw(t_input, t_filter, t_ref, n,c,h,w,k,fy,fx,py,px,sy,sx,dh,dw);
        err_cnt = valid_vector_rms(t_out, t_ref, n*k*oh*ow);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate fwd");
        delete [] t_ref;
        
        t_ref = new float[n*c*h*w];
        rand_vector(t_out, n*k*oh*ow);
        rand_vector(t_filter, k*c*fy*fx);
        mkldnn_conv_bwd_d_cnhw(t_input, t_filter, t_out, n,c,h,w,k,fy,fx,py,px,sy,sx,dh,dw);
        naive_conv_bwd_d_cnhw(t_ref, t_filter, t_out, n,c,h,w,k,fy,fx,py,px,sy,sx,dh,dw);
        err_cnt = valid_vector_rms(t_input, t_ref, n*c*h*w);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate bwd_d");
        delete [] t_ref;

        t_ref = new float[k*c*fy*fx];
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_out, n*k*oh*ow);
        mkldnn_conv_bwd_f_cnhw(t_input, t_filter, t_out, n,c,h,w,k,fy,fx,py,px,sy,sx,dh,dw);
        naive_conv_bwd_f_cnhw(t_input, t_ref, t_out, n,c,h,w,k,fy,fx,py,px,sy,sx,dh,dw);
        err_cnt = valid_vector_rms(t_filter, t_ref, k*c*fy*fx);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate bwd_f");
        delete [] t_ref;
        
        delete [] t_input;
        delete [] t_filter;
        delete [] t_out;
        printf("\n");
    }
}
