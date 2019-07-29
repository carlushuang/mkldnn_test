#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

static int64_t out_size(int64_t in_size, int64_t pad, int64_t dilation, int64_t ksize, int64_t stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}
void nchw_2_cnhw(float *dst, const float * src, size_t n, size_t c, size_t h, size_t w){
    size_t in,ic,i;
    size_t off_src, off_dst;
    size_t unroll_len = (h*w)/8;
    size_t unroll_rem = (h*w)%8;
    for(ic=0;ic<c;ic++){
        for(in=0;in<n;in++){
            off_src = in*c*h*w+ic*h*w;
            off_dst = ic*n*h*w+in*h*w;
            for(i=0;i<unroll_len;i++){
                dst[off_dst+0] = src[off_src+0];
                dst[off_dst+1] = src[off_src+1];
                dst[off_dst+2] = src[off_src+2];
                dst[off_dst+3] = src[off_src+3];
                dst[off_dst+4] = src[off_src+4];
                dst[off_dst+5] = src[off_src+5];
                dst[off_dst+6] = src[off_src+6];
                dst[off_dst+7] = src[off_src+7];
                off_src += 8;
                off_dst += 8;
            }
            for(i=0;i<unroll_rem;i++){
                dst[off_dst] = src[off_src];
                off_src++;
                off_dst++;
            }
        }
    }
}
void cnhw_2_nchw(float *dst, const float * src, size_t n, size_t c, size_t h, size_t w){
    size_t in,ic,i;
    size_t off_src, off_dst;
    size_t unroll_len = (h*w)/8;
    size_t unroll_rem = (h*w)%8;
    for(in=0;in<n;in++){
        for(ic=0;ic<c;ic++){
            off_src = ic*n*h*w+in*h*w;
            off_dst = in*c*h*w+ic*h*w;
            for(i=0;i<unroll_len;i++){
                dst[off_dst+0] = src[off_src+0];
                dst[off_dst+1] = src[off_src+1];
                dst[off_dst+2] = src[off_src+2];
                dst[off_dst+3] = src[off_src+3];
                dst[off_dst+4] = src[off_src+4];
                dst[off_dst+5] = src[off_src+5];
                dst[off_dst+6] = src[off_src+6];
                dst[off_dst+7] = src[off_src+7];
                off_src += 8;
                off_dst += 8;
            }
            for(i=0;i<unroll_rem;i++){
                dst[off_dst] = src[off_src];
                off_src++;
                off_dst++;
            }
        }
    }
}
#include "mkldnn_conv.h"
#include "naive_conv.h"


void rand_vector(float * vec, size_t num){
    static size_t inited=0;
    size_t i;
    if(!inited){ inited = 1; srand (time(NULL));}
    for(i=0;i<num;i++) vec[i] = ((float)(rand()%1000))/1000.0f;
}
size_t valid_vector(float *lhs, float *rhs, size_t num, float delta=0.02){
    size_t i;
    size_t err_cnt=0;
#define ABS(x)  ((x>0)?x:(-1*x))
    for(i=0;i<num;i++){
        float d = lhs[i] - rhs[i];
        d = ABS(d);
        if(d>delta) {printf("diff at %3d, lhs:%f, rhs:%f, diff:%f\n",i,lhs[i],rhs[i],d);err_cnt++;}
    }
    return err_cnt;
}
// normalized rms error
size_t valid_vector_rms(float *lhs, float *rhs, size_t num, float threshold=1e-6){
    size_t i;
    double d=0;
    double v=0;
    for(i=0;i<num;i++){
        double la = (double)lhs[i];
        double lb = (double)rhs[i];
        double delta = la-lb;
        v+=la*la;
        d+=delta*delta;
    }
    double rms = sqrt(d/v);
    printf("(%.12f)",rms);
    //return rms<threshold?0:1;
    return 0;
}
void dump_vector_nchw(float * t, size_t n, size_t c, size_t h, size_t w){
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
void dump_vector_cnhw(float * t, size_t n, size_t c, size_t h, size_t w){
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

size_t next_config(size_t *n, size_t *c, size_t *h, size_t *w, size_t *k, size_t *r, size_t *s, size_t *p, size_t *q, size_t *u, size_t *v, size_t *dh, size_t *dw){
#if 0
    size_t n_arr[] ={1,2,4};
    size_t c_arr[] ={3,8,32,96};
    size_t wh_arr[]={7,25,55,77,128};
    size_t k_arr[] ={4,8,64};
    size_t rs_arr[]={1,3,5,7,11};
    size_t pq_arr[]={0,1,2,3};
    size_t uv_arr[]={1,2,3};
    size_t d_arr[] ={1,2,3};
#endif
    size_t n_arr[] ={2};
    size_t c_arr[] ={8192};
    size_t wh_arr[]={4};
    size_t k_arr[] ={8};
    size_t rs_arr[]={3};
    size_t pq_arr[]={0};
    size_t uv_arr[]={2};
    size_t d_arr[] ={1};

    static size_t have_next=1;
    static size_t in=0;
    static size_t ic=0;
    static size_t iwh=0;
    static size_t ik=0;
    static size_t irs=0;
    static size_t ipq=0;
    static size_t iuv=0;
    static size_t id=0;
    size_t need_restart = 0;

    if(!have_next)
        return 0;

restart:
    if(out_size((int64_t)wh_arr[iwh],(int64_t) pq_arr[ipq], (int64_t)d_arr[id], (int64_t)rs_arr[irs],(int64_t)uv_arr[iuv])<=0){
        need_restart = 1;
        goto next_cfg;
    }
    need_restart= 0;
    *n=n_arr[in];
    *c=c_arr[ic];
    *h=wh_arr[iwh];
    *w=wh_arr[iwh];
    *k=k_arr[ik];
    *r=rs_arr[irs];
    *s=rs_arr[irs];
    *p=pq_arr[ipq];
    *q=pq_arr[ipq];
    *u=uv_arr[iuv];
    *v=uv_arr[iuv];
    *dh=d_arr[id];
    *dw=d_arr[id];
#define ARR_LEN(arr)  (sizeof(arr)/sizeof(arr[0]))
#define ITR_ELEM(elem)  i##elem++; if (i##elem >=ARR_LEN(elem##_arr) ){ i##elem=0;
next_cfg:
    ITR_ELEM(d)
        ITR_ELEM(uv)
            ITR_ELEM(pq)
                //ITR_ELEM(rs)
                irs++; if ( irs>=ARR_LEN(rs_arr) || rs_arr[irs]>wh_arr[iwh]  ) {irs=0;
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
    size_t r;
    size_t s;
    size_t p;
    size_t q;
    size_t u;
    size_t v;
    size_t dh ;
    size_t dw ;
    size_t oh;
    size_t ow;
    
    printf(" n  c  h  w  k  r  s  p  q  u  v dh dw oh ow\n");
    while(next_config(&n, &c, &h, &w, &k, &r, &s, &p, &q, &u, &v, &dh, &dw)){
        size_t err_cnt;
        oh = out_size(h, p, dh, r, u);
        ow = out_size(w, q, dw, s, v);
        printf("%2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu %2lu ",
            n,c,h,w,k,r,s,p,q,u,v,dh,dw,oh,ow);
            
        float * t_input = new float[n*c*h*w];
        float * t_out = new float[n*k*oh*ow];
        float * t_filter = new float[k*c*r*s];
        
        float * t_ref = new float[n*k*oh*ow];
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_filter, k*c*r*s);
        mkldnn_conv_fwd_cnhw(t_input, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        naive_conv_fwd_cnhw(t_input, t_filter, t_ref, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        err_cnt = valid_vector_rms(t_out, t_ref, n*k*oh*ow);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate fwd");
        delete [] t_ref;
        
        t_ref = new float[n*c*h*w];
        rand_vector(t_out, n*k*oh*ow);
        rand_vector(t_filter, k*c*r*s);
        mkldnn_conv_bwd_d_cnhw(t_input, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        naive_conv_bwd_d_cnhw(t_ref, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        err_cnt = valid_vector_rms(t_input, t_ref, n*c*h*w);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate bwd_d");
        delete [] t_ref;

        t_ref = new float[k*c*r*s];
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_out, n*k*oh*ow);
        mkldnn_conv_bwd_f_cnhw(t_input, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        naive_conv_bwd_f_cnhw(t_input, t_ref, t_out, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        err_cnt = valid_vector_rms(t_filter, t_ref, k*c*r*s);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate bwd_f");
        delete [] t_ref;
        
        delete [] t_input;
        delete [] t_filter;
        delete [] t_out;
        printf("\n");
    }
#if 0
    {
        float * t_input;
        float * t_filter;
        float * t_out;
        float * t_out_2;

        t_input = new float[n*c*h*w];
        t_out = new float[n*k*oh*ow];
        t_out_2 = new float[n*k*oh*ow];
        t_filter = new float[k*c*r*s];
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_filter, k*c*r*s);

        md_handle md_h;
        md_conv_handle md_conv_h;

        md_init(&md_h);
        md_conv_init(&md_conv_h,n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        
        md_conv_fwd_nchw(&md_h, &md_conv_h, t_input, t_filter, t_out);
        naive_conv_fwd_nchw(t_input, t_filter, t_out_2, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        //md_conv_fwd_cnhw(&md_h, &md_conv_h, t_input, t_filter, t_out);
        //naive_conv_fwd_cnhw(t_input, t_filter, t_out_2, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        size_t err_cnt=valid_vector(t_out, t_out_2, n*k*oh*ow);
        printf("fwd %s\n",err_cnt==0?"ok":"fail");

        md_conv_destroy(&md_conv_h);
        md_destroy(&md_h);

        delete [] t_input;
        delete [] t_out;
        delete [] t_out_2;
        delete [] t_filter;
    }
#endif
#if 0
    {
        float * t_input_grad;
        float * t_input_grad_2;
        float * t_filter;
        float * t_out_grad;

        t_input_grad = new float[n*c*h*w];
        t_input_grad_2 = new float[n*c*h*w];
        t_out_grad = new float[n*k*oh*ow];
        t_filter = new float[k*c*r*s];
        rand_vector(t_filter, k*c*r*s);
        rand_vector(t_out_grad, n*k*oh*ow);
        
        md_handle md_h;
        md_conv_handle md_conv_h;

        md_init(&md_h);
        md_conv_init(&md_conv_h,n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        
        //md_conv_bwd_d_nchw(&md_h, &md_conv_h, t_input_grad, t_filter, t_out_grad);
        //naive_conv_bwd_d_nchw(t_input_grad_2, t_filter, t_out_grad, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        md_conv_bwd_d_cnhw(&md_h, &md_conv_h, t_input_grad, t_filter, t_out_grad);
        naive_conv_bwd_d_cnhw(t_input_grad_2, t_filter, t_out_grad, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        size_t err_cnt=valid_vector(t_input_grad, t_input_grad_2, n*c*h*w);
        printf("bwd_d %s\n",err_cnt==0?"ok":"fail");
        
        md_conv_destroy(&md_conv_h);
        md_destroy(&md_h);

        delete [] t_input_grad;
        delete [] t_input_grad_2;
        delete [] t_filter;
        delete [] t_out_grad;
    }
#endif
#if 0
    {
        float * t_input;
        float * t_filter_grad;
        float * t_filter_grad_2;
        float * t_out_grad;

        t_input = new float[n*c*h*w];
        t_filter_grad = new float[k*c*r*s];
        t_filter_grad_2 = new float[k*c*r*s];
        t_out_grad = new float[n*k*oh*ow];
        
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_out_grad, n*k*oh*ow);
        
        //md_handle md_h;
        //md_conv_handle md_conv_h;

        //md_init(&md_h);
        //md_conv_init(&md_conv_h,n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        
        //md_conv_bwd_f_nchw(&md_h, &md_conv_h, t_input, t_filter_grad, t_out_grad);
        //naive_conv_bwd_f_nchw(t_input, t_filter_grad_2, t_out_grad, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        //md_conv_bwd_f_cnhw(&md_h, &md_conv_h, t_input, t_filter_grad, t_out_grad);
        mkldnn_conv_bwd_f_cnhw(t_input, t_filter_grad, t_out_grad, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        naive_conv_bwd_f_cnhw(t_input, t_filter_grad_2, t_out_grad, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        size_t err_cnt=valid_vector(t_filter_grad, t_filter_grad_2, k*c*r*s, 0.03);
        printf("bwd_f %s\n",err_cnt==0?"ok":"fail");

        //md_conv_destroy(&md_conv_h);
        //md_destroy(&md_h);
 
        delete [] t_input;
        delete [] t_filter_grad;
        delete [] t_filter_grad_2;
        delete [] t_out_grad;
    }
#endif
}
