#include <stdlib.h>
#include <time.h>

static int out_size(int in_size, int pad, int dilation, int ksize, int stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}
void nchw_2_cnhw(float *dst, const float * src, int n, int c, int h, int w){
    int in,ic,i;
    int off_src, off_dst;
    int unroll_len = (h*w)/8;
    int unroll_rem = (h*w)%8;
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
void cnhw_2_nchw(float *dst, const float * src, int n, int c, int h, int w){
    int in,ic,i;
    int off_src, off_dst;
    int unroll_len = (h*w)/8;
    int unroll_rem = (h*w)%8;
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


void rand_vector(float * vec, int num){
    static int inited=0;
    int i;
    if(!inited){ inited = 1; srand (time(NULL));}
    for(i=0;i<num;i++) vec[i] = ((float)(rand()%1000))/1000.0f;
}
int valid_vector(float *lhs, float *rhs, int num, float delta=0.01){
    int i;
    int err_cnt=0;
#define ABS(x)  ((x>0)?x:(-1*x))
    for(i=0;i<num;i++){
        float d = lhs[i] - rhs[i];
        d = ABS(d);
        if(d>delta) {printf("diff at %3d, lhs:%f, rhs:%f, diff:%f\n",i,lhs[i],rhs[i],d);err_cnt++;}
    }
    return err_cnt;
}
void dump_vector_nchw(float * t, int n, int c, int h, int w){
    int in,ic,ih,iw;
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
void dump_vector_cnhw(float * t, int n, int c, int h, int w){
    int in,ic,ih,iw;
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

int main(){
    int n = 4;
    int c = 64;
    int h = 55;
    int w = 55;
    int k = 16;
    int r = 7;
    int s = 7;
    int p = 3;
    int q = 3;
    int u = 2;
    int v = 2;
    int dh = 2;
    int dw = 2;
    int oh = out_size(h, p, dh, r, u);
    int ow = out_size(w, q, dw, s, v);
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
        int err_cnt=valid_vector(t_out, t_out_2, n*k*oh*ow);
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
        int err_cnt=valid_vector(t_input_grad, t_input_grad_2, n*c*h*w);
        printf("bwd_d %s\n",err_cnt==0?"ok":"fail");
        
        md_conv_destroy(&md_conv_h);
        md_destroy(&md_h);

        delete [] t_input_grad;
        delete [] t_input_grad_2;
        delete [] t_filter;
        delete [] t_out_grad;
    }
#endif
#if 1
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
        
        md_handle md_h;
        md_conv_handle md_conv_h;

        md_init(&md_h);
        md_conv_init(&md_conv_h,n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        
        //md_conv_bwd_f_nchw(&md_h, &md_conv_h, t_input, t_filter_grad, t_out_grad);
        //naive_conv_bwd_f_nchw(t_input, t_filter_grad_2, t_out_grad, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        md_conv_bwd_f_cnhw(&md_h, &md_conv_h, t_input, t_filter_grad, t_out_grad);
        naive_conv_bwd_f_cnhw(t_input, t_filter_grad_2, t_out_grad, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
        int err_cnt=valid_vector(t_filter_grad, t_filter_grad_2, k*c*r*s, 0.03);
        printf("bwd_f %s\n",err_cnt==0?"ok":"fail");

        md_conv_destroy(&md_conv_h);
        md_destroy(&md_h);

        delete [] t_input;
        delete [] t_filter_grad;
        delete [] t_filter_grad_2;
        delete [] t_out_grad;
    }
#endif
}
