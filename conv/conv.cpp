#include <mkldnn.hpp>
#include <stdlib.h>
#include <time.h>

typedef struct {
    mkldnn::engine * eng;
}md_handle;
typedef struct{
    int n; // batch
    int c;
    int h;
    int w;
    int k;
    int r; // filter_h
    int s; // filter_w
    int p; // pad h
    int q; // pad w
    int u; // stride h
    int v; // stride w
    int dh; // dilation h
    int dw; // dilation w

    int oh;
    int ow;

    mkldnn::memory::desc *src_desc;
    mkldnn::memory::desc *filter_desc;
    mkldnn::memory::desc *dst_desc;
    mkldnn::convolution_forward::desc *fwd_desc;
    mkldnn::convolution_forward *fwd;
}md_conv_handle;

void md_init(md_handle * mh){
    mh->eng = new mkldnn::engine(mkldnn::engine::kind::cpu,0);
}
void md_destroy(md_handle * mh){
    delete mh->eng;
}
static int out_size(int in_size, int pad, int dilation, int ksize, int stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}
void md_conv_init(md_conv_handle *conv,int n, int c, int h, int w, int k, int r, int s, int p, int q, int u, int v, int dh, int dw){
    conv->n = n;
    conv->c = c;
    conv->h = h;
    conv->w = w;
    conv->k = k;
    conv->r = r;
    conv->s = s;
    conv->p = p;
    conv->q = q;
    conv->u = u;
    conv->v = v;
    conv->dh = dh;
    conv->dw = dw;
    conv->oh = out_size(h, p, dh, r, u);
    conv->ow = out_size(w, q, dw, s, v);
    
    conv->src_desc = new mkldnn::memory::desc({n,c,h,w}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);
    conv->filter_desc = new mkldnn::memory::desc({k,c,r,s}, mkldnn::memory::data_type::f32, mkldnn::memory::format::oihw);
    conv->dst_desc = new mkldnn::memory::desc({n,k,conv->oh,conv->ow}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);

    conv->fwd_desc = new mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward_inference,
                        mkldnn::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {u,v},{dh-1,dw-1},{p,q},{p,q},mkldnn::padding_kind::zero);
    // note in mkl-dnn, dilation is 0 when unit dilation. 
}
void md_conv_destroy(md_conv_handle * conv){
    delete conv->src_desc;
    delete conv->filter_desc;
    delete conv->dst_desc;
    delete conv->fwd_desc;
}
void md_conv(md_handle *mh, md_conv_handle *conv, float * src, float * filter, float * dst){
    auto src_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src  );
    auto filter_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter  );
    auto dst_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst);
    mkldnn::convolution_forward conv_fwd({*conv->fwd_desc,*mh->eng},src_memory,filter_memory,dst_memory  );
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_fwd}).wait();
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
void md_conv_cnhw(md_handle *mh, md_conv_handle *conv, float * src, float * filter, float * dst){
    int n = conv->src_desc->data.dims[0];
    int c = conv->src_desc->data.dims[1];
    int h = conv->src_desc->data.dims[2];
    int w = conv->src_desc->data.dims[3];
    int k = conv->dst_desc->data.dims[1];
    int oh = conv->dst_desc->data.dims[2];
    int ow = conv->dst_desc->data.dims[3];
    float * src_nchw = new float[n*c*h*w];
    float * dst_nchw = new float[n*k*oh*ow];
    cnhw_2_nchw(src_nchw, src, n, c, h, w);
    auto src_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src_nchw  );
    auto filter_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter  );
    auto dst_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst_nchw);
    mkldnn::convolution_forward conv_fwd({*conv->fwd_desc,*mh->eng},src_memory,filter_memory,dst_memory  );
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_fwd}).wait();
    nchw_2_cnhw(dst, dst_nchw, n, k, oh, ow);
    delete [] src_nchw;
    delete [] dst_nchw;
}

void naive_conv_fwd_nchw(const float *src, const float *filter, float *dst,
    int n, int c, int h, int w, int k, int r, int s, int p, int q, int u, int v, int dh, int dw)
{
    int in,ik,ioh,iow,ic,is,ir;
    int cur_h, cur_w, o_idx, i_idx, f_idx;
    int oh = out_size(h, p, dh, r, u);
    int ow = out_size(w, q, dw, s, v);
    for(in=0;in<n;in++){
        for(ik=0;ik<k;ik++){
            for(ioh=0;ioh<oh;ioh++){
                for(iow=0;iow<ow;iow++){
                    // sliding window for this filter
                    float value = .0f;
                    o_idx = in*k*oh*ow+ik*oh*ow+ioh*ow+iow;
                    for(ic=0;ic<c;ic++){
                        for(ir=0;ir<r;ir++){
                            cur_h = u*ioh-p+dh*ir;
                            if(cur_h<0 || cur_h>=h) continue;
                            for(is=0;is<s;is++){
                                cur_w = v*iow-q+dw*is;
                                if(cur_w<0 || cur_w>=w) continue;
                                i_idx = in*c*h*w+ic*h*w+cur_h*w+cur_w;
                                f_idx = ik*c*r*s+ic*r*s+ir*s+is;
                                value += src[i_idx]*filter[f_idx];
                            }
                        }
                    }
                    dst[o_idx] = value;
                }
            }
        }
    }
}
void naive_conv_fwd_cnhw(const float *src, const float *filter, float *dst,
    int n, int c, int h, int w, int k, int r, int s, int p, int q, int u, int v, int dh, int dw)
{
    int in,ik,ioh,iow,ic,is,ir;
    int cur_h, cur_w, o_idx, i_idx, f_idx;
    int oh = out_size(h, p, dh, r, u);
    int ow = out_size(w, q, dw, s, v);
    for(ik=0;ik<k;ik++){
        for(in=0;in<n;in++){
            for(ioh=0;ioh<oh;ioh++){
                for(iow=0;iow<ow;iow++){
                    // sliding window for this filter
                    float value = .0f;
                    o_idx = ik*n*oh*ow+in*oh*ow+ioh*ow+iow;
                    for(ic=0;ic<c;ic++){
                        for(ir=0;ir<r;ir++){
                            cur_h = u*ioh-p+dh*ir;
                            if(cur_h<0 || cur_h>=h) continue;
                            for(is=0;is<s;is++){
                                cur_w = v*iow-q+dw*is;
                                if(cur_w<0 || cur_w>=w) continue;
                                i_idx = ic*n*h*w+in*h*w+cur_h*w+cur_w;
                                f_idx = ik*c*r*s+ic*r*s+ir*s+is;
                                value += src[i_idx]*filter[f_idx];
                            }
                        }
                    }
                    dst[o_idx] = value;
                    //printf("o)idx:%d, value:%f\n",o_idx, value);
                }
            }
        }
    }
}

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
    int r = 4;
    int s = 4;
    int p = 3;
    int q = 3;
    int u = 2;
    int v = 2;
    int dh = 2;
    int dw = 2;
    int oh = out_size(h, p, dh, r, u);
    int ow = out_size(w, q, dw, s, v);
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

#if 1
    md_handle md_h;
    md_conv_handle md_conv_h;

    md_init(&md_h);
    md_conv_init(&md_conv_h,n,c,h,w,k,r,s,p,q,u,v,dh,dw);
    
    //md_conv(&md_h, &md_conv_h, t_input, t_filter, t_out);
    //naive_conv_fwd_nchw(t_input, t_filter, t_out_2, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
    md_conv_cnhw(&md_h, &md_conv_h, t_input, t_filter, t_out);
    naive_conv_fwd_cnhw(t_input, t_filter, t_out_2, n,c,h,w,k,r,s,p,q,u,v,dh,dw);
    valid_vector(t_out, t_out_2, n*k*oh*ow);


    md_conv_destroy(&md_conv_h);
    md_destroy(&md_h);
#endif
    delete [] t_input;
    delete [] t_out;
    delete [] t_out_2;
    delete [] t_filter;
}
