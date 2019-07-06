#ifndef __NAIVE_CONV_H
#define __NAIVE_CONV_H
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
void naive_conv_bwd_d_nchw(float *src_grad, const float *filter, const float *dst_grad,
    int n, int c, int h, int w, int k, int r, int s, int p, int q, int u, int v, int dh, int dw)
{
    int in,ik,ih,iw,ic,is,ir;
    int cur_oh, cur_ow, o_idx, i_idx, f_idx;
    int oh = out_size(h, p, dh, r, u);
    int ow = out_size(w, q, dw, s, v);
    for(in=0;in<n;in++){
        for(ic=0;ic<c;ic++){
            for(ih=0;ih<h;ih++){
                for(iw=0;iw<w;iw++){
                    float value = .0f;
                    i_idx = in*c*h*w+ic*h*w+ih*w+iw;
                    for(ik=0;ik<k;ik++){
                        for(ir=0;ir<r;ir++){
                            cur_oh = ih+p-dh*ir; // cur_h = u*ioh-p+dh*ir;
                            if(cur_oh<0 || cur_oh % u) continue;
                            cur_oh/=u;
                            if(cur_oh>=oh) continue;
                            for(is=0;is<s;is++){
                                cur_ow = iw+q-dw*is;  // cur_w = v*iow-q+dw*is;
                                if(cur_ow<0 || cur_ow %v) continue;
                                cur_ow /= v;
                                if(cur_ow>=ow) continue;
                                
                                o_idx = in*k*oh*ow+ik*oh*ow+cur_oh*ow+cur_ow;
                                f_idx = ik*c*r*s+ic*r*s+ir*s+is;
                                
                                value += dst_grad[o_idx]*filter[f_idx];
                            }
                        }
                    }
                    src_grad[i_idx] = value;
                }
            }
        }
    }
}
void naive_conv_bwd_d_cnhw(float *src_grad, const float *filter, const float *dst_grad,
    int n, int c, int h, int w, int k, int r, int s, int p, int q, int u, int v, int dh, int dw)
{
    int in,ik,ih,iw,ic,is,ir;
    int cur_oh, cur_ow, o_idx, i_idx, f_idx;
    int oh = out_size(h, p, dh, r, u);
    int ow = out_size(w, q, dw, s, v);
    for(ic=0;ic<c;ic++){
        for(in=0;in<n;in++){
            for(ih=0;ih<h;ih++){
                for(iw=0;iw<w;iw++){
                    float value = .0f;
                    i_idx = ic*n*h*w+in*h*w+ih*w+iw;
                    for(ik=0;ik<k;ik++){
                        for(ir=0;ir<r;ir++){
                            cur_oh = ih+p-dh*ir; // cur_h = u*ioh-p+dh*ir;
                            if(cur_oh<0 || cur_oh % u) continue;
                            cur_oh/=u;
                            if(cur_oh>=oh) continue;
                            for(is=0;is<s;is++){
                                cur_ow = iw+q-dw*is;  // cur_w = v*iow-q+dw*is;
                                if(cur_ow<0 || cur_ow %v) continue;
                                cur_ow /= v;
                                if(cur_ow>=ow) continue;
                                
                                o_idx = ik*n*oh*ow+in*oh*ow+cur_oh*ow+cur_ow;
                                f_idx = ik*c*r*s+ic*r*s+ir*s+is;
                                
                                value += dst_grad[o_idx]*filter[f_idx];
                            }
                        }
                    }
                    src_grad[i_idx] = value;
                }
            }
        }
    }
}
#endif