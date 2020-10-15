#ifndef __ONEDNN_CONV_H
#define __ONEDNN_CONV_H

#include <dnnl.hpp>
static inline size_t onednn_conv_out_size(size_t in_size, size_t pad, size_t dilation, size_t ksize, size_t stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}

static inline void onednn_conv_nchw_2_cnhw(float *dst, const float * src, size_t n, size_t c, size_t h, size_t w){
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
static inline void onednn_conv_cnhw_2_nchw(float *dst, const float * src, size_t n, size_t c, size_t h, size_t w){
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
typedef struct {
    dnnl::engine * eng;
}onednn_handle_t;

typedef struct{
    size_t n;
    size_t w;
    size_t h;
    size_t c;
    size_t k;
    size_t fx;
    size_t fy;
    size_t px;
    size_t py;
    size_t sx;
    size_t sy;
    size_t dx;
    size_t dy;
    size_t ow;
    size_t oh;
    size_t group;

    dnnl::memory::desc *src_desc;
    dnnl::memory::desc *filter_desc;
    dnnl::memory::desc *dst_desc;

    dnnl::convolution_forward::desc           *fwd_desc;
    dnnl::convolution_backward_data::desc     *bwd_desc;
    dnnl::convolution_backward_weights::desc  *wrw_desc;
}onednn_conv_handle_t;

#define TO_DIM_T(t) static_cast<dnnl::memory::dim>(t)

static inline void onednn_init(onednn_handle_t * handle){
    handle->eng = new dnnl::engine(dnnl::engine::kind::cpu,0);
}
static inline void onednn_destroy(onednn_handle_t * handle){
    delete handle->eng;
}

inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        std::copy(src, src + bytes, (uint8_t *)handle);
    }
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

static inline void onednn_conv_init(onednn_conv_handle_t *conv,size_t n, size_t w, size_t h, size_t c, size_t k, size_t fx, size_t fy, size_t px, size_t py, size_t sx, size_t sy, size_t dx, size_t dy, size_t group){
    conv->n = n;
    conv->w = w;
    conv->h = h;
    conv->c = c;
    conv->k = k;
    conv->fx = fx;
    conv->fy = fy;
    conv->px = px;
    conv->py = py;
    conv->sx = sx;
    conv->sy = sy;
    conv->dx = dx;
    conv->dy = dy;
    conv->ow = onednn_conv_out_size(w, px, dx, fx, sx);
    conv->oh = onednn_conv_out_size(h, py, dy, fy, sy);
    conv->group = group;
    assert((group >= 1) && (c % group == 0) && (k % group == 0));

    conv->src_desc = new dnnl::memory::desc({TO_DIM_T(n),TO_DIM_T(c),TO_DIM_T(h),TO_DIM_T(w)}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
    if(group == 1)
        conv->filter_desc = new dnnl::memory::desc({TO_DIM_T(k),TO_DIM_T(c),TO_DIM_T(fy),TO_DIM_T(fx)}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw);
    else
        conv->filter_desc = new dnnl::memory::desc({TO_DIM_T(group),TO_DIM_T(k/group),TO_DIM_T(c/group),TO_DIM_T(fy),TO_DIM_T(fx)}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::goihw);
    conv->dst_desc = new dnnl::memory::desc({TO_DIM_T(n),TO_DIM_T(k),TO_DIM_T(conv->oh),TO_DIM_T(conv->ow)}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);

    conv->fwd_desc = new dnnl::convolution_forward::desc(dnnl::prop_kind::forward,
                        dnnl::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {TO_DIM_T(sy),TO_DIM_T(sx)},{TO_DIM_T(dy-1),TO_DIM_T(dx-1)},{TO_DIM_T(py),TO_DIM_T(px)},{TO_DIM_T(py),TO_DIM_T(px)});

    conv->bwd_desc = new dnnl::convolution_backward_data::desc(dnnl::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {TO_DIM_T(sy),TO_DIM_T(sx)},{TO_DIM_T(dy-1),TO_DIM_T(dx-1)},{TO_DIM_T(py),TO_DIM_T(px)},{TO_DIM_T(py),TO_DIM_T(px)});
    
    conv->wrw_desc = new dnnl::convolution_backward_weights::desc(dnnl::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {TO_DIM_T(sy),TO_DIM_T(sx)},{TO_DIM_T(dy-1),TO_DIM_T(dx-1)},{TO_DIM_T(py),TO_DIM_T(px)},{TO_DIM_T(py),TO_DIM_T(px)});

}
static inline void onednn_conv_destroy(onednn_conv_handle_t * conv){
    delete conv->src_desc;
    delete conv->filter_desc;
    delete conv->dst_desc;
    delete conv->fwd_desc;
    delete conv->bwd_desc;
    delete conv->wrw_desc;
}
static inline void onednn_conv_fwd_nchw(onednn_handle_t *handle, onednn_conv_handle_t *conv, float * src, float * filter, float * dst){
    auto src_memory = dnnl::memory( *conv->src_desc,*handle->eng);
    auto filter_memory = dnnl::memory(*conv->filter_desc, *handle->eng);
    auto dst_memory = dnnl::memory(*conv->dst_desc, *handle->eng);

    write_to_dnnl_memory(src, src_memory);
    write_to_dnnl_memory(filter, filter_memory);

    dnnl::convolution_forward conv_fwd({*conv->fwd_desc, *handle->eng});

    auto stream = dnnl::stream(*handle->eng);
    conv_fwd.execute(stream, {{DNNL_ARG_SRC, src_memory},
                                {DNNL_ARG_WEIGHTS, filter_memory},
                                {DNNL_ARG_DST, dst_memory}});
    stream.wait();
    read_from_dnnl_memory(dst, dst_memory);
}

static inline void onednn_conv_bwd_nchw(onednn_handle_t *handle, onednn_conv_handle_t *conv, float * src_grad, float * filter, float * dst_grad){
    auto stream = dnnl::stream(*handle->eng);

    auto src_grad_memory = dnnl::memory( *conv->src_desc,*handle->eng);
    auto filter_memory = dnnl::memory(*conv->filter_desc, *handle->eng);
    auto dst_grad_memory = dnnl::memory(*conv->dst_desc, *handle->eng);

    write_to_dnnl_memory(filter, filter_memory);
    write_to_dnnl_memory(dst_grad, dst_grad_memory);

    dnnl::convolution_backward_data conv_bwd({*conv->bwd_desc, *handle->eng, {*conv->fwd_desc, *handle->eng}});

    conv_bwd.execute(stream, {{DNNL_ARG_DIFF_SRC, src_grad_memory},
                                {DNNL_ARG_WEIGHTS, filter_memory},
                                {DNNL_ARG_DIFF_DST, dst_grad_memory}});
    stream.wait();
    read_from_dnnl_memory(src_grad, src_grad_memory);

}

static inline void onednn_conv_wrw_nchw(onednn_handle_t *handle, onednn_conv_handle_t *conv, float * src, float * filter_grad, float * dst_grad){

    auto stream = dnnl::stream(*handle->eng);

    auto src_memory = dnnl::memory( *conv->src_desc,*handle->eng);
    auto filter_grad_memory = dnnl::memory(*conv->filter_desc, *handle->eng);
    auto dst_grad_memory = dnnl::memory(*conv->dst_desc, *handle->eng);

    write_to_dnnl_memory(src, src_memory);
    write_to_dnnl_memory(dst_grad, dst_grad_memory);

    dnnl::convolution_backward_weights conv_wrw({*conv->wrw_desc, *handle->eng, {*conv->fwd_desc, *handle->eng}});

    conv_wrw.execute(stream, {{DNNL_ARG_SRC, src_memory},
                                {DNNL_ARG_DIFF_WEIGHTS, filter_grad_memory},
                                {DNNL_ARG_DIFF_DST, dst_grad_memory}});
    stream.wait();
    read_from_dnnl_memory(filter_grad, filter_grad_memory);
}

#define DNNL_CONV_WARP(dir, layout)                                                         \
    static inline void onednn_conv_ ## dir ## _ ## layout (float *ts, float *tf, float *td, \
    size_t n, size_t w, size_t h, size_t c, size_t k, size_t fx, size_t fy,                 \
    size_t px, size_t py, size_t sx, size_t sy, size_t dx, size_t dy, size_t group)         \
    {                                                                                       \
        onednn_handle_t onednn_h;                                                           \
        onednn_conv_handle_t onednn_conv_h;                                                 \
        onednn_init(&onednn_h);                                                             \
        onednn_conv_init(&onednn_conv_h,n,w,h,c,k,fx,fy,px,py,sx,sy,dx,dy,group);           \
        onednn_conv_## dir ## _ ## layout (&onednn_h, &onednn_conv_h, ts, tf, td);          \
        onednn_conv_destroy(&onednn_conv_h);                                                \
        onednn_destroy(&onednn_h);                                                          \
    }

DNNL_CONV_WARP(fwd, nchw)
DNNL_CONV_WARP(bwd, nchw)
DNNL_CONV_WARP(wrw, nchw)

#endif