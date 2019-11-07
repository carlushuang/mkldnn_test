#ifndef __MKLDNN_CONV_H
#define __MKLDNN_CONV_H

#include <mkldnn.hpp>
static inline size_t md_conv_out_size(size_t in_size, size_t pad, size_t dilation, size_t ksize, size_t stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}

static inline void md_conv_nchw_2_cnhw(float *dst, const float * src, size_t n, size_t c, size_t h, size_t w){
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
static inline void md_conv_cnhw_2_nchw(float *dst, const float * src, size_t n, size_t c, size_t h, size_t w){
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
    mkldnn::engine * eng;
}md_handle;
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

    mkldnn::memory::desc *src_desc;
    mkldnn::memory::desc *filter_desc;
    mkldnn::memory::desc *dst_desc;

    mkldnn::convolution_forward::desc           *fwd_desc;
    mkldnn::convolution_backward_data::desc     *bwd_d_desc;
    mkldnn::convolution_backward_weights::desc  *bwd_f_desc;

    //mkldnn::convolution_forward *fwd;
}md_conv_handle;

static inline void md_init(md_handle * mh){
    mh->eng = new mkldnn::engine(mkldnn::engine::kind::cpu,0);
}
static inline void md_destroy(md_handle * mh){
    delete mh->eng;
}

#if MKLDNN_VERSION_MAJOR >= 1
inline void read_from_mkldnn_memory(void *handle, mkldnn::memory &mem) {
    mkldnn::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == mkldnn::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        std::copy(src, src + bytes, (uint8_t *)handle);
    }
#if MKLDNN_GPU_RUNTIME == MKLDNN_RUNTIME_OCL
    else if (eng.get_kind() == mkldnn::engine::kind::gpu) {
        mkldnn::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();

        cl_int ret = clEnqueueReadBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueReadBuffer failed. Status Code: "
                    + std::to_string(ret) + "\n");
    }
#endif
}

// Read from handle, write to memory
inline void write_to_mkldnn_memory(void *handle, mkldnn::memory &mem) {
    mkldnn::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == mkldnn::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        std::copy((uint8_t *)handle, (uint8_t *)handle + bytes, dst);
    }
#if MKLDNN_GPU_RUNTIME == MKLDNN_RUNTIME_OCL
    else if (eng.get_kind() == mkldnn::engine::kind::gpu) {
        mkldnn::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();
        size_t bytes = mem.get_desc().get_size();

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error(
                    "clEnqueueWriteBuffer failed. Status Code: "
                    + std::to_string(ret) + "\n");
    }
#endif
}
#endif

static inline void md_conv_init(md_conv_handle *conv,size_t n, size_t w, size_t h, size_t c, size_t k, size_t fx, size_t fy, size_t px, size_t py, size_t sx, size_t sy, size_t dx, size_t dy){
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
    conv->ow = md_conv_out_size(w, px, dx, fx, sx);
    conv->oh = md_conv_out_size(h, py, dy, fy, sy);
#if MKLDNN_VERSION_MAJOR >= 1
    conv->src_desc = new mkldnn::memory::desc({(int)n,(int)c,(int)h,(int)w}, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::nchw);
    conv->filter_desc = new mkldnn::memory::desc({(int)k,(int)c,(int)fy,(int)fx}, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::oihw);
    conv->dst_desc = new mkldnn::memory::desc({(int)n,(int)k,(int)conv->oh,(int)conv->ow}, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::nchw);

    conv->fwd_desc = new mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward,
                        mkldnn::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {(int)sy,(int)sx},{(int)dy-1,(int)dx-1},{(int)py,(int)px},{(int)py,(int)px});
    // note in mkl-dnn, dilation is 0 when unit dilation.
    conv->bwd_d_desc = new mkldnn::convolution_backward_data::desc(mkldnn::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {(int)sy,(int)sx},{(int)dy-1,(int)dx-1},{(int)py,(int)px},{(int)py,(int)px});
    
    conv->bwd_f_desc = new mkldnn::convolution_backward_weights::desc(mkldnn::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {(int)sy,(int)sx},{(int)dy-1,(int)dx-1},{(int)py,(int)px},{(int)py,(int)px});

#else
    conv->src_desc = new mkldnn::memory::desc({(int)n,(int)c,(int)h,(int)w}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);
    conv->filter_desc = new mkldnn::memory::desc({(int)k,(int)c,(int)fy,(int)fx}, mkldnn::memory::data_type::f32, mkldnn::memory::format::oihw);
    conv->dst_desc = new mkldnn::memory::desc({(int)n,(int)k,(int)conv->oh,(int)conv->ow}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);

    conv->fwd_desc = new mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward,
                        mkldnn::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {(int)sy,(int)sx},{(int)dy-1,(int)dx-1},{(int)py,(int)px},{(int)py,(int)px},mkldnn::padding_kind::zero);
    // note in mkl-dnn, dilation is 0 when unit dilation.
    conv->bwd_d_desc = new mkldnn::convolution_backward_data::desc(mkldnn::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {(int)sy,(int)sx},{(int)dy-1,(int)dx-1},{(int)py,(int)px},{(int)py,(int)px},mkldnn::padding_kind::zero);
    
    conv->bwd_f_desc = new mkldnn::convolution_backward_weights::desc(mkldnn::algorithm::convolution_direct,
                        *conv->src_desc,*conv->filter_desc,*conv->dst_desc,
                        {(int)sy,(int)sx},{(int)dy-1,(int)dx-1},{(int)py,(int)px},{(int)py,(int)px},mkldnn::padding_kind::zero);
#endif
}
static inline void md_conv_destroy(md_conv_handle * conv){
    delete conv->src_desc;
    delete conv->filter_desc;
    delete conv->dst_desc;
    delete conv->fwd_desc;
    delete conv->bwd_d_desc;
    delete conv->bwd_f_desc;
}
static inline void md_conv_fwd_nchw(md_handle *mh, md_conv_handle *conv, float * src, float * filter, float * dst){
#if MKLDNN_VERSION_MAJOR >= 1
    auto src_memory = mkldnn::memory( *conv->src_desc,*mh->eng);
    auto filter_memory = mkldnn::memory(*conv->filter_desc, *mh->eng);
    auto dst_memory = mkldnn::memory(*conv->dst_desc, *mh->eng);

    write_to_mkldnn_memory(src, src_memory);
    write_to_mkldnn_memory(filter, filter_memory);

    mkldnn::convolution_forward conv_fwd({*conv->fwd_desc, *mh->eng});

    auto stream = mkldnn::stream(*mh->eng);
    conv_fwd.execute(stream, {{MKLDNN_ARG_SRC, src_memory},
                                {MKLDNN_ARG_WEIGHTS, filter_memory},
                                {MKLDNN_ARG_DST, dst_memory}});
    stream.wait();
    read_from_mkldnn_memory(dst, dst_memory);
#else
    auto src_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src  );
    auto filter_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter  );
    auto dst_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst);
    mkldnn::convolution_forward conv_fwd({*conv->fwd_desc,*mh->eng},src_memory,filter_memory,dst_memory  );
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_fwd}).wait();
#endif
}

static inline void md_conv_fwd_cnhw(md_handle *mh, md_conv_handle *conv, float * src, float * filter, float * dst){
    int n = conv->src_desc->data.dims[0];
    int c = conv->src_desc->data.dims[1];
    int h = conv->src_desc->data.dims[2];
    int w = conv->src_desc->data.dims[3];
    int k = conv->dst_desc->data.dims[1];
    int oh = conv->dst_desc->data.dims[2];
    int ow = conv->dst_desc->data.dims[3];
    float * src_nchw = new float[n*c*h*w];
    float * dst_nchw = new float[n*k*oh*ow];
    md_conv_cnhw_2_nchw(src_nchw, src, n, c, h, w);
#if MKLDNN_VERSION_MAJOR >= 1
    auto stream = mkldnn::stream(*mh->eng);

    auto src_memory = mkldnn::memory( *conv->src_desc,*mh->eng);
    auto filter_memory = mkldnn::memory(*conv->filter_desc, *mh->eng);
    auto dst_memory = mkldnn::memory(*conv->dst_desc, *mh->eng);

    write_to_mkldnn_memory(src_nchw, src_memory);
    write_to_mkldnn_memory(filter, filter_memory);

    mkldnn::convolution_forward conv_fwd({*conv->fwd_desc, *mh->eng});

    conv_fwd.execute(stream, {{MKLDNN_ARG_SRC, src_memory},
                                {MKLDNN_ARG_WEIGHTS, filter_memory},
                                {MKLDNN_ARG_DST, dst_memory}});
    stream.wait();
    read_from_mkldnn_memory(dst_nchw, dst_memory);
#else
    auto src_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src_nchw  );
    auto filter_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter  );
    auto dst_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst_nchw);
    mkldnn::convolution_forward conv_fwd({*conv->fwd_desc,*mh->eng},src_memory,filter_memory,dst_memory  );
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_fwd}).wait();
#endif
    md_conv_nchw_2_cnhw(dst, dst_nchw, n, k, oh, ow);
    delete [] src_nchw;
    delete [] dst_nchw;
}
static inline void md_conv_bwd_d_nchw(md_handle *mh, md_conv_handle *conv, float * src_grad, float * filter, float * dst_grad){
#if MKLDNN_VERSION_MAJOR >= 1
    auto stream = mkldnn::stream(*mh->eng);

    auto src_grad_memory = mkldnn::memory( *conv->src_desc,*mh->eng);
    auto filter_memory = mkldnn::memory(*conv->filter_desc, *mh->eng);
    auto dst_grad_memory = mkldnn::memory(*conv->dst_desc, *mh->eng);

    write_to_mkldnn_memory(filter, filter_memory);
    write_to_mkldnn_memory(dst_grad, dst_grad_memory);

    mkldnn::convolution_backward_data conv_bwd_d({*conv->bwd_d_desc, *mh->eng, {*conv->fwd_desc, *mh->eng}});

    conv_bwd_d.execute(stream, {{MKLDNN_ARG_DIFF_SRC, src_grad_memory},
                                {MKLDNN_ARG_WEIGHTS, filter_memory},
                                {MKLDNN_ARG_DIFF_DST, dst_grad_memory}});
    stream.wait();
    read_from_mkldnn_memory(src_grad, src_grad_memory);
#else
    auto src_grad_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src_grad  );
    auto filter_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter  );
    auto dst_grad_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst_grad);
    mkldnn::convolution_backward_data conv_bwd_d({*conv->bwd_d_desc,*mh->eng, {*conv->fwd_desc,*mh->eng}},dst_grad_memory,filter_memory,src_grad_memory );
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_bwd_d}).wait();
#endif
}
static inline void md_conv_bwd_d_cnhw(md_handle *mh, md_conv_handle *conv, float * src_grad, float * filter, float * dst_grad){
    int n = conv->src_desc->data.dims[0];
    int c = conv->src_desc->data.dims[1];
    int h = conv->src_desc->data.dims[2];
    int w = conv->src_desc->data.dims[3];
    int k = conv->dst_desc->data.dims[1];
    int oh = conv->dst_desc->data.dims[2];
    int ow = conv->dst_desc->data.dims[3];
    float * src_grad_nchw = new float[n*c*h*w];
    float * dst_grad_nchw = new float[n*k*oh*ow];
    md_conv_cnhw_2_nchw(dst_grad_nchw, dst_grad, n, k, oh, ow);
#if MKLDNN_VERSION_MAJOR >= 1
    auto stream = mkldnn::stream(*mh->eng);

    auto src_grad_memory = mkldnn::memory( *conv->src_desc,*mh->eng);
    auto filter_memory = mkldnn::memory(*conv->filter_desc, *mh->eng);
    auto dst_grad_memory = mkldnn::memory(*conv->dst_desc, *mh->eng);

    write_to_mkldnn_memory(filter, filter_memory);
    write_to_mkldnn_memory(dst_grad_nchw, dst_grad_memory);

    mkldnn::convolution_backward_data conv_bwd_d({*conv->bwd_d_desc, *mh->eng, {*conv->fwd_desc, *mh->eng}});

    conv_bwd_d.execute(stream, {{MKLDNN_ARG_DIFF_SRC, src_grad_memory},
                                {MKLDNN_ARG_WEIGHTS, filter_memory},
                                {MKLDNN_ARG_DIFF_DST, dst_grad_memory}});
    stream.wait();
    read_from_mkldnn_memory(src_grad_nchw, src_grad_memory);
#else
    auto src_grad_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src_grad_nchw  );
    auto filter_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter  );
    auto dst_grad_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst_grad_nchw);
    mkldnn::convolution_backward_data conv_bwd_d({*conv->bwd_d_desc,*mh->eng, {*conv->fwd_desc,*mh->eng}},dst_grad_memory,filter_memory,src_grad_memory );
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_bwd_d}).wait();
#endif
    md_conv_nchw_2_cnhw(src_grad, src_grad_nchw, n, c, h, w);
    delete [] src_grad_nchw;
    delete [] dst_grad_nchw;
}
static inline void md_conv_bwd_f_nchw(md_handle *mh, md_conv_handle *conv, float * src, float * filter_grad, float * dst_grad){
#if MKLDNN_VERSION_MAJOR >= 1
    auto stream = mkldnn::stream(*mh->eng);

    auto src_memory = mkldnn::memory( *conv->src_desc,*mh->eng);
    auto filter_grad_memory = mkldnn::memory(*conv->filter_desc, *mh->eng);
    auto dst_grad_memory = mkldnn::memory(*conv->dst_desc, *mh->eng);

    write_to_mkldnn_memory(src, src_memory);
    write_to_mkldnn_memory(dst_grad, dst_grad_memory);

    mkldnn::convolution_backward_weights conv_bwd_f({*conv->bwd_f_desc, *mh->eng, {*conv->fwd_desc, *mh->eng}});

    conv_bwd_f.execute(stream, {{MKLDNN_ARG_SRC, src_memory},
                                {MKLDNN_ARG_DIFF_WEIGHTS, filter_grad_memory},
                                {MKLDNN_ARG_DIFF_DST, dst_grad_memory}});
    stream.wait();
    read_from_mkldnn_memory(filter_grad, filter_grad_memory);
#else
    auto src_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src  );
    auto filter_grad_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter_grad  );
    auto dst_grad_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst_grad);
    mkldnn::convolution_backward_weights conv_bwd_f({*conv->bwd_f_desc,*mh->eng, {*conv->fwd_desc,*mh->eng}},src_memory,dst_grad_memory,filter_grad_memory);
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_bwd_f}).wait();
#endif
}
static inline void md_conv_bwd_f_cnhw(md_handle *mh, md_conv_handle *conv, float * src, float * filter_grad, float * dst_grad){
    int n = conv->src_desc->data.dims[0];
    int c = conv->src_desc->data.dims[1];
    int h = conv->src_desc->data.dims[2];
    int w = conv->src_desc->data.dims[3];
    int k = conv->dst_desc->data.dims[1];
    int oh = conv->dst_desc->data.dims[2];
    int ow = conv->dst_desc->data.dims[3];
    float * src_nchw = new float[n*c*h*w];
    float * dst_grad_nchw = new float[n*k*oh*ow];
    md_conv_cnhw_2_nchw(dst_grad_nchw, dst_grad, n, k, oh, ow);
    md_conv_cnhw_2_nchw(src_nchw, src, n, c, h, w);
#if MKLDNN_VERSION_MAJOR >= 1
    auto stream = mkldnn::stream(*mh->eng);

    auto src_memory = mkldnn::memory( *conv->src_desc,*mh->eng);
    auto filter_grad_memory = mkldnn::memory(*conv->filter_desc, *mh->eng);
    auto dst_grad_memory = mkldnn::memory(*conv->dst_desc, *mh->eng);

    write_to_mkldnn_memory(src_nchw, src_memory);
    write_to_mkldnn_memory(dst_grad_nchw, dst_grad_memory);

    mkldnn::convolution_backward_weights conv_bwd_f({*conv->bwd_f_desc, *mh->eng, {*conv->fwd_desc, *mh->eng}});

    conv_bwd_f.execute(stream, {{MKLDNN_ARG_SRC, src_memory},
                                {MKLDNN_ARG_DIFF_WEIGHTS, filter_grad_memory},
                                {MKLDNN_ARG_DIFF_DST, dst_grad_memory}});
    stream.wait();
    read_from_mkldnn_memory(filter_grad, filter_grad_memory);
#else
    auto src_memory = mkldnn::memory( {*conv->src_desc,*mh->eng}, src_nchw  );
    auto filter_grad_memory = mkldnn::memory({*conv->filter_desc, *mh->eng}, filter_grad  );
    auto dst_grad_memory = mkldnn::memory({*conv->dst_desc, *mh->eng}, dst_grad_nchw);
    mkldnn::convolution_backward_weights conv_bwd_f({*conv->bwd_f_desc,*mh->eng, {*conv->fwd_desc,*mh->eng}},src_memory,dst_grad_memory,filter_grad_memory);
    mkldnn::stream(mkldnn::stream::kind::eager).submit({conv_bwd_f}).wait();
#endif
    delete [] src_nchw;
    delete [] dst_grad_nchw;
}

#define MKLDNN_CONV_WARP(dir, layout)                                               \
    static inline void mkldnn_conv_ ## dir ## _ ## layout (float *ts, float *tf, float *td, \
    size_t n, size_t w, size_t h, size_t c, size_t k, size_t fx, size_t fy, size_t px, size_t py, size_t sx, size_t sy, size_t dx, size_t dy) \
    {                                                                               \
        md_handle md_h;                                                             \
        md_conv_handle md_conv_h;                                                   \
        md_init(&md_h);                                                             \
        md_conv_init(&md_conv_h,n,w,h,c,k,fx,fy,px,py,sx,sy,dx,dy);                 \
        md_conv_## dir ## _ ## layout (&md_h, &md_conv_h, ts, tf, td);              \
        md_conv_destroy(&md_conv_h);                                                \
        md_destroy(&md_h);                                                          \
    }

MKLDNN_CONV_WARP(fwd, nchw)
MKLDNN_CONV_WARP(fwd, cnhw)
MKLDNN_CONV_WARP(bwd_d, nchw)
MKLDNN_CONV_WARP(bwd_d, cnhw)
MKLDNN_CONV_WARP(bwd_f, nchw)
MKLDNN_CONV_WARP(bwd_f, cnhw)

#endif
