#include <hip/hip_runtime.h>

// design block_size 256
extern "C" __global__
void hip_naive_conv_fwd_nchw_fp32(
    const float * __restrict__ p_in,
    const float * __restrict__ p_wei,
    float * __restrict__       p_out,
    int hi,
    int wi,
    int n,
    int k_per_group,
    int c_per_group,
    int ho,
    int wo,
    int sy,
    int sx,
    int dy,
    int dx,
    int py,
    int px,
    int fy,
    int fx,
    int group)
{
    int k = k_per_group * group;
    int c = c_per_group * group;
    int thread_length = ho * wo;
    int bid = blockIdx.x;
    int ik = bid % k_per_group;
    int in = (bid / k_per_group) % n;
    int ig = bid / (n * k_per_group);
    

    p_in  += in * c * hi * wi + ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256){
        int iho = tid / wo;
        int iwo = tid % wo;

        float value = .0f;

        for (int ic = 0; ic < c_per_group; ic++) {
            for (int iy = 0; iy < fy; iy++) {
                int valid_h = 1;
                int cur_h = sy * iho - py + dy * iy;
                if (cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for (int ix = 0; ix < fx; ix++) {
                    int valid_w = 1;
                    int cur_w = sx * iwo - px + dx * ix;
                    if (cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h){
                        int i_idx = ic * hi * wi + cur_h * wi + cur_w;
                        int w_idx = ic * fy * fx + iy * fx + ix;
                        value += p_in[i_idx] * p_wei[w_idx];
                    }
                }
            }
        }
        int o_idx = iho * wo + iwo;
        p_out[o_idx] = value;
    }
}

extern "C" __global__
void hip_naive_conv_bwd_nchw_fp32(
    float * __restrict__       p_in,
    const float * __restrict__ p_wei,
    const float * __restrict__ p_out,
    int hi,
    int wi,
    int n,
    int k_per_group,
    int c_per_group,
    int ho,
    int wo,
    int sy,
    int sx,
    int dy,
    int dx,
    int py,
    int px,
    int fy,
    int fx,
    int group)
{
    int k = k_per_group * group;
    int c = c_per_group * group;
    int thread_length = hi * wi;
    int bid = blockIdx.x;
    int ic = bid % c_per_group;
    int in = (bid / c_per_group) % n;
    int ig = bid / (n * c_per_group);

    p_in  += in * c * hi * wi + ig * c_per_group * hi * wi + ic * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ic * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256){
        int ihi = tid / wi;
        int iwi = tid % wi;

        float value = .0f;

        for (int ik = 0; ik < k_per_group; ik++) {
            for (int iy = 0; iy < fy; iy++) {
                int valid_h = 1;
                int cur_ho = ihi + py - dy * iy;    // cur_h = sy*iho-py+dy*iy;
                if (cur_ho < 0 || cur_ho % sy)
                    valid_h &= 0;
                cur_ho /= sy;
                if (cur_ho >= ho)
                    valid_h &= 0;
                for (int ix = 0; ix < fx; ix++) {
                    int valid_w = 1;
                    int cur_wo = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                    if (cur_wo < 0 || cur_wo % sx)
                        valid_w &= 0;
                    cur_wo /= sx;
                    if (cur_wo >= wo)
                        valid_w &= 0;

                    if(valid_h & valid_w){
                        int o_idx = ik * ho * wo +  cur_ho * wo + cur_wo;
                        int f_idx = ik * c_per_group * fy * fx + iy * fx + ix;
                        value += p_out[o_idx] * p_wei[f_idx];
                    }
                }
            }
        }
        int i_idx = ihi * wi + iwi;
        p_in[i_idx] = value;
    }
}

extern "C" __global__
void hip_naive_conv_wrw_nchw_fp32(
    const float * __restrict__ p_in,
    float * __restrict__       p_wei,
    const float * __restrict__ p_out,
    int hi,
    int wi,
    int n,
    int k_per_group,
    int c_per_group,
    int ho,
    int wo,
    int sy,
    int sx,
    int dy,
    int dx,
    int py,
    int px,
    int fy,
    int fx,
    int group)
{
    int k = k_per_group * group;
    int c = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid = blockIdx.x;
    int ik = bid % k_per_group;
    int ig = bid / k_per_group;

    p_in  += ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256){
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int ic = tid / (fx * fy);

        float value = .0f;
        
        for (int in = 0; in < n; in++) {
            for (int iho = 0; iho < ho; iho++) {
                int valid_h = 1;
                int cur_h = sy * iho - py + dy * iy;
                if (cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for (int iwo = 0; iwo < wo; iwo++) {
                    int valid_w = 1;
                    int cur_w = sx * iwo - px + dx * ix;
                    if (cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_h & valid_w){
                        int i_idx = in * c * hi * wi + ic * hi * wi + cur_h * wi + cur_w;
                        int o_idx = in * k * ho * wo + iho * wo + iwo;
                        value += p_in[i_idx] * p_out[o_idx];
                    }
                }
            }
        }
        int f_idx = ic * fy * fx + iy * fx + ix;
        p_wei[f_idx] = value;
    }
}