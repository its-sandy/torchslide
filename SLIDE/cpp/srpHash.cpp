#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <chrono>
#include <vector>
#include <random>

void reset_hashes(
        torch::Tensor& nz_indices,
        torch::Tensor& plus_mask,
        torch::Tensor& minus_mask) {

    int64_t L = nz_indices.size(0);
    int64_t K = nz_indices.size(1);
    int64_t nz_dim = nz_indices.size(2);
    int64_t dim = plus_mask.size(2);

    auto nz_indices_0 = nz_indices.accessor<int32_t, 3>(); // L x K x nz_dim
    auto plus_mask_0 = plus_mask.accessor<bool, 3>(); // L x K x dim
    auto minus_mask_0 = minus_mask.accessor<bool, 3>(); // L x K x dim

    at::parallel_for(0, L * K, 0, [&](int64_t start, int64_t end) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + start;
        std::minstd_rand prng(seed);

        for (int64_t ii = start; ii < end; ii++) {
            int64_t l = ii/K, k = ii%K;
            auto nz_indices_2 = nz_indices_0[l][k];
            auto plus_mask_2 = plus_mask_0[l][k];
            auto minus_mask_2 = minus_mask_0[l][k];
            
            int32_t tot_left = (dim<<1), nz_left = (nz_dim<<1);
            int64_t ind = 0;
            for(int32_t d=0; d<dim; d++) {
                if(nz_left > 0) {
                    int32_t rn = prng() % tot_left;
                    if(rn < nz_left) {
                        nz_indices_2[ind] = d;
                        ind++;
                        if(rn & 1) {
                            plus_mask_2[d] = true;
                            minus_mask_2[d] = false;
                        }
                        else {
                            plus_mask_2[d] = false;
                            minus_mask_2[d] = true;
                        }
                        nz_left -= 2;
                    }
                    else {
                        plus_mask_2[d] = false;
                        minus_mask_2[d] = false;
                    }
                }
                else {
                    plus_mask_2[d] = false;
                    minus_mask_2[d] = false;
                }
                tot_left -= 2;
            }
        }
    });
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void get_hash_indices_dense_kernel(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& nz_indices,
        const torch::Tensor& plus_mask) {

    int64_t batch_size = in_values.size(0);
    int64_t L = nz_indices.size(0);
    int64_t K = nz_indices.size(1);
    int64_t nz_dim = nz_indices.size(2);

    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // Batch x L
    auto in_values_0 = in_values.accessor<scalar_t, 2>(); // Batch x dim
    auto nz_indices_0 = nz_indices.accessor<int32_t, 3>(); // L x K x nz_dim
    auto plus_mask_0 = plus_mask.accessor<bool, 3>(); // L x K x dim

    at::parallel_for(0, batch_size * L, 0, [&](int64_t start, int64_t end) {
        for (int64_t ii = start; ii < end; ii++) {
            int64_t i = ii/L, l = ii%L;
            auto in_values_1 = in_values_0[i];
            auto nz_indices_1 = nz_indices_0[l];
            auto plus_mask_1 = plus_mask_0[l];
            
            int32_t hash_index = 0;
            for(int32_t k=0; k<K; k++) {
                auto nz_indices_2 = nz_indices_1[k];
                auto plus_mask_2 = plus_mask_1[k];
                
                scalar_t res = 0;
                for(int64_t d=0; d<nz_dim; d++) {
                    auto index = nz_indices_2[d];
                    if(plus_mask_2[index])
                        res += in_values_1[index];
                    else
                        res -= in_values_1[index];
                }
                if(res < 0)
                    hash_index |= (1<<k);
            }
            hash_indices_0[i][l] = hash_index;
        }
    });
}

void get_hash_indices_dense(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& nz_indices,
        const torch::Tensor& plus_mask) {
    
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "get_hash_indices_dense", [&] {
            get_hash_indices_dense_kernel<scalar_t>(hash_indices, in_values, nz_indices, plus_mask);
        });
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void get_hash_indices_sparse_kernel(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& plus_mask,
        const torch::Tensor& minus_mask) {

    int64_t batch_size = in_values.size(0);
    int64_t active_in_dim = in_values.size(1);
    int64_t L = plus_mask.size(0);
    int64_t K = plus_mask.size(1);

    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // Batch x L
    auto in_values_0 = in_values.accessor<scalar_t, 2>(); // Batch x active_in_dim
    auto active_in_indices_0 = active_in_indices.accessor<int64_t,2>(); // Batch x active_in_dim
    auto plus_mask_0 = plus_mask.accessor<bool, 3>(); // L x K x dim
    auto minus_mask_0 = minus_mask.accessor<bool, 3>(); // L x K x dim

    at::parallel_for(0, batch_size * L, 0, [&](int64_t start, int64_t end) {
        for (int64_t ii = start; ii < end; ii++) {
            int64_t i = ii/L, l = ii%L;
            auto in_values_1 = in_values_0[i];
            auto active_in_indices_1 = active_in_indices_0[i];
            auto plus_mask_1 = plus_mask_0[l];
            auto minus_mask_1 = minus_mask_0[l];
            
            int32_t hash_index = 0;
            for(int32_t k=0; k<K; k++) {
                auto plus_mask_2 = plus_mask_1[k];
                auto minus_mask_2 = minus_mask_1[k];
                
                scalar_t res = 0;
                for(int64_t d=0; d<active_in_dim; d++) {
                    auto index = active_in_indices_1[d];
                    if(plus_mask_2[index])
                        res += in_values_1[d];
                    else if(minus_mask_2[index])
                        res -= in_values_1[d];
                }
                if(res < 0)
                    hash_index |= (1<<k);
            }
            hash_indices_0[i][l] = hash_index;
        }
    });
}

void get_hash_indices_sparse(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& plus_mask,
        const torch::Tensor& minus_mask) {
    
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "get_hash_indices_sparse", [&] {
            get_hash_indices_sparse_kernel<scalar_t>(hash_indices, in_values, active_in_indices, plus_mask, minus_mask);
        });
}
///////////////////////////////////////////////////////////////////////////////


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reset_hashes", &reset_hashes, "reset_hashes");
    m.def("get_hash_indices_dense", &get_hash_indices_dense, "get_hash_indices_dense");
    m.def("get_hash_indices_sparse", &get_hash_indices_sparse, "get_hash_indices_sparse");
}