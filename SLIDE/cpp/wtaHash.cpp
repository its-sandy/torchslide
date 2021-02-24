#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

template <typename scalar_t>
void get_hash_indices_dense_kernel(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& perm_pos,
        const torch::Tensor& perm_ind,
        int32_t K,
        int32_t L,
        int32_t shift_len) {
    
    int32_t batch_size = in_values.size(0);
    int32_t in_dim = in_values.size(1);
    int32_t num_full_perms = perm_pos.size(0);
    int32_t num_hashes = K*L;

    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // Batch x L
    auto in_values_0 = in_values.accessor<scalar_t, 2>(); // Batch x in_dim
    auto perm_pos_0 = perm_pos.accessor<int32_t,2>(); // num_full_perms x in_dim
    auto perm_ind_0 = perm_ind.accessor<int32_t,2>(); // num_full_perms x in_dim

    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {
            auto in_values_1 = in_values_0[i];
            std::vector<scalar_t> max_vals(num_hashes, INT_MIN);
            std::vector<int32_t> max_inds(num_hashes, 0);

            for (int32_t p=0; p<num_full_perms; p++) {
                auto perm_pos_1 = perm_pos_0[p];
                auto perm_ind_1 = perm_ind_0[p];

                for(int32_t d=0; d<in_dim; d++) {
                    int32_t perm_id = perm_ind_1[d];
                    if(perm_id < num_hashes && max_vals[perm_id] < in_values_1[d]) {
                        max_vals[perm_id] = in_values_1[d];
                        max_inds[perm_id] = perm_pos_1[d];
                    }
                }
            }

            auto hash_indices_1 = hash_indices_0[i];
            int32_t j=0;
            for(int32_t l=0; l<L; l++) {
                int32_t hash_index = 0;
                for(int32_t k=0; k<K; k++) {
                    hash_index += (max_inds[j]<<(k*shift_len));
                    j++;
                }
                hash_indices_1[l] = hash_index;
            }
        }
    });
}

void get_hash_indices_dense(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& perm_pos,
        const torch::Tensor& perm_ind,
        int32_t K,
        int32_t L,
        int32_t shift_len) {
    
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "get_hash_indices_dense", [&] {
            get_hash_indices_dense_kernel<scalar_t>(hash_indices, in_values, perm_pos, perm_ind, K, L, shift_len);
        });
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void get_hash_indices_sparse_kernel(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& perm_pos,
        const torch::Tensor& perm_ind,
        int32_t K,
        int32_t L,
        int32_t shift_len) {
    
    int32_t batch_size = in_values.size(0);
    int32_t active_in_dim = in_values.size(1);
    int32_t num_full_perms = perm_pos.size(0);
    int32_t num_hashes = K*L;

    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // Batch x L
    auto in_values_0 = in_values.accessor<scalar_t, 2>(); // Batch x active_in_dim
    auto active_in_indices_0 = active_in_indices.accessor<int32_t,2>(); // Batch x active_in_dim
    auto perm_pos_0 = perm_pos.accessor<int32_t,2>(); // num_full_perms x in_dim
    auto perm_ind_0 = perm_ind.accessor<int32_t,2>(); // num_full_perms x in_dim

    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {
            auto in_values_1 = in_values_0[i];
            auto active_in_indices_1 = active_in_indices_0[i];
            std::vector<scalar_t> max_vals(num_hashes, INT_MIN);
            std::vector<int32_t> max_inds(num_hashes, 0);

            for (int32_t p=0; p<num_full_perms; p++) {
                auto perm_pos_1 = perm_pos_0[p];
                auto perm_ind_1 = perm_ind_0[p];

                for(int32_t d=0; d<active_in_dim; d++) {
                    int32_t index = active_in_indices_1[d];
                    int32_t perm_id = perm_ind_1[index];
                    if(perm_id < num_hashes && max_vals[perm_id] < in_values_1[d]) {
                        max_vals[perm_id] = in_values_1[d];
                        max_inds[perm_id] = perm_pos_1[index];
                    }
                }
            }

            auto hash_indices_1 = hash_indices_0[i];
            int32_t j=0;
            for(int32_t l=0; l<L; l++) {
                int32_t hash_index = 0;
                for(int32_t k=0; k<K; k++) {
                    hash_index += (max_inds[j]<<(k*shift_len));
                    j++;
                }
                hash_indices_1[l] = hash_index;
            }
        }
    });
}

void get_hash_indices_sparse(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& perm_pos,
        const torch::Tensor& perm_ind,
        int32_t K,
        int32_t L,
        int32_t shift_len) {
    
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "get_hash_indices_sparse", [&] {
            get_hash_indices_sparse_kernel<scalar_t>(hash_indices, in_values, active_in_indices, perm_pos, perm_ind, K, L, shift_len);
        });
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_hash_indices_dense", &get_hash_indices_dense, "get_hash_indices_dense");
    m.def("get_hash_indices_sparse", &get_hash_indices_sparse, "get_hash_indices_sparse");
}