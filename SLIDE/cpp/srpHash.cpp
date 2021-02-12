#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

template <typename scalar_t>
void get_hash_indices_sparse_kernel(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& hash_vecs) {
    // Some functionality of this function is similar to the sparse multiply
    // operation that was implemented already. But we don't use that to avoid
    // storing intermediate hash values separately, and to restrict number of parallel_fors used.
    
    int32_t batch_size = in_values.size(0);
    int32_t active_in_dim = in_values.size(1);
    int32_t L = hash_vecs.size(0);
    int32_t K = hash_vecs.size(1);

    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // Batch x L
    auto in_values_0 = in_values.accessor<scalar_t, 2>(); // Batch x active_in_dim
    auto active_in_indices_0 = active_in_indices.accessor<int32_t,2>(); // Batch x active_in_dim
    auto hash_vecs_0 = hash_vecs.accessor<scalar_t, 3>(); // L x K x dim

    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {
            auto in_values_1 = in_values_0[i];
            auto active_in_indices_1 = active_in_indices_0[i];
            
            for (int32_t l = 0; l < L; l++) {
                auto hash_vecs_1 = hash_vecs_0[l];
                
                int32_t hash_index = 0;
                for(int32_t k=0; k<K; k++) {
                    auto hash_vecs_2 = hash_vecs_1[k];
                    
                    scalar_t res = 0;
                    for(int32_t d=0; d<active_in_dim; d++) {
                        auto index = active_in_indices_1[d];
                        res += hash_vecs_2[index]*in_values_1[d]; // hash_vecs can also take values other than 0,1,-1
                    }
                    if(res < 0)
                        hash_index |= (1<<k);
                }
                hash_indices_0[i][l] = hash_index;
            }
        }
    });
}

void get_hash_indices_sparse(
        torch::Tensor& hash_indices,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& hash_vecs) {
    
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "get_hash_indices_sparse", [&] {
            get_hash_indices_sparse_kernel<scalar_t>(hash_indices, in_values, active_in_indices, hash_vecs);
        });
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void get_hash_indices_from_values_kernel(
        torch::Tensor& hash_indices,
        const torch::Tensor& hash_values) {

    int32_t batch_size = hash_values.size(0);
    int32_t L = hash_values.size(1);
    int32_t K = hash_values.size(2);

    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // Batch x L
    auto hash_values_0 = hash_values.accessor<scalar_t, 3>(); // Batch x L x K

    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {
            for (int32_t l = 0; l < L; l++) {
                auto hash_values_2 = hash_values_0[i][l];
                
                int32_t hash_index = 0;
                for(int32_t k=0; k<K; k++) {
                    if(hash_values_2[k] < 0)
                        hash_index |= (1<<k);
                }
                hash_indices_0[i][l] = hash_index;
            }
        }
    });
}

void get_hash_indices_from_values(
        torch::Tensor& hash_indices,
        const torch::Tensor& hash_values) {
    
    AT_DISPATCH_FLOATING_TYPES(
        hash_values.scalar_type(), "get_hash_indices_from_values", [&] {
            get_hash_indices_from_values_kernel<scalar_t>(hash_indices, hash_values);
        });
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_hash_indices_sparse", &get_hash_indices_sparse, "get_hash_indices_sparse");
    m.def("get_hash_indices_from_values", &get_hash_indices_from_values, "get_hash_indices_from_values");
}