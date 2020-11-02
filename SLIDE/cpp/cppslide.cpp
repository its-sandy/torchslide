#include <torch/extension.h>
#include <vector>
#include <ATen/Parallel.h>

template <typename scalar_t>
void diso_non_threaded_kernel(
        torch::Tensor& out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    int64_t batch_size = active_out_indices.size(0);
    int64_t active_out_dim = active_out_indices.size(1);
    int64_t in_dim = weights.size(1);

    auto out_values_0 = out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto active_out_indices_0 = active_out_indices.accessor<int64_t,2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();
    auto bias_0 = bias.accessor<scalar_t, 1>();

    for(int64_t i=0; i<batch_size; i++){
        auto out_values_1 = out_values_0[i];
        auto in_values_1 = in_values_0[i];
        auto active_out_indices_1 = active_out_indices_0[i];
        
        for(int64_t j=0; j<active_out_dim; j++){
            scalar_t &res = out_values_1[j];
            int64_t out_index = active_out_indices_1[j];
            auto weights_1 = weights_0[out_index];

            res = bias_0[out_index];
            for(int64_t k=0; k<in_dim; k++)
                res += in_values_1[k]*weights_1[k];
        }
    }
}

torch::Tensor diso_non_threaded(
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    auto out_values = torch::empty(active_out_indices.sizes(), in_values.options());
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "diso_non_threaded", [&] {
            diso_non_threaded_kernel<scalar_t>(out_values, in_values, active_out_indices, weights, bias);
        });
    return out_values;
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void diso_kernel(
        torch::Tensor& out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    int64_t batch_size = active_out_indices.size(0);
    int64_t active_out_dim = active_out_indices.size(1);
    int64_t in_dim = weights.size(1);

    auto out_values_0 = out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto active_out_indices_0 = active_out_indices.accessor<int64_t,2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();
    auto bias_0 = bias.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
        for (int64_t i = start; i < end; i++){
            auto out_values_1 = out_values_0[i];
            auto in_values_1 = in_values_0[i];
            auto active_out_indices_1 = active_out_indices_0[i];
            
            for(int64_t j=0; j<active_out_dim; j++){
                scalar_t &res = out_values_1[j];
                int64_t out_index = active_out_indices_1[j];
                auto weights_1 = weights_0[out_index];

                res = bias_0[out_index];
                for(int64_t k=0; k<in_dim; k++)
                    res += in_values_1[k]*weights_1[k];
            }
        }
    });
}

torch::Tensor diso(
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){
    // dense input sparse output

    auto out_values = torch::empty(active_out_indices.sizes(), in_values.options());
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "diso", [&] {
            diso_kernel<scalar_t>(out_values, in_values, active_out_indices, weights, bias);
        });
    return out_values;
}

template <typename scalar_t>
void diso_backward_kernel(
        torch::Tensor& grad_in_values,
        torch::Tensor& grad_weights,
        torch::Tensor& grad_bias,
        const torch::Tensor& grad_out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& weights){

    int64_t batch_size = active_out_indices.size(0);
    int64_t active_out_dim = active_out_indices.size(1);
    int64_t in_dim = weights.size(1);

    auto grad_in_values_0 = grad_in_values.accessor<scalar_t, 2>();
    auto grad_weights_0 = grad_weights.accessor<scalar_t, 2>();
    auto grad_bias_0 = grad_bias.accessor<scalar_t, 1>();
    auto grad_out_values_0 = grad_out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto active_out_indices_0 = active_out_indices.accessor<int64_t,2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
        for (int64_t i = start; i < end; i++){
            auto grad_out_values_1 = grad_out_values_0[i];
            auto grad_in_values_1 = grad_in_values_0[i];
            auto in_values_1 = in_values_0[i];
            auto active_out_indices_1 = active_out_indices_0[i];
            
            for(int64_t j=0; j<active_out_dim; j++){
                scalar_t grad = grad_out_values_1[j];
                int64_t out_index = active_out_indices_1[j];
                auto weights_1 = weights_0[out_index];
                auto grad_weights_1 = grad_weights_0[out_index];

                for(int64_t k=0; k<in_dim; k++){
                    grad_in_values_1[k] += weights_1[k]*grad;
                    grad_weights_1[k] += in_values_1[k]*grad;
                }
                grad_bias_0[out_index] += grad;
            }
        }
    });
}

std::vector<torch::Tensor> diso_backward(
        const torch::Tensor& grad_out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    auto grad_in_values = torch::zeros_like(in_values);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_bias = torch::zeros_like(bias);

    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "diso_backward", [&] {
            diso_backward_kernel<scalar_t>(grad_in_values, grad_weights, grad_bias,
                                          grad_out_values, in_values, active_out_indices, weights);
        });
    return {grad_in_values, grad_weights, grad_bias};
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void sido_kernel(
        torch::Tensor& out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    int64_t batch_size = active_in_indices.size(0);
    int64_t active_in_dim = active_in_indices.size(1);
    int64_t out_dim = weights.size(0);

    auto out_values_0 = out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto active_in_indices_0 = active_in_indices.accessor<int64_t,2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();
    auto bias_0 = bias.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
        for (int64_t i = start; i < end; i++){
            auto out_values_1 = out_values_0[i];
            auto in_values_1 = in_values_0[i];
            auto active_in_indices_1 = active_in_indices_0[i];
            
            for(int64_t j=0; j<out_dim; j++){
                scalar_t &res = out_values_1[j];
                auto weights_1 = weights_0[j];

                res = bias_0[j];
                for(int64_t k=0; k<active_in_dim; k++)
                    res += in_values_1[k]*weights_1[active_in_indices_1[k]];
            }
        }
    });
}

torch::Tensor sido(
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){
    // sparse input dense output

    auto out_values = torch::empty({in_values.size(0), weights.size(0)}, in_values.options());
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "sido", [&] {
            sido_kernel<scalar_t>(out_values, in_values, active_in_indices, weights, bias);
        });
    return out_values;
}

template <typename scalar_t>
void sido_backward_kernel(
        torch::Tensor& grad_in_values,
        torch::Tensor& grad_weights,
        torch::Tensor& grad_bias,
        const torch::Tensor& grad_out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights){

    int64_t batch_size = active_in_indices.size(0);
    int64_t active_in_dim = active_in_indices.size(1);
    int64_t out_dim = weights.size(0);

    auto grad_in_values_0 = grad_in_values.accessor<scalar_t, 2>();
    auto grad_weights_0 = grad_weights.accessor<scalar_t, 2>();
    auto grad_bias_0 = grad_bias.accessor<scalar_t, 1>();
    auto grad_out_values_0 = grad_out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto active_in_indices_0 = active_in_indices.accessor<int64_t,2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();

    // version 1 (might lead to large collisions for weights and bias gradients)
    // at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
    //     for (int64_t i = start; i < end; i++){
    //         auto grad_out_values_1 = grad_out_values_0[i];
    //         auto grad_in_values_1 = grad_in_values_0[i];
    //         auto in_values_1 = in_values_0[i];
    //         auto active_in_indices_1 = active_in_indices_0[i];
            
    //         int64_t cyc_shift_l = (out_dim/batch_size)*i;
    //         int64_t cyc_shift_r = cyc_shift_l + out_dim;
    //         for(int64_t jj=cyc_shift_l; jj<cyc_shift_r; jj++){
    //             int64_t j = jj%out_dim; // doing to possibly decrease collision probability
    //             scalar_t grad = grad_out_values_1[j];
    //             auto weights_1 = weights_0[j];
    //             auto grad_weights_1 = grad_weights_0[j];

    //             for(int64_t k=0; k<active_in_dim; k++){
    //                 int64_t in_index = active_in_indices_1[k];
    //                 grad_in_values_1[k] += weights_1[in_index]*grad;
    //                 grad_weights_1[in_index] += in_values_1[k]*grad;
    //             }
    //             grad_bias_0[j] += grad; // might be better to calculate grad_bias in python using builtin pytorch function
    //         }
    //     }
    // });

    // version 2 (reduces collisions for weights and bias gradients at the 
    // expense of 2 parallel_for dispatches and maybe more accessor indexing)
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
        for (int64_t i = start; i < end; i++){
            auto grad_out_values_1 = grad_out_values_0[i];
            auto grad_in_values_1 = grad_in_values_0[i];
            auto in_values_1 = in_values_0[i];
            auto active_in_indices_1 = active_in_indices_0[i];
            
            for(int64_t j=0; j<out_dim; j++){
                scalar_t grad = grad_out_values_1[j];
                auto weights_1 = weights_0[j];
                
                for(int64_t k=0; k<active_in_dim; k++)
                    grad_in_values_1[k] += weights_1[active_in_indices_1[k]]*grad;
            }
        }
    });
    at::parallel_for(0, out_dim, 0, [&](int64_t start, int64_t end){
        for (int64_t j = start; j < end; j++){
            auto weights_1 = weights_0[j];
            auto grad_weights_1 = grad_weights_0[j];
            scalar_t &grad_bias_1 = grad_bias_0[j];

            for(int64_t i=0; i<batch_size; i++){
                scalar_t grad = grad_out_values_0[i][j];
                auto in_values_1 = in_values_0[i];
                auto active_in_indices_1 = active_in_indices_0[i];

                for(int64_t k=0; k<active_in_dim; k++)
                    grad_weights_1[active_in_indices_1[k]] += in_values_1[k]*grad;
                grad_bias_1 += grad;
            }
        }
    });
}

std::vector<torch::Tensor> sido_backward(
        const torch::Tensor& grad_out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    auto grad_in_values = torch::zeros_like(in_values);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_bias = torch::zeros_like(bias);

    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "sido_backward", [&] {
            sido_backward_kernel<scalar_t>(grad_in_values, grad_weights, grad_bias,
                                          grad_out_values, in_values, active_in_indices, weights);
        });
    return {grad_in_values, grad_weights, grad_bias};
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void siso_kernel(
        torch::Tensor& out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    int64_t batch_size = active_out_indices.size(0);
    int64_t active_out_dim = active_out_indices.size(1);
    int64_t active_in_dim = active_in_indices.size(1);

    auto out_values_0 = out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto active_out_indices_0 = active_out_indices.accessor<int64_t,2>();
    auto active_in_indices_0 = active_in_indices.accessor<int64_t,2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();
    auto bias_0 = bias.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
        for (int64_t i = start; i < end; i++){
            auto out_values_1 = out_values_0[i];
            auto in_values_1 = in_values_0[i];
            auto active_out_indices_1 = active_out_indices_0[i];
            auto active_in_indices_1 = active_in_indices_0[i];
            
            for(int64_t j=0; j<active_out_dim; j++){
                scalar_t &res = out_values_1[j];
                int64_t out_index = active_out_indices_1[j];
                auto weights_1 = weights_0[out_index];

                res = bias_0[out_index];
                for(int64_t k=0; k<active_in_dim; k++)
                    res += in_values_1[k]*weights_1[active_in_indices_1[k]];
            }
        }
    });
}

torch::Tensor siso(
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){
    // sparse input sparse output
    
    auto out_values = torch::empty(active_out_indices.sizes(), in_values.options());
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "siso", [&] {
            siso_kernel<scalar_t>(out_values, in_values, active_out_indices, active_in_indices, weights, bias);
        });
    return out_values;
}

template <typename scalar_t>
void siso_backward_kernel(
        torch::Tensor& grad_in_values,
        torch::Tensor& grad_weights,
        torch::Tensor& grad_bias,
        const torch::Tensor& grad_out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights){

    int64_t batch_size = active_out_indices.size(0);
    int64_t active_out_dim = active_out_indices.size(1);
    int64_t active_in_dim = active_in_indices.size(1);

    auto grad_in_values_0 = grad_in_values.accessor<scalar_t, 2>();
    auto grad_weights_0 = grad_weights.accessor<scalar_t, 2>();
    auto grad_bias_0 = grad_bias.accessor<scalar_t, 1>();
    auto grad_out_values_0 = grad_out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto active_out_indices_0 = active_out_indices.accessor<int64_t,2>();
    auto active_in_indices_0 = active_in_indices.accessor<int64_t,2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();

    // The implementation currently parallelizes over the batch dimension. This is
    // similar to the implementation in the SLIDE paper. However, under the assumption
    // that no two out_indices in the same input row are the same, it might possibly be
    // better (to reduce update collisions) to parallelize over active_out_dim dimension
    // only while computing grad_weights. However, this requires that active_out_dim be
    // same for all inputs, so that work allocation for threads is balanced.
    // The implementation would be similar to version 2 of sido
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
        for (int64_t i = start; i < end; i++){
            auto grad_out_values_1 = grad_out_values_0[i];
            auto grad_in_values_1 = grad_in_values_0[i];
            auto in_values_1 = in_values_0[i];
            auto active_out_indices_1 = active_out_indices_0[i];
            auto active_in_indices_1 = active_in_indices_0[i];
            
            for(int64_t j=0; j<active_out_dim; j++){
                scalar_t grad = grad_out_values_1[j];
                int64_t out_index = active_out_indices_1[j];
                auto weights_1 = weights_0[out_index];
                auto grad_weights_1 = grad_weights_0[out_index];

                for(int64_t k=0; k<active_in_dim; k++){
                    int64_t in_index = active_in_indices_1[k];
                    grad_in_values_1[k] += weights_1[in_index]*grad;
                    grad_weights_1[in_index] += in_values_1[k]*grad;
                }
                grad_bias_0[out_index] += grad;
            }
        }
    });
}

std::vector<torch::Tensor> siso_backward(
        const torch::Tensor& grad_out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& active_out_indices,
        const torch::Tensor& active_in_indices,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    auto grad_in_values = torch::zeros_like(in_values);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_bias = torch::zeros_like(bias);

    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "siso_backward", [&] {
            siso_backward_kernel<scalar_t>(grad_in_values, grad_weights, grad_bias,
                                          grad_out_values, in_values, active_out_indices, active_in_indices, weights);
        });
    return {grad_in_values, grad_weights, grad_bias};
}
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
void dido_naive_kernel(
        torch::Tensor& out_values,
        const torch::Tensor& in_values,
        const torch::Tensor& weights,
        const torch::Tensor& bias){

    int64_t batch_size = out_values.size(0);
    int64_t out_dim = out_values.size(1);
    int64_t in_dim = weights.size(1);

    auto out_values_0 = out_values.accessor<scalar_t, 2>();
    auto in_values_0 = in_values.accessor<scalar_t, 2>();
    auto weights_0 = weights.accessor<scalar_t, 2>();
    auto bias_0 = bias.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end){
        for (int64_t i = start; i < end; i++){
            auto out_values_1 = out_values_0[i];
            auto in_values_1 = in_values_0[i];
            
            for(int64_t j=0; j<out_dim; j++){
                scalar_t &res = out_values_1[j];
                auto weights_1 = weights_0[j];

                res = bias_0[j];
                for(int64_t k=0; k<in_dim; k++)
                    res += in_values_1[k]*weights_1[k];
            }
        }
        // std::cout << "in thread " << omp_get_thread_num() << std::endl;
    });
}

torch::Tensor dido_naive(
        const torch::Tensor& in_values,
        const torch::Tensor& weights,
        const torch::Tensor& bias){
    // dense in dense out naive
    // dido_naive_backward also simply uses calls to dido_naive
    
    // std::cout<< "omp_get_max_threads() " << omp_get_max_threads() << std::endl;
    // std::cout<< "omp_get_num_thresads() " << omp_get_num_threads() << std::endl;

    auto out_values = torch::empty({in_values.size(0), weights.size(0)}, in_values.options());
    AT_DISPATCH_FLOATING_TYPES(
        in_values.scalar_type(), "dido_naive", [&] {
            dido_naive_kernel<scalar_t>(out_values, in_values, weights, bias);
        });
    return out_values;
}
///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diso_non_threaded", &diso_non_threaded, "dense in sparse out non threaded");
    m.def("diso", &diso, "dense in sparse out");
    m.def("diso_backward", &diso_backward, "dense in sparse out backward");
    m.def("sido", &sido, "sparse in dense out");
    m.def("sido_backward", &sido_backward, "sparse in dense out backward");
    m.def("siso", &siso, "sparse in sparse out");
    m.def("siso_backward", &siso_backward, "sparse in sparse out backward");
    m.def("dido_naive", &dido_naive, "dense in dense out naive");
}