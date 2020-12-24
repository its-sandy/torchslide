#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

template <typename scalar_t>
// torch::Tensor cross_entropy_loss_from_log_softmax_kernel(
void cross_entropy_loss_from_log_softmax_kernel(
        const torch::Tensor& log_softmax_vals,
        const torch::Tensor& label_counts) {
    
    int32_t batch_size = log_softmax_vals.size(0);
    // int32_t sample_size = log_softmax_vals.size(1);

    auto log_softmax_vals_0 = log_softmax_vals.accessor<scalar_t, 2>(); // Batch x sample_size
    auto label_counts_0 = label_counts.accessor<int32_t, 1>(); // Batch

    std::vector<scalar_t> sample_losses(batch_size, 0);

    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {
            auto log_softmax_vals_1 = log_softmax_vals_0[i];
            int32_t label_count = label_counts_0[i];
            scalar_t &sample_loss = sample_losses[i];

            for(int32_t j=0; j<label_count; j++) {
                sample_loss -= log_softmax_vals_1[j];
            }
            sample_loss /= label_count;
        }
    });

    scalar_t ce_loss = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        ce_loss += sample_losses[i];
    }
    ce_loss /= batch_size;
    // std::cout<<ce_loss<<std::endl;
    // return torch::tensor(ce_loss);
}

// torch::Tensor cross_entropy_loss_from_log_softmax(
void cross_entropy_loss_from_log_softmax(
        const torch::Tensor& log_softmax_vals,
        const torch::Tensor& label_counts) {
    
    AT_DISPATCH_FLOATING_TYPES(
        log_softmax_vals.scalar_type(), "cross_entropy_loss_from_log_softmax", [&] {
            // return cross_entropy_loss_from_log_softmax_kernel<scalar_t>(log_softmax_vals, label_counts);
            cross_entropy_loss_from_log_softmax_kernel<scalar_t>(log_softmax_vals, label_counts);
        });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_entropy_loss_from_log_softmax", &cross_entropy_loss_from_log_softmax, "cross_entropy_loss_from_log_softmax");
}