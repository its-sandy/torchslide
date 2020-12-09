#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <chrono>
#include <random>
#include <unordered_set>
#include <vector>


void fill_buckets_FIFO(
        const torch::Tensor& hash_indices,
        torch::Tensor& buckets,
        torch::Tensor& bucket_counts) {
    
    int32_t num_nodes = hash_indices.size(0);
    int32_t L = buckets.size(0);
    // int32_t num_buckets = buckets.size(1);
    int32_t bucket_size = buckets.size(2);

    auto buckets_0 = buckets.accessor<int32_t, 3>(); // L x num_buckets x bucket_size
    auto bucket_counts_0 = bucket_counts.accessor<int32_t, 2>(); // L x num_buckets
    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // num_nodes x L

    at::parallel_for(0, L, 0, [&](int32_t start, int32_t end) {
        for (int32_t l = start; l < end; l++) {
            auto buckets_1 = buckets_0[l];
            auto bucket_counts_1 = bucket_counts_0[l];

            // For FIFO filling, going over num_nodes in sorted order gives
            // inherent bias to nodes of larger index. They have a better chance
            // of being in the fixed size bucket. This can be avoided using the
            // randperm_nodes (though this was observed to increase latency a lot
            // reservoir_sampling doesn't have this issue
            for(int32_t i = 0; i < num_nodes; i++) {
                int32_t bucket_index = hash_indices_0[i][l];
                int32_t &bucket_count = bucket_counts_1[bucket_index];
                buckets_1[bucket_index][bucket_count % bucket_size] = i;
                bucket_count++;
            }
        }
    });
}

void fill_buckets_FIFO_rand(
        const torch::Tensor& hash_indices,
        torch::Tensor& buckets,
        torch::Tensor& bucket_counts) {

    // The below implementation might help decrease the bias mentioned above.
    // But this comes with an increase in latency that might be significant for large sizes.
    
    int32_t num_nodes = hash_indices.size(0);
    int32_t L = buckets.size(0);
    // int32_t num_buckets = buckets.size(1);
    int32_t bucket_size = buckets.size(2);

    auto buckets_0 = buckets.accessor<int32_t, 3>(); // L x num_buckets x bucket_size
    auto bucket_counts_0 = bucket_counts.accessor<int32_t, 2>(); // L x num_buckets
    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // num_nodes x L

    at::parallel_for(0, L, 0, [&](int32_t start, int32_t end) {
        for (int32_t l = start; l < end; l++) {
            auto buckets_1 = buckets_0[l];
            auto bucket_counts_1 = bucket_counts_0[l];

            int32_t start_node = rand() % num_nodes;
            for(int32_t i = start_node; i < num_nodes; i++) {
                int32_t bucket_index = hash_indices_0[i][l];
                int32_t &bucket_count = bucket_counts_1[bucket_index];
                buckets_1[bucket_index][bucket_count % bucket_size] = i;
                bucket_count++;
            }
            for(int32_t i = 0; i < start_node; i++) {
                int32_t bucket_index = hash_indices_0[i][l];
                int32_t &bucket_count = bucket_counts_1[bucket_index];
                buckets_1[bucket_index][bucket_count % bucket_size] = i;
                bucket_count++;
            }
        }
    });
}
///////////////////////////////////////////////////////////////////////////////

void fill_buckets_reservoir_sampling(
        const torch::Tensor& hash_indices,
        torch::Tensor& buckets,
        torch::Tensor& bucket_counts) {
    
    int32_t num_nodes = hash_indices.size(0);
    int32_t L = buckets.size(0);
    // int32_t num_buckets = buckets.size(1);
    int32_t bucket_size = buckets.size(2);

    auto buckets_0 = buckets.accessor<int32_t, 3>(); // L x num_buckets x bucket_size
    auto bucket_counts_0 = bucket_counts.accessor<int32_t, 2>(); // L x num_buckets
    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // num_nodes x L

    at::parallel_for(0, L, 0, [&](int32_t start, int32_t end) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + start;
        std::minstd_rand prng(seed);

        for (int32_t l = start; l < end; l++) {
            auto buckets_1 = buckets_0[l];
            auto bucket_counts_1 = bucket_counts_0[l];

            for(int32_t i = 0; i < num_nodes; i++) {
                int32_t bucket_index = hash_indices_0[i][l];
                int32_t &bucket_count = bucket_counts_1[bucket_index];

                if(bucket_count < bucket_size) {
                    buckets_1[bucket_index][bucket_count] = i;
                    bucket_count++;
                }
                else {
                    bucket_count++;
                    int32_t ind = prng() % bucket_count;
                    if(ind < bucket_size)
                        buckets_1[bucket_index][ind] = i;
                }
            }
        }
    });
}
///////////////////////////////////////////////////////////////////////////////

void sample_nodes_vanilla(
        torch::Tensor& sampled_nodes,
        const torch::Tensor& presample_counts,
        const torch::Tensor& hash_indices,
        const torch::Tensor& buckets,
        const torch::Tensor& bucket_counts,
        const torch::Tensor& randperm_nodes) {
    
    int32_t batch_size = sampled_nodes.size(0);
    int32_t sample_size = sampled_nodes.size(1);
    int32_t L = buckets.size(0);
    // int32_t num_buckets = buckets.size(1);
    int32_t bucket_size = buckets.size(2);
    int32_t num_nodes = randperm_nodes.size(0);

    auto sampled_nodes_0 = sampled_nodes.accessor<int32_t, 2>(); // batch_size x sample_size
    auto presample_counts_0 = presample_counts.accessor<int32_t, 1>(); // batch_size
    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // batch_size x L
    auto buckets_0 = buckets.accessor<int32_t, 3>(); // L x num_buckets x bucket_size
    auto bucket_counts_0 = bucket_counts.accessor<int32_t, 2>(); // L x num_buckets
    auto randperm_nodes_0 = randperm_nodes.accessor<int32_t, 1>(); // num_nodes

    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {            
            auto sampled_nodes_1 = sampled_nodes_0[i];
            int32_t cur_count = presample_counts_0[i]; // to include true labels for last layer
            auto hash_indices_1 = hash_indices_0[i];

            std::unordered_set<int32_t> selected_nodes;
            for(int32_t j = 0; j < cur_count; j++)
                selected_nodes.insert(sampled_nodes_1[j]);

            // assuming the hash functions are random, looking at the buckets in order
            // from 0 might be expected to create only a small bias. sample_nodes_vanilla_rand
            // does a slightly better job at this at the expense of more latency.
            for(int32_t l = 0; (l < L) && (cur_count < sample_size); l++) {
                int32_t bucket_index = hash_indices_1[l];
                auto buckets_2 = buckets_0[l][bucket_index];
                int32_t bucket_count = std::min(bucket_counts_0[l][bucket_index], bucket_size);

                for(int32_t j = 0; (j < bucket_count) && (cur_count < sample_size); j++) {
                    int32_t node = buckets_2[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
            }

            if(cur_count < sample_size) {
                // sampling these random unselected nodes on the fly would become
                // too expensive if sample_size is close to num_nodes. So, we use
                // pre-defined random permutations.

                // is rand() threadsafe ?
                int32_t start_node = rand() % num_nodes;
                for(int32_t j = start_node; (j < num_nodes) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        // selected_nodes.insert(node); // not needed here
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
                for(int32_t j = 0; (j < start_node) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        // selected_nodes.insert(node); // not needed here
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
            }
        }
    });
}

void sample_nodes_vanilla_rand(
        torch::Tensor& sampled_nodes,
        const torch::Tensor& presample_counts,
        const torch::Tensor& hash_indices,
        const torch::Tensor& buckets,
        const torch::Tensor& bucket_counts,
        const torch::Tensor& randperm_nodes) {

    // The below implementation might help decrease the bias mentioned above.
    // But this comes with a small increase in latency.
    
    int32_t batch_size = sampled_nodes.size(0);
    int32_t sample_size = sampled_nodes.size(1);
    int32_t L = buckets.size(0);
    // int32_t num_buckets = buckets.size(1);
    int32_t bucket_size = buckets.size(2);
    int32_t num_nodes = randperm_nodes.size(0);

    auto sampled_nodes_0 = sampled_nodes.accessor<int32_t, 2>(); // batch_size x sample_size
    auto presample_counts_0 = presample_counts.accessor<int32_t, 1>(); // batch_size
    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // batch_size x L
    auto buckets_0 = buckets.accessor<int32_t, 3>(); // L x num_buckets x bucket_size
    auto bucket_counts_0 = bucket_counts.accessor<int32_t, 2>(); // L x num_buckets
    auto randperm_nodes_0 = randperm_nodes.accessor<int32_t, 1>(); // num_nodes

    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {            
            auto sampled_nodes_1 = sampled_nodes_0[i];
            int32_t cur_count = presample_counts_0[i]; // to include true labels for last layer
            auto hash_indices_1 = hash_indices_0[i];

            std::unordered_set<int32_t> selected_nodes;
            for(int32_t j = 0; j < cur_count; j++)
                selected_nodes.insert(sampled_nodes_1[j]);

            int32_t start_l = rand() % L;
            int32_t l = start_l;
            do {
                int32_t bucket_index = hash_indices_1[l];
                auto buckets_2 = buckets_0[l][bucket_index];
                int32_t bucket_count = std::min(bucket_counts_0[l][bucket_index], bucket_size);

                for(int32_t j = 0; (j < bucket_count) && (cur_count < sample_size); j++) {
                    int32_t node = buckets_2[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
                l = (l+1)%L;
            } while((l != start_l) && (cur_count < sample_size));

            if(cur_count < sample_size) {

                // is rand() threadsafe ?
                int32_t start_node = rand() % num_nodes;
                for(int32_t j = start_node; (j < num_nodes) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        // selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
                for(int32_t j = 0; (j < start_node) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        // selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
            }
        }
    });
}

///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_buckets_FIFO", &fill_buckets_FIFO, "fill_buckets_FIFO");
    m.def("fill_buckets_FIFO_rand", &fill_buckets_FIFO_rand, "fill_buckets_FIFO_rand");
    m.def("fill_buckets_reservoir_sampling", &fill_buckets_reservoir_sampling, "fill_buckets_reservoir_sampling");
    m.def("sample_nodes_vanilla", &sample_nodes_vanilla, "sample_nodes_vanilla");
    m.def("sample_nodes_vanilla_rand", &sample_nodes_vanilla_rand, "sample_nodes_vanilla_rand");
}