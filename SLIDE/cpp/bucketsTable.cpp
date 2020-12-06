#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <chrono>
#include <random>
#include <set>
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

    at::parallel_for(0, num_nodes, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {
            // For FIFO filling, parallelizing over num_nodes in sorted order gives
            // some inherent bias to nodes of larger index. They have a better chance
            // of being in the fixed size bucket. This effect is decreased as number
            // of threads is increased. This can also be avoided using randperm_nodes.
            // Find incremental latency cause of this
            auto hash_indices_1 = hash_indices_0[i];

            for(int32_t l = 0; l < L; l++) {
                int32_t bucket_index = hash_indices_1[l];
                int32_t &bucket_count = bucket_counts_0[l][bucket_index];
                buckets_0[l][bucket_index][bucket_count % bucket_size] = i;
                // below works if bucket_size is power of 2. Check latency improvement
                // buckets_0[l][bucket_index][bucket_count & (bucket_size-1)] = i;
                bucket_count++;
            }
        }
    });
}

void fill_buckets_FIFO_2(
        const torch::Tensor& hash_indices,
        torch::Tensor& buckets,
        torch::Tensor& bucket_counts,
        const torch::Tensor& randperm_nodes) {
    
    int32_t num_nodes = hash_indices.size(0);
    int32_t L = buckets.size(0);
    // int32_t num_buckets = buckets.size(1);
    int32_t bucket_size = buckets.size(2);

    auto buckets_0 = buckets.accessor<int32_t, 3>(); // L x num_buckets x bucket_size
    auto bucket_counts_0 = bucket_counts.accessor<int32_t, 2>(); // L x num_buckets
    auto hash_indices_0 = hash_indices.accessor<int32_t, 2>(); // num_nodes x L
    auto randperm_nodes_0 = randperm_nodes.accessor<int32_t, 1>(); // num_nodes

    at::parallel_for(0, num_nodes, 0, [&](int32_t start, int32_t end) {
        for (int32_t ii = start; ii < end; ii++) {
            int32_t i = randperm_nodes_0[ii];
            auto hash_indices_1 = hash_indices_0[i];

            for(int32_t l = 0; l < L; l++) {
                int32_t bucket_index = hash_indices_1[l];
                int32_t &bucket_count = bucket_counts_0[l][bucket_index];
                buckets_0[l][bucket_index][bucket_count % bucket_size] = i;
                // below works if bucket_size is power of 2. Check latency improvement
                // buckets_0[l][bucket_index][bucket_count & (bucket_size-1)] = i;
                bucket_count++;
            }
        }
    });
}

void fill_buckets_FIFO_3(
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

    // assumes bucket_size is power of 2
    bucket_size--;
    at::parallel_for(0, num_nodes, 0, [&](int32_t start, int32_t end) {
        for (int32_t i = start; i < end; i++) {
            auto hash_indices_1 = hash_indices_0[i];

            for(int32_t l = 0; l < L; l++) {
                int32_t bucket_index = hash_indices_1[l];
                int32_t &bucket_count = bucket_counts_0[l][bucket_index];
                buckets_0[l][bucket_index][bucket_count & bucket_size] = i;
                bucket_count++;
            }
        }
    });
}

void fill_buckets_FIFO_4(
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

    at::parallel_for(0, num_nodes * L, 0, [&](int32_t start, int32_t end) {
        for (int32_t ii = start; ii < end; ii++) {
            // this tries to decrease collisions/missed updates to buckets due to race conditions
            // works best when L is a multiple of num_threads
            int32_t l = ii/num_nodes;
            int32_t i = ii%num_nodes;

            int32_t bucket_index = hash_indices_0[i][l];
            int32_t &bucket_count = bucket_counts_0[l][bucket_index];
            buckets_0[l][bucket_index][bucket_count % bucket_size] = i;
            bucket_count++;
        }
    });
}

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

    at::parallel_for(0, num_nodes, 0, [&](int32_t start, int32_t end) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + start;
        std::minstd_rand prng(seed);

        for (int32_t i = start; i < end; i++) {
            auto hash_indices_1 = hash_indices_0[i];

            for(int32_t l = 0; l < L; l++) {
                int32_t bucket_index = hash_indices_1[l];
                int32_t &bucket_count = bucket_counts_0[l][bucket_index];

                if(bucket_count < bucket_size) {
                    buckets_0[l][bucket_index][bucket_count] = i;
                    bucket_count++;
                }
                else {
                    bucket_count++;
                    int32_t ind = prng() % bucket_count;
                    if(ind < bucket_size)
                        buckets_0[l][bucket_index][ind] = i;
                }
            }
        }
    });
}

void fill_buckets_reservoir_sampling_2(
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

    at::parallel_for(0, num_nodes * L, 0, [&](int32_t start, int32_t end) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + start;
        std::minstd_rand prng(seed);

        for (int32_t ii = start; ii < end; ii++) {
            // this tries to decrease collisions/missed updates to buckets due to race conditions
            int32_t l = ii/num_nodes;
            int32_t i = ii%num_nodes;
            
            int32_t bucket_index = hash_indices_0[i][l];
            int32_t &bucket_count = bucket_counts_0[l][bucket_index];

            if(bucket_count < bucket_size) {
                buckets_0[l][bucket_index][bucket_count] = i;
                bucket_count++;
            }
            else {
                bucket_count++;
                int32_t ind = prng() % bucket_count;
                if(ind < bucket_size)
                    buckets_0[l][bucket_index][ind] = i;
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

            // compare latency of set vs unordered_set
            std::unordered_set<int32_t> selected_nodes;
            for(int32_t j = 0; j < cur_count; j++)
                selected_nodes.insert(sampled_nodes_1[j]);

            // iterate over random permutation of hash tables to avoid biases
            // find additional cost of generating permutation and indexing twice
            // maybe instead of randperm, just start at random lth table and loop modulo
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
                int32_t start = rand() % num_nodes;
                for(int32_t j = start; (j < num_nodes) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
                for(int32_t j = 0; (j < start) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
            }
        }
    });
}

void sample_nodes_vanilla_2(
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

            std::set<int32_t> selected_nodes;
            for(int32_t j = 0; j < cur_count; j++)
                selected_nodes.insert(sampled_nodes_1[j]);

            // iterate over random permutation of hash tables to avoid biases
            // find additional cost of generating permutation and indexing twice
            // maybe instead of randperm, just start at random lth table and loop modulo
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
                int32_t start = rand() % num_nodes;
                for(int32_t j = start; (j < num_nodes) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
                for(int32_t j = 0; (j < start) && (cur_count < sample_size); j++) {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                }
            }
        }
    });
}

void sample_nodes_vanilla_3(
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

            // compare latency of set vs unordered_set
            std::unordered_set<int32_t> selected_nodes;
            for(int32_t j = 0; j < cur_count; j++)
                selected_nodes.insert(sampled_nodes_1[j]);

            // iterate over random permutation of hash tables to avoid biases
            // find additional cost of generating permutation and indexing twice
            // maybe instead of randperm, just start at random lth table and loop modulo
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

                // is rand() threadsafe ?
                int32_t start = rand() % num_nodes;
                int32_t j = start;
                do {
                    int32_t node = randperm_nodes_0[j];
                    if(selected_nodes.find(node) == selected_nodes.end()) {
                        selected_nodes.insert(node);
                        sampled_nodes_1[cur_count] = node;
                        cur_count++;
                    }
                    j = (j+1)%num_nodes;
                } while((j != start) && (cur_count < sample_size));
            }
        }
    });
}
///////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_buckets_FIFO", &fill_buckets_FIFO, "fill_buckets_FIFO");
    m.def("fill_buckets_FIFO_2", &fill_buckets_FIFO_2, "fill_buckets_FIFO_2");
    m.def("fill_buckets_FIFO_3", &fill_buckets_FIFO_3, "fill_buckets_FIFO_3");
    m.def("fill_buckets_FIFO_4", &fill_buckets_FIFO_4, "fill_buckets_FIFO_4");
    m.def("fill_buckets_reservoir_sampling", &fill_buckets_reservoir_sampling, "fill_buckets_reservoir_sampling");
    m.def("fill_buckets_reservoir_sampling_2", &fill_buckets_reservoir_sampling_2, "fill_buckets_reservoir_sampling_2");
    m.def("sample_nodes_vanilla", &sample_nodes_vanilla, "sample_nodes_vanilla");
    m.def("sample_nodes_vanilla_2", &sample_nodes_vanilla_2, "sample_nodes_vanilla_3");
    m.def("sample_nodes_vanilla_3", &sample_nodes_vanilla_3, "sample_nodes_vanilla_3");
}