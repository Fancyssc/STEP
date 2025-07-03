#include <torch/extension.h>
#include <vector>

__global__ void initialize_distance(float* distance, int B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N) {
        distance[idx] = 1e10;
    }
}

__global__ void update_selected_mask(int64_t* centroids, bool* selected_mask, int B, int N, int npoint, int round) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B) {
        int farthest = centroids[idx * npoint + round];
        selected_mask[idx * N + farthest] = true;
    }
}

__global__ void compute_distance(const float* xyz, float* distance, const int64_t* centroids, const bool* selected_mask, int B, int N, int round, int npoint) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N) {
        int b_idx = idx / N;
        int n_idx = idx % N;

        if (!selected_mask[b_idx * N + n_idx]) {
            float dist = 0.0;
            for (int i = 0; i < 3; i++) {
                float diff = xyz[b_idx * N * 3 + n_idx * 3 + i] - xyz[b_idx * N * 3 + centroids[b_idx * npoint + round] * 3 + i];
                dist += diff * diff;
            }
            distance[idx] = fminf(distance[idx], dist);
        } else {
            distance[idx] = -1e10;
        }
    }
}

__global__ void select_farthest(float* distance, int64_t* farthest, int B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B) {
        float max_dist = -1e10;
        int64_t max_idx = 0;
        for (int i = 0; i < N; i++) {
            float dist = distance[idx * N + i];
            if (dist > max_dist) {
                max_dist = dist;
                max_idx = i;
            }
        }
        farthest[idx] = max_idx;
    }
}

std::vector<torch::Tensor> farthest_point_sample_cuda(torch::Tensor xyz, int npoint) {
    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int C = xyz.size(2);

    // Initialize output tensors
    auto centroids = torch::zeros({B, npoint}, torch::dtype(torch::kInt64).device(xyz.device()));  // Changed to int64
    auto distance = torch::ones({B, N}, torch::dtype(torch::kFloat32).device(xyz.device())) * 1e10;
    auto selected_mask = torch::zeros({B, N}, torch::dtype(torch::kBool).device(xyz.device()));

    // Random initialization of farthest points
    auto farthest = torch::randint(0, N, {B}, torch::dtype(torch::kInt64).device(xyz.device()));  // Changed to int64

    // Kernel launches
    int threads = 256;
    int blocks = (B * N + threads - 1) / threads;

    initialize_distance<<<blocks, threads>>>(distance.data_ptr<float>(), B, N);

    for (int i = 0; i < npoint; i++) {
        // Update centroids and selected_mask
        centroids.index_put_({torch::arange(B, torch::dtype(torch::kInt64).device(xyz.device())), i}, farthest);
        update_selected_mask<<<(B + threads - 1) / threads, threads>>>(centroids.data_ptr<int64_t>(), selected_mask.data_ptr<bool>(), B, N, npoint, i);

        // Compute distances
        compute_distance<<<blocks, threads>>>(
            xyz.data_ptr<float>(),
            distance.data_ptr<float>(),
            centroids.data_ptr<int64_t>(),
            selected_mask.data_ptr<bool>(),
            B, N, i, npoint
        );

        // Select farthest point
        select_farthest<<<(B + threads - 1) / threads, threads>>>(distance.data_ptr<float>(), farthest.data_ptr<int64_t>(), B, N);
    }

    return {centroids};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sample_cuda", &farthest_point_sample_cuda, "Farthest Point Sampling CUDA");
}

