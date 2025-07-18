#pragma once

#include <filesystem>

#include "MLP.hh"
#include "GATConv.hh"

class EdgeClassificationGNNImpl final : public torch::nn::Module 
{
public:
    EdgeClassificationGNNImpl(const int node_attr_size, const int edge_attr_size, const size_t n_iters_ = 6, 
                              const int node_hidden_size = 32, const int edge_hidden_size = 32, 
                              const std::vector<int>& node_encoder_hidden_sizes = std::vector<int>{64, 64},
                              const std::vector<int>& edge_encoder_hidden_sizes = std::vector<int>{64, 64},
                              const std::vector<int>& initial_edge_classifier_hidden_sizes = std::vector<int>{64, 64},
                              const std::vector<int>& node_gatconv_hidden_sizes = std::vector<int>{64, 64},
                              const std::vector<int>& edge_mlp_hidden_sizes = std::vector<int>{64, 64},
                              const std::vector<int>& edge_classifier_hidden_sizes = std::vector<int>{64, 64});
    ~EdgeClassificationGNNImpl() override = default;
    torch::Tensor forward(torch::Tensor edge_index, torch::Tensor node_attr, torch::Tensor edge_attr);
    void load_model(const std::string& file_name);

private:
    size_t n_iters;
    MLP<> node_encoder{nullptr}, edge_encoder{nullptr}, edge_mlp{nullptr};
    MLP<torch::nn::Tanh, torch::nn::Sigmoid> initial_edge_classification_mlp{nullptr},
                                             edge_classification_mlp{nullptr};
    GATConv<> node_gatconv{nullptr};
};

TORCH_MODULE(EdgeClassificationGNN);