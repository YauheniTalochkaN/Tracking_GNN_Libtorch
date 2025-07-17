#pragma once

#include <vector>
#include "KATLayer.hh"

class KATMLPImpl final : public torch::nn::Module 
{
public:
    KATMLPImpl(const int& input_size, const std::vector<int>& hidden_sizes, const int& output_size,
               const std::vector<std::pair<double, double>>& xranges = {{0, 1}, {-1, 1}}, const int& n = 10, 
               const int& order = 1, const double& std_w = 0.001, const double& dropout_prob = 0.0);
    ~KATMLPImpl() override = default;
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential model{nullptr};
};

TORCH_MODULE(KATMLP);