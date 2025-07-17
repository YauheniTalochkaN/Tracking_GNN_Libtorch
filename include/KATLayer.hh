#pragma once

#include <iostream>
#include <stdexcept>
#include "torch/torch.h"

class KATLayerImpl : public torch::nn::Module 
{
public:
    KATLayerImpl(const int& input_dim_, const int& output_dim_, const std::pair<double, double>& xrange, 
                 const int& n_ = 10, const int& order_ = 1, const double& std_w = 0.001, const double& dropout_prob = 0.0);
    virtual ~KATLayerImpl() override = default;
    virtual torch::Tensor basis(torch::Tensor z, torch::Tensor alpha, int s);
    virtual torch::Tensor forward(torch::Tensor x);
    virtual double eval_func(const double& x, const int& i, const int& j);

protected:
    int input_dim, output_dim, n, order;
    torch::nn::Dropout dropout{nullptr};
    torch::Tensor mx_start, mx_train, scale, sigma, alpha, w;
};

TORCH_MODULE(KATLayer);