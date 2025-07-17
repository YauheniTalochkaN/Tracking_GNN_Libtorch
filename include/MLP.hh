#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include "torch/torch.h"

template <typename ActivationType = torch::nn::Tanh, 
          typename EndActivationType = torch::nn::Identity>
class MLPImpl final : public torch::nn::Module 
{
public:
    MLPImpl(const int& input_size,
            const std::vector<int>& hidden_sizes,
            const int& output_size,
            const double& dropout_prob = 0.0,
            const bool& use_layer_norm = true)
    {
        if (input_size < 1)
        {
            throw std::invalid_argument("MLPImpl::MLPImpl: input_size cannot be less than one.");
        }

        if (output_size < 1)
        {
            throw std::invalid_argument("MLPImpl::MLPImpl: output_size cannot be less than one.");
        }
        
        if(hidden_sizes.empty()) 
        {
            throw std::invalid_argument("MLPImpl::MLPImpl: hidden_sizes cannot be empty.");
        }

        for(auto& it : hidden_sizes)
        {
            if (it < 1)
            {
                throw std::invalid_argument("MLPImpl::MLPImpl: All components of hidden_sizes must be greater than zero.");
            }
        }

        model = register_module("model", torch::nn::Sequential());

        model->push_back(torch::nn::Linear(input_size, hidden_sizes[0]));
        
        if (use_layer_norm) 
        {
            model->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes[0]})));
        }

        model->push_back(ActivationType());

        if (dropout_prob > 0.0) 
        {
            model->push_back(torch::nn::Dropout(dropout_prob));
        }

        for (size_t i = 1; i < hidden_sizes.size(); ++i) 
        {
            model->push_back(torch::nn::Linear(hidden_sizes[i-1], hidden_sizes[i]));

            if (use_layer_norm) 
            {
                model->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes[i]})));
            }

            model->push_back(ActivationType());

            if (dropout_prob > 0.0) 
            {
                model->push_back(torch::nn::Dropout(dropout_prob));
            }
        }

        model->push_back(torch::nn::Linear(hidden_sizes.back(), output_size));

        if constexpr(!std::is_same_v<EndActivationType, torch::nn::Identity>) 
        {
            model->push_back(EndActivationType());
        }
    }

    ~MLPImpl() override = default;

    torch::Tensor forward(torch::Tensor x) 
    {
        return model->forward(x);
    }

private:
    torch::nn::Sequential model{nullptr};
};

template <typename ActivationType = torch::nn::Tanh, typename EndActivationType = torch::nn::Identity>
class MLP : public torch::nn::ModuleHolder<MLPImpl<ActivationType, EndActivationType>> 
{
public: 
    using torch::nn::ModuleHolder<MLPImpl<ActivationType, EndActivationType>>::ModuleHolder; 
    using Impl TORCH_UNUSED_EXCEPT_CUDA = MLPImpl<ActivationType, EndActivationType>;
};