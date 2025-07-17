#pragma once

#include "MLP.hh"

template <typename ActivationType = torch::nn::Tanh, 
          typename EndActivationType = torch::nn::Identity>
class GATConvImpl : public torch::nn::Module 
{
public:
    GATConvImpl(const int& input_node_attr_size,
                const std::vector<int>& hidden_sizes,
                const int& output_node_attr_size,
                const int& initial_node_attr_size,
                const int& edge_attr_size,
                const double& dropout_prob = 0.0,
                const bool& use_layer_norm = true)
    {
        if (input_node_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: input_node_attr_size cannot be less than one.");
        }

        if (output_node_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: output_node_attr_size cannot be less than one.");
        }

        if (initial_node_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: initial_node_attr_size cannot be less than one.");
        }
        
        if (edge_attr_size < 1)
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: edge_attr_size cannot be less than one.");
        }
        
        if(hidden_sizes.empty()) 
        {
            throw std::invalid_argument("GATConvImpl::GATConvImpl: hidden_sizes cannot be empty.");
        }

        for(auto& it : hidden_sizes)
        {
            if (it < 1)
            {
                throw std::invalid_argument("GATConvImpl::GATConvImpl: All components of hidden_sizes must be greater than zero.");
            }
        }

        mlp = register_module("mlp", MLP<ActivationType, EndActivationType>(5 * input_node_attr_size + initial_node_attr_size + 4 * edge_attr_size, 
                                                                            hidden_sizes, 
                                                                            output_node_attr_size, 
                                                                            dropout_prob, 
                                                                            use_layer_norm));
    }

    virtual ~GATConvImpl() override = default;

    virtual torch::Tensor forward(torch::Tensor edge_index, torch::Tensor node_attr,
                                  torch::Tensor edge_attr, torch::Tensor edge_weight, torch::Tensor initial_node_attr)
    {
        auto reversed_edge_index = edge_index.flip(0);

        auto one_hop_incoming = propagate(edge_index, node_attr, edge_attr, edge_weight, 1);
        auto one_hop_outgoing = propagate(reversed_edge_index, node_attr, edge_attr, edge_weight, 1);

        auto two_hop_incoming = propagate(edge_index, one_hop_incoming, edge_attr, edge_weight, 2);
        auto two_hop_outgoing = propagate(reversed_edge_index, one_hop_outgoing, edge_attr, edge_weight, 2);

        auto combined = torch::cat({initial_node_attr, node_attr, 
                                    one_hop_incoming, one_hop_outgoing, 
                                    two_hop_incoming, two_hop_outgoing}, -1);

        return mlp->forward(combined);
    }

protected:
    virtual torch::Tensor propagate(torch::Tensor edge_index, torch::Tensor node_attr, 
                                    torch::Tensor edge_attr, torch::Tensor edge_weight, int hop)
    {
        auto messages = message(edge_index, node_attr, edge_attr, edge_weight, hop);

        return aggregate(edge_index, messages, node_attr.size(0));
    }

    virtual torch::Tensor message(torch::Tensor edge_index, torch::Tensor node_attr,
                                  torch::Tensor edge_attr, torch::Tensor edge_weight, int hop)
    {
        auto source_nodes = edge_index[0];
        auto node_attr_j = node_attr.index_select(0, source_nodes);

        if(hop == 1) 
        {
            return edge_weight * torch::cat({node_attr_j, edge_attr}, -1);
        }
        else 
        {
            return edge_weight * node_attr_j;
        }
    }

    virtual torch::Tensor aggregate(torch::Tensor edge_index, torch::Tensor messages, int num_nodes)
    {
        auto target_nodes = edge_index[1];
        
        return torch::zeros({num_nodes, messages.size(1)}, messages.options()).index_add_(0, target_nodes, messages);
    }

    MLP<ActivationType, EndActivationType> mlp{nullptr};
};

template <typename ActivationType = torch::nn::Tanh, typename EndActivationType = torch::nn::Identity>
class GATConv : public torch::nn::ModuleHolder<GATConvImpl<ActivationType, EndActivationType>> 
{
public: 
    using torch::nn::ModuleHolder<GATConvImpl<ActivationType, EndActivationType>>::ModuleHolder; 
    using Impl TORCH_UNUSED_EXCEPT_CUDA = GATConvImpl<ActivationType, EndActivationType>;
};