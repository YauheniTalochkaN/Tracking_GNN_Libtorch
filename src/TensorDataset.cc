#include "TensorDataset.hh"

TensorDataset::TensorDataset(torch::Tensor inputs, torch::Tensor targets, torch::Device device) : 
                             inputs(inputs.to(device)), targets(targets.to(device))
{
    
}

torch::data::Example<> TensorDataset::get(size_t index)
{
    return {inputs[index], targets[index]};
}

torch::optional<size_t> TensorDataset::size() const
{
    return inputs.size(0);
}