#pragma once

#include <torch/data.h>
#include <torch/data/dataloader.h>

class TensorDataset : public torch::data::Dataset<TensorDataset>
{
public:
    TensorDataset(torch::Tensor inputs, torch::Tensor targets, torch::Device device);
    virtual ~TensorDataset() override = default;
    virtual torch::data::Example<> get(size_t index) override;
    virtual torch::optional<size_t> size() const override;

protected:
    torch::Tensor inputs, targets;
};