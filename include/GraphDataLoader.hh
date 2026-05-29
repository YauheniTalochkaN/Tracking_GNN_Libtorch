#pragma once

#include "GraphDataset.hh"

template <typename SampleType = GraphSample>
class GraphDataLoader
{
public:
    GraphDataLoader(const GraphDataset<SampleType>* dataset_, size_t batch_size_ = 1) :
                    dataset(dataset_), batch_size(batch_size_)
    {
        if (!dataset)
        {
            throw std::invalid_argument("GraphDataLoader::GraphDataLoader: dataset pointer is null.");
        }

        if (batch_size == 0)
        {
            throw std::invalid_argument("GraphDataLoader::GraphDataLoader: batch_size cannot be zero.");
        }
        
        num_batches = (dataset->size().value() + batch_size - 1) / batch_size;
    }

    virtual ~GraphDataLoader() = default;

    class BatchIterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = SampleType;
        using pointer = SampleType*;
        using reference = SampleType&;
        using iterator_category = std::input_iterator_tag;

        BatchIterator(const GraphDataset<SampleType>* dataset, size_t batch_size, size_t idx) :
                      dataset_(dataset), batch_size_(batch_size), idx_(idx) {}

        value_type operator*() const
        {
            size_t start = idx_ * batch_size_;
            size_t end = std::min(start + batch_size_, dataset_->size().value());
            std::vector<SampleType> batch_samples(dataset_->get_samples().begin() + start,
                                                  dataset_->get_samples().begin() + end);

            return SampleType::batch(batch_samples);
        }

        BatchIterator& operator++() { ++idx_; return *this; }
        BatchIterator operator++(int) { BatchIterator tmp = *this; ++(*this); return tmp; }
        bool operator==(const BatchIterator& other) const { return idx_ == other.idx_; }
        bool operator!=(const BatchIterator& other) const { return !(*this == other); }

    private:
        const GraphDataset<SampleType>* dataset_;
        size_t batch_size_;
        size_t idx_;
    };

    virtual BatchIterator begin() const { return BatchIterator(dataset, batch_size, 0); }
    virtual BatchIterator end() const { return BatchIterator(dataset, batch_size, num_batches); }

    virtual size_t batch_count() const { return num_batches; }
    virtual size_t batch_size_value() const { return batch_size; }

protected:
    const GraphDataset<SampleType>* dataset;
    size_t batch_size;
    size_t num_batches;
};