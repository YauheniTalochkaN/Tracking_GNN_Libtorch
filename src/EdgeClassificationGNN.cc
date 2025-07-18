#include "EdgeClassificationGNN.hh"

EdgeClassificationGNNImpl::EdgeClassificationGNNImpl(const int node_attr_size, const int edge_attr_size, const size_t n_iters_, 
                                                     const int node_hidden_size, const int edge_hidden_size, 
                                                     const std::vector<int>& node_encoder_hidden_sizes,
                                                     const std::vector<int>& edge_encoder_hidden_sizes,
                                                     const std::vector<int>& initial_edge_classifier_hidden_sizes,
                                                     const std::vector<int>& node_gatconv_hidden_sizes,
                                                     const std::vector<int>& edge_mlp_hidden_sizes,
                                                     const std::vector<int>& edge_classifier_hidden_sizes) : n_iters(n_iters_)
{
    try
    {    
        node_encoder = register_module("node_encoder", MLP<>(node_attr_size, 
                                                             node_encoder_hidden_sizes, 
                                                             node_hidden_size, 
                                                             0.1));

        edge_encoder = register_module("edge_encoder", MLP<>(2 * node_attr_size + edge_attr_size, 
                                                             edge_encoder_hidden_sizes, 
                                                             edge_hidden_size, 
                                                             0.1));

        initial_edge_classification_mlp = register_module("initial_edge_classification_mlp", 
                                                          MLP<torch::nn::Tanh, torch::nn::Sigmoid>(2 * node_hidden_size + edge_hidden_size +
                                                                                                   2 * node_attr_size + edge_attr_size, 
                                                                                                   initial_edge_classifier_hidden_sizes, 
                                                                                                   1, 
                                                                                                   0.1));

        node_gatconv = register_module("node_gatconv", GATConv<>(node_hidden_size, 
                                                                 node_gatconv_hidden_sizes, 
                                                                 node_hidden_size, 
                                                                 node_attr_size,
                                                                 edge_hidden_size,
                                                                 0.1));

        edge_mlp = register_module("edge_mlp", MLP<>(2 * node_hidden_size + edge_hidden_size +
                                                     2 * node_attr_size + edge_attr_size + 1, 
                                                     edge_mlp_hidden_sizes, 
                                                     edge_hidden_size, 
                                                     0.1));

        edge_classification_mlp = register_module("edge_classification_mlp", 
                                                  MLP<torch::nn::Tanh, torch::nn::Sigmoid>(2 * node_hidden_size + edge_hidden_size +
                                                                                           2 * node_attr_size + edge_attr_size + 1, 
                                                                                           edge_classifier_hidden_sizes, 
                                                                                           1, 
                                                                                           0.1));
    }
    catch(const std::exception& ex)
    {
        std::cerr << "EdgeClassificationGNNImpl::EdgeClassificationGNNImpl: " << ex.what() << '\n';
        std::exit(1);
    }
}

torch::Tensor EdgeClassificationGNNImpl::forward(torch::Tensor edge_index, torch::Tensor node_attr, torch::Tensor edge_attr)
{
    auto initial_node_attr = node_attr;
    auto row = edge_index[0];
    auto col = edge_index[1];

    auto initial_edge_features = torch::cat({initial_node_attr.index_select(0, row),
                                             initial_node_attr.index_select(0, col),
                                             edge_attr}, 1);

    auto edge_hidden = edge_encoder->forward(initial_edge_features);

    node_attr = node_encoder->forward(initial_node_attr);

    auto complex_features = torch::cat({node_attr.index_select(0, row),
                                        node_attr.index_select(0, col),
                                        edge_hidden,
                                        initial_node_attr.index_select(0, row),
                                        initial_node_attr.index_select(0, col),
                                        edge_attr}, 1);

    auto edge_labels = initial_edge_classification_mlp->forward(complex_features);

    for (size_t i = 0; i < n_iters; ++i)
    {
        node_attr = node_gatconv->forward(edge_index, node_attr, edge_hidden, edge_labels, initial_node_attr);

        complex_features = torch::cat({node_attr.index_select(0, row),
                                       node_attr.index_select(0, col),
                                       edge_hidden,
                                       edge_labels,
                                       initial_node_attr.index_select(0, row),
                                       initial_node_attr.index_select(0, col),
                                       edge_attr}, 1);

        edge_hidden = edge_mlp->forward(complex_features);

        complex_features = torch::cat({node_attr.index_select(0, row),
                                       node_attr.index_select(0, col),
                                       edge_hidden,
                                       edge_labels,
                                       initial_node_attr.index_select(0, row),
                                       initial_node_attr.index_select(0, col),
                                       edge_attr}, 1);

        edge_labels = edge_classification_mlp->forward(complex_features);
    }

    return edge_labels;
}

void EdgeClassificationGNNImpl::load_model(const std::string& file_name)
{
    int checkpoint_epoch = 1;
    
    if (std::filesystem::exists(file_name)) 
    {
        torch::serialize::InputArchive archive;
        archive.load_from(file_name);

        torch::Tensor epoch_t;
        archive.read("epoch", epoch_t);
        checkpoint_epoch = epoch_t.item<int>();

        this->load(archive);

        std::cout << "Loaded checkpoint from epoch " << checkpoint_epoch << std::endl;
    }
    else
    {
        throw std::invalid_argument("EdgeClassificationGNNImpl::load_model: model_checkpoint.pth file is not found.");
    }
}