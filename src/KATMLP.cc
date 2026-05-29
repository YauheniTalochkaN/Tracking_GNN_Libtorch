#include "KATMLP.hh"

KATMLPImpl::KATMLPImpl(const int input_size, const std::vector<int>& hidden_sizes, const int output_size,
                       const std::vector<std::pair<double, double>>& xranges, const int n, const int order, 
                       const double std_w, const double dropout_prob) 
{
    if (input_size < 1)
    {
        throw std::invalid_argument("KATMLPImpl::KATMLPImpl: input_size cannot be less than one.");
    }

    if (output_size < 1)
    {
        throw std::invalid_argument("KATMLPImpl::KATMLPImpl: output_size cannot be less than one.");
    }
    
    if(hidden_sizes.empty()) 
    {
        throw std::invalid_argument("KATMLPImpl::KATMLPImpl: hidden_sizes cannot be empty.");
    }

    for(auto& it : hidden_sizes)
    {
        if (it < 1)
        {
            throw std::invalid_argument("KATMLPImpl::KATMLPImpl: All components of hidden_sizes must be greater than zero.");
        }
    }
    
    if (xranges.empty()) 
    {
        throw std::invalid_argument("KATMLPImpl::KATMLPImpl: xranges must contain at least one range.");
    }

    try
    {
        std::vector<KATLayer> layers;

        layers.push_back(KATLayer(input_size, hidden_sizes[0], xranges[0], n, order, std_w, dropout_prob));

        for (size_t i = 1; i < hidden_sizes.size(); ++i) 
        {
            std::pair<double, double> xrange = (xranges.size() >= 2) ? xranges[1] : xranges[0];
            layers.push_back(KATLayer(hidden_sizes[i-1], hidden_sizes[i], xrange, n, order, std_w, dropout_prob));
        }

        std::pair<double, double> last_xrange = (xranges.size() >= 2) ? xranges[1] : xranges[0];
        layers.push_back(KATLayer(hidden_sizes.back(), output_size, last_xrange, n, order, std_w, dropout_prob));

        model = register_module("model", torch::nn::Sequential());

        for (auto& layer : layers) 
        {
            model->push_back(layer);
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "KATMLPImpl::KATMLPImpl: " << ex.what() << '\n';
        std::exit(1);
    }
}

torch::Tensor KATMLPImpl::forward(torch::Tensor x) 
{
    return model->forward(x);
}