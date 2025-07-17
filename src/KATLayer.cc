#include "KATLayer.hh"

KATLayerImpl::KATLayerImpl(const int& input_dim_, const int& output_dim_, const std::pair<double, double>& xrange, 
                           const int& n_, const int& order_, const double& std_w, const double& dropout_prob) :
                           input_dim(input_dim_), output_dim(output_dim_), n(n_), order(order_) 
{   
    if (input_dim < 1)
    {
        throw std::invalid_argument("KATLayerImpl::KATLayerImpl: input_size cannot be less than one.");
    }

    if (output_dim < 1)
    {
        throw std::invalid_argument("KATLayerImpl::KATLayerImpl: output_size cannot be less than one.");
    }
    
    if (xrange.second <= xrange.first) 
    {
        throw std::invalid_argument("KATLayerImpl::KATLayerImpl: Invalid xrange. Upper bound must be greater than lower bound.");
    }

    double sigma_init = (xrange.second - xrange.first) / n / 3.0;
    mx_start = register_buffer("mx_start", torch::linspace(xrange.first, xrange.second, n).view({1, 1, 1, n}));

    mx_train = register_parameter("mx_train", torch::zeros({input_dim, output_dim}));
    scale = register_parameter("scale", torch::ones({input_dim, output_dim}));
    sigma = register_parameter("sigma", torch::full({input_dim, output_dim, n}, sigma_init));
    alpha = register_parameter("alpha", torch::zeros({input_dim, output_dim, n}));
    w = register_parameter("w", torch::randn({input_dim, output_dim, n}) * std_w);

    if ((dropout_prob > 0.0) && (dropout_prob < 1.0))
    {
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(dropout_prob)));
    }
}

torch::Tensor KATLayerImpl::basis(torch::Tensor z, torch::Tensor alpha, int s)
{
    return torch::exp(-torch::pow(z, 2 * s)) * 2 * torch::special::ndtr(alpha * z);
}

torch::Tensor KATLayerImpl::forward(torch::Tensor x) 
{
    auto x_expanded = x.unsqueeze(-1).unsqueeze(-1);
    auto mx_train_expanded = mx_train.unsqueeze(-1);
    auto scale_expanded = scale.unsqueeze(-1);
    
    auto z = (x_expanded - torch::abs(scale_expanded) * mx_start - mx_train_expanded) / (torch::abs(sigma) + 1e-8);
    
    torch::Tensor f = basis(z, alpha, order);
    
    if (dropout) 
    {
        f = dropout(f);
    }
    
    return torch::sum(w * f, torch::IntArrayRef{1, 3});
}
    
double KATLayerImpl::eval_func(const double& x, const int& i, const int& j) 
{
    if (i < 0 || i >= input_dim) 
    {
        throw std::out_of_range("KATLayerImpl::eval_func: Input index out of range");
    }

    if (j < 0 || j >= output_dim)
    {
        throw std::out_of_range("KATLayerImpl::eval_func: Output index out of range");
    }
    
    torch::NoGradGuard no_grad;
    auto device = mx_train.device();
    auto x_tensor = torch::full({n}, x, torch::TensorOptions().device(device));
    
    auto mx_start_flat = mx_start.view({n});
    auto mx_train_ij = mx_train.index({i, j}).expand({n});
    auto scale_ij = scale.index({i, j}).expand({n});
    auto sigma_ij = sigma.index({i, j, torch::indexing::Slice()});
    auto alpha_ij = alpha.index({i, j, torch::indexing::Slice()});
    auto w_ij = w.index({i, j, torch::indexing::Slice()});
    
    auto z = (x_tensor - torch::abs(scale_ij) * mx_start_flat - mx_train_ij) / (torch::abs(sigma_ij) + 1e-8);
    
    torch::Tensor f = basis(z, alpha_ij, order);
    
    return torch::sum(w_ij * f).item<double>();
}