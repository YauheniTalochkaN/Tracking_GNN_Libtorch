#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <sstream>
#include <filesystem>

#include <yaml-cpp/yaml.h>

#include <torch/optim.h>
#include <torch/script.h>
#include <torch/nn/functional.h>

#include "EdgeClassificationGNN.hh"
#include "GraphDataLoader.hh"
#include "ROOTPlots.hh"

std::vector<GraphSample> load_graph_samples(const std::vector<std::string>& file_paths) 
{
    std::vector<GraphSample> samples;

    for (const auto& path : file_paths) 
    {
        if (std::filesystem::exists(path)) 
        {
            torch::jit::script::Module module = torch::jit::load(path);

            auto edge_index = module.attr("edge_index").toTensor();
            auto node_attr = module.attr("node_attr").toTensor();
            auto node_hit_id = module.attr("node_hit_id").toTensor();
            auto edge_attr = module.attr("edge_attr").toTensor();
            auto answer = module.attr("answer").toTensor();

            samples.emplace_back(edge_index, node_attr, node_hit_id, edge_attr, answer);
        }
        else
        {
            throw std::invalid_argument("load_graph_samples: File " + path + " is not found.");
        }
    }
    
    return samples;
}

torch::Tensor balanced_binary_cross_entropy(torch::Tensor pred, torch::Tensor target,  
                                            float pos_weight, float neg_weight) 
{
    auto weights = target * pos_weight + (1.0 - target) * neg_weight;

    auto loss = torch::nn::functional::binary_cross_entropy(pred, target,
                                                            torch::nn::functional::BinaryCrossEntropyFuncOptions()
                                                            .weight(weights)
                                                            .reduction(torch::kMean));
    
    return loss;
}

torch::Tensor total_loss(torch::Tensor pred, torch::Tensor target,
                         torch::Tensor final_node_attr, torch::Tensor edge_index,
                         float emb_margin, float emb_alpha, float degree_alpha,
                         float pos_weight, float neg_weight)
{
    auto bce = balanced_binary_cross_entropy(pred, target, pos_weight, neg_weight);

    auto in_indices = edge_index[0];
    auto out_indices = edge_index[1];

    auto x_i = final_node_attr.index_select(0, in_indices);
    auto x_j = final_node_attr.index_select(0, out_indices);

    auto dxij = torch::norm(x_i - x_j, 2, 1);

    auto pos_mask = target.eq(1).squeeze();
    auto neg_mask = target.eq(0).squeeze();

    auto emb_pos = torch::tensor(0.0, pred.options());
    auto emb_neg = torch::tensor(0.0, pred.options());

    if (pos_mask.any().item<bool>()) 
    {
        emb_pos = dxij.masked_select(pos_mask).pow(2).mean();
    }

    if (neg_mask.any().item<bool>()) 
    {
        emb_neg = torch::clamp(emb_margin - dxij.masked_select(neg_mask), 0.0).pow(2).mean();
    }

    auto emb = emb_alpha * (emb_pos + (neg_weight / pos_weight) * emb_neg);

    int64_t num_nodes = final_node_attr.size(0);

    auto soft_degrees = torch::zeros({num_nodes}, pred.options());
    auto true_degrees = torch::zeros({num_nodes}, target.options());

    soft_degrees.scatter_add_(0, out_indices, pred.squeeze(-1));
    true_degrees.scatter_add_(0, out_indices, target.squeeze(-1));

    auto degree = degree_alpha * torch::mse_loss(soft_degrees, true_degrees);

    return bce + emb + degree;
}

int main(int argc, char* argv[]) 
{
    auto start = std::chrono::steady_clock::now();
    
    YAML::Node config = YAML::LoadFile("../configs/training_parameters.yaml");

    if (!config) 
    {
        std::cerr << "Error: File \"training_parameters.yaml\" not found.";
        std::exit(1);
    }

    const std::string train_graph_dir  = config["train_graph_dir"].as<std::string>();
    const std::string saved_model_file = config["saved_model_file"].as<std::string>();

    const int   num_epochs     = config["num_epochs"].as<int>();
    const int   batch_size     = config["batch_size"].as<int>();
    const float test_size      = config["test_size"].as<float>();
    const float learning_rate  = config["learning_rate"].as<float>();
    const int   dataset_size   = config["dataset_size"].as<int>();
    const int   section_num    = config["section_num"].as<int>();
    const int   node_attr_size = config["node_attr_size"].as<int>();
    const int   edge_attr_size = config["edge_attr_size"].as<int>();
    const float pos_weight     = config["pos_weight"].as<float>();
    const float neg_weight     = config["neg_weight"].as<float>();
    const int   step_size      = config["step_size"].as<int>();
    const float gamma          = config["gamma"].as<float>();
    const float emb_alpha      = config["emb_alpha"].as<float>();
    const float emb_margin     = config["emb_margin"].as<float>();
    const float degree_alpha   = config["degree_alpha"].as<float>();
    const float threshold      = config["threshold"].as<float>();

    std::filesystem::path dirPath = std::filesystem::path(saved_model_file).parent_path();

    if (!std::filesystem::exists(dirPath)) 
    {
        if (!std::filesystem::create_directories(dirPath)) 
        {
            std::cerr << "Error: Fail to create the folder " << dirPath.string();
            std::exit(1);
        }
    }

    std::vector<std::string> train_file_paths;
    for (int evtid = 0; evtid < (int)((1.0f - test_size) * dataset_size); ++evtid) 
    {
        for (int section_id = 0; section_id < section_num; ++section_id) 
        {
            std::ostringstream oss;
            oss << train_graph_dir
                << "event_" << evtid
                << "_section_" << section_id
                << "_graph.pt";

            train_file_paths.push_back(oss.str());
        }
    }

    std::vector<std::string> test_file_paths;
    for (int evtid = (int)((1.0f - test_size) * dataset_size); evtid < dataset_size; ++evtid) 
    {
        for (int section_id = 0; section_id < section_num; ++section_id) 
        {
            std::ostringstream oss;
            oss << train_graph_dir
                << "event_" << evtid
                << "_section_" << section_id
                << "_graph.pt";

            test_file_paths.push_back(oss.str());
        }
    }

    std::vector<GraphSample> train_data;

    try
    {
        train_data = load_graph_samples(train_file_paths);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        std::exit(1);
    }

    auto train_dataset = GraphDataset<>(std::move(train_data));
    auto train_dataloader = GraphDataLoader<>(&train_dataset, batch_size);

    std::vector<GraphSample> test_data;

    try
    {
        test_data = load_graph_samples(test_file_paths);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        std::exit(1);
    }

    auto test_dataset = GraphDataset<>(std::move(test_data));
    auto test_dataloader = GraphDataLoader<>(&test_dataset, batch_size);

    auto model = EdgeClassificationGNN(node_attr_size, edge_attr_size);
    model->to(torch::kCUDA);

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate));
    torch::optim::StepLR scheduler(optimizer, step_size, gamma);

    int start_epoch = 1;
    if (std::filesystem::exists(saved_model_file)) 
    {
        torch::serialize::InputArchive archive;
        archive.load_from(saved_model_file);

        torch::Tensor epoch_t;
        archive.read("epoch", epoch_t);
        start_epoch = epoch_t.item<int>() + 1;

        model->load(archive);
        optimizer.load(archive);

        for (int i = 0; i < start_epoch-1; ++i) 
        {
            scheduler.step();
        }

        std::cout << "Loaded checkpoint from epoch " << start_epoch - 1 << std::endl;
    }

    float epoch_array[num_epochs - start_epoch + 1], 
          train_error[num_epochs - start_epoch + 1], 
          test_error[num_epochs - start_epoch + 1];
    std::iota(epoch_array, epoch_array + (num_epochs - start_epoch + 1), start_epoch);

    for (int epoch = start_epoch; epoch <= num_epochs; ++epoch) 
    {
        model->train();
        
        float train_epoch_loss = 0.0;

        optimizer.zero_grad();

        for (auto batch : train_dataloader) 
        {
            batch = batch.to(torch::kCUDA);

            auto edge_index   = batch.edge_index;
            auto node_attr    = batch.node_attr;
            auto edge_attr    = batch.edge_attr;
            auto answer_true  = batch.answer;

            auto [answer_pred, final_node_attr] = model->forward(edge_index, node_attr, edge_attr); 

            auto loss = total_loss(answer_pred, answer_true, 
                                   final_node_attr, edge_index,
                                   emb_margin, emb_alpha, degree_alpha,
                                   pos_weight, neg_weight);

            loss.backward();
            optimizer.step();
            train_epoch_loss += loss.item<float>();

            optimizer.zero_grad();
        }

        train_epoch_loss /= (1.0f - test_size) * dataset_size;

        train_error[epoch - start_epoch] = train_epoch_loss;

        scheduler.step();

        model->eval();
        torch::NoGradGuard no_grad;

        float test_epoch_loss = 0.0;
        float true_positive  = 0.0;
        float true_negative  = 0.0;
        float false_positive = 0.0;
        float false_negative = 0.0;

        for (auto batch : test_dataloader) 
        {
            batch = batch.to(torch::kCUDA);

            auto edge_index   = batch.edge_index;
            auto node_attr    = batch.node_attr;
            auto edge_attr    = batch.edge_attr;
            auto answer_true  = batch.answer;

            auto [answer_pred, final_node_attr] = model->forward(edge_index, node_attr, edge_attr);

            auto loss = total_loss(answer_pred, answer_true, 
                                   final_node_attr, edge_index,
                                   emb_margin, emb_alpha, degree_alpha,
                                   pos_weight, neg_weight);

            test_epoch_loss += loss.item<float>();

            auto pred_labels = (answer_pred >= threshold).to(torch::kInt32);
            auto true_labels = answer_true.to(torch::kInt32);

            true_positive  += ((pred_labels == 1) * (true_labels == 1)).sum().item<float>();
            true_negative  += ((pred_labels == 0) * (true_labels == 0)).sum().item<float>();
            false_positive += ((pred_labels == 1) * (true_labels == 0)).sum().item<float>();
            false_negative += ((pred_labels == 0) * (true_labels == 1)).sum().item<float>();
        }

        test_epoch_loss /= test_size * dataset_size;

        float accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative);
        float purity = (true_positive + false_positive) > 0 ? true_positive / (true_positive + false_positive) : 0.0;
        float efficiency = (true_positive + false_negative) > 0 ? true_positive / (true_positive + false_negative) : 0.0;

        if ((epoch % 10 == 0) || (epoch == num_epochs))
        {
            std::cout << "Epoch " << epoch << " | "
                      << "Train Loss: " << train_epoch_loss << "; "
                      << "Test Loss: " << test_epoch_loss << "; "
                      << "Accuracy: " << accuracy << "; "
                      << "Purity: " << purity << "; "
                      << "Efficiency: " << efficiency << std::endl;

            if (std::filesystem::exists(saved_model_file))
            {
                std::filesystem::remove(saved_model_file);
            }

            torch::serialize::OutputArchive archive;

            archive.write("epoch", torch::tensor(epoch));
            model->save(archive);
            optimizer.save(archive);
            archive.save_to(saved_model_file);

            std::cout << "Checkpoint saved at epoch " << epoch << std::endl;
        }

        test_error[epoch - start_epoch] = test_epoch_loss;
    }

    std::cout << "Training completed.\n";

    if(start_epoch < num_epochs)
    {
        ROOTPlots::PlotTrainingLoss(epoch_array, train_error, test_error, num_epochs - start_epoch + 1);
    }

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Total CPU/GPU time: " << elapsed.count() << " s.\n";
    
    return 0;
}