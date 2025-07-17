#include <iostream>
#include <vector>
#include <string>
#include <chrono>
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

    const int   batch_size     = config["batch_size"].as<int>();
    const float test_size      = config["test_size"].as<float>();
    const int   dataset_size   = config["dataset_size"].as<int>();
    const int   section_num    = config["section_num"].as<int>();
    const int   node_attr_size = config["node_attr_size"].as<int>();
    const int   edge_attr_size = config["edge_attr_size"].as<int>();

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

    try
    {
        model->load_model(saved_model_file);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << '\n';
        std::exit(1);
    }

    model->eval();
    torch::NoGradGuard no_grad;

    const int num_points = 100;

    float threshold_array[num_points], 
          accuracy_array[num_points], 
          purity_array[num_points],
          efficiency_array[num_points];

    for (int i = 0; i < num_points; ++i)
    {
        threshold_array[i] = static_cast<float>(i) / static_cast<float>(num_points-1);
    }

    int true_positive[num_points]{0};
    int true_negative[num_points]{0};
    int false_positive[num_points]{0};
    int false_negative[num_points]{0};

    for (auto batch : test_dataloader) 
    {
        batch = batch.to(torch::kCUDA);

        auto edge_index   = batch.edge_index;
        auto node_attr    = batch.node_attr;
        auto edge_attr    = batch.edge_attr;
        auto answer_true  = batch.answer;

        auto answer_pred = model->forward(edge_index, node_attr, edge_attr);

        auto true_labels = answer_true.to(torch::kInt32);

        for (int i = 0; i < num_points; ++i)
        {            
            auto pred_labels = (answer_pred >= threshold_array[i]).to(torch::kInt32);

            true_positive[i]  += ((pred_labels == 1) * (true_labels == 1)).sum().item<int>();
            true_negative[i]  += ((pred_labels == 0) * (true_labels == 0)).sum().item<int>();
            false_positive[i] += ((pred_labels == 1) * (true_labels == 0)).sum().item<int>();
            false_negative[i] += ((pred_labels == 0) * (true_labels == 1)).sum().item<int>();
        }
    }

    for (int i = 0; i < num_points; ++i)
    {
        accuracy_array[i] = static_cast<float>(true_positive[i] + true_negative[i]) / static_cast<float>(true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i]);
        purity_array[i] = (true_positive[i] + false_positive[i]) > 0 ? static_cast<float>(true_positive[i]) / static_cast<float>(true_positive[i] + false_positive[i]) : 0.0;
        efficiency_array[i] = (true_positive[i] + false_negative[i]) > 0 ? static_cast<float>(true_positive[i]) / static_cast<float>(true_positive[i] + false_negative[i]) : 0.0;
    }

    std::cout << "Evaluation of the GNN is completed.\n";

    ROOTPlots::PlotMetrics(threshold_array, accuracy_array, purity_array, efficiency_array, num_points);

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Total CPU time: " << elapsed.count() << " s.\n";

    return 0;
}