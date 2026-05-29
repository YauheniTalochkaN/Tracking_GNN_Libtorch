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
#include "PostProcessing.hh"
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

            samples.emplace_back(edge_index, node_attr, node_hit_id, edge_attr);
    
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
    
    YAML::Node config = YAML::LoadFile("../configs/evaluate_parameters.yaml");

    if (!config) 
    {
        std::cerr << "Error: File \"evaluate_parameters.yaml\" is not found.";
        std::exit(1);
    }

    const std::string graph_dir  = config["graph_dir"].as<std::string>();
    const std::string saved_model_file = config["saved_model_file"].as<std::string>();

    const int   dataset_size   = config["dataset_size"].as<int>();
    const int   section_num    = config["section_num"].as<int>();
    const int   node_attr_size = config["node_attr_size"].as<int>();
    const int   edge_attr_size = config["edge_attr_size"].as<int>();
    const float threshold      = config["threshold"].as<float>();

    std::vector<std::string> test_file_paths;
    for (int evtid = 0; evtid < dataset_size; ++evtid) 
    {
        for (int section_id = 0; section_id < section_num; ++section_id) 
        {
            std::ostringstream oss;
            oss << graph_dir
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

    std::vector<std::vector<std::set<int>>> results;

    for (auto& graph : test_data) 
    {
        graph = graph.to(torch::kCUDA);

        auto edge_index = graph.edge_index;
        auto node_attr  = graph.node_attr;
        auto edge_attr  = graph.edge_attr;

        auto [answer_pred, final_node_attr] = model->forward(edge_index, node_attr, edge_attr);

        graph.answer = answer_pred;

        graph = graph.to(torch::kCPU);

        results.push_back(PostProcessing::GetTracks(graph, threshold));
    }

    /* const std::string out_dir_path = "../answers";

    std::filesystem::path dirPath(out_dir_path);
    
    if(!std::filesystem::exists(dirPath)) 
    {
        if(!std::filesystem::create_directories(dirPath)) 
        {
            throw std::runtime_error("The directory " + out_dir_path + " can not be created.");
        }
    } 
    else if(!std::filesystem::is_directory(dirPath)) 
    {
        throw std::invalid_argument("The path " + out_dir_path + " exists but is not a directory.");
    }

    for(size_t evt_id = 0; evt_id < results.size(); ++evt_id)
    {
        std::filesystem::path filePath = dirPath / ("answer_" + std::to_string(evt_id) + ".csv");

        std::ofstream file(filePath, std::ios::trunc);

        file << "hit_id,track_id\n";

        auto& event = results[evt_id];

        for(size_t tr_id = 0; tr_id < event.size(); ++tr_id)
        {
            auto& track = event[tr_id];
            
            for(auto hit : track)
            {
                file << hit << "," << tr_id << "\n";
            }
        }

        file.close();
    } */

    ROOTPlots::PlotGraphSample3D(test_data.back(), threshold, false);

    std::cout << "Dataset has been processed.\n";

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Total CPU/GPU time: " << elapsed.count() << " s.\n";

    return 0;
}