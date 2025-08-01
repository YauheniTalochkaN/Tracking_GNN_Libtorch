#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

#include "PreProcessing.hh"

#include "ROOTPlots.hh"

void SelectHitsFromCSV(const std::string& hit_file_name, 
                       std::vector<PreProcessing::Hit>& hits) 
{    
    hits.clear();
    
    auto ParseCSVLine = [](const std::string& line) -> std::vector<std::string> 
    {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, ',')) 
        {
            tokens.push_back(token);
        }
        
        return tokens;
    };
    
    std::vector<int> hit_id_vec, row_id_vec, sector_id_vec;
    std::vector<float> z_vec, r_vec, phi_vec;
    
    std::ifstream hits_file(hit_file_name);
    if (!hits_file.is_open()) 
    {
        throw std::runtime_error("SelectHitsForTraining: Cannot open hit file: " + hit_file_name);
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(hits_file, line)) 
    {
        if (first_line) 
        {
            first_line = false;
            continue;
        }
        
        auto tokens = ParseCSVLine(line);

        if (tokens.size() >= 6) 
        {
            int hit_id = std::stoll(tokens[0]);
            float x = std::stod(tokens[1]);
            float y = std::stod(tokens[2]);
            float z = std::stod(tokens[3]);
            int sector_id = std::stoll(tokens[4]);
            int row_id = std::stoll(tokens[5]);
            float r = std::sqrt(x * x + y * y);
            float phi = std::atan2(y, x);
            
            hit_id_vec.push_back(hit_id);
            z_vec.push_back(z);
            r_vec.push_back(r);
            phi_vec.push_back(phi);
            row_id_vec.push_back(row_id);
            sector_id_vec.push_back(sector_id);
        }
    }

    hits_file.close();
    
    size_t n_hits = hit_id_vec.size();
    hits.reserve(n_hits);
    
    for (size_t i = 0; i < n_hits; ++i) 
    {
        PreProcessing::Hit hit;
        hit.hit_id = hit_id_vec[i];
        hit.z = z_vec[i];
        hit.r = r_vec[i];
        hit.phi = phi_vec[i];
        hit.row_id = row_id_vec[i];
        hit.sector_id = sector_id_vec[i];
            
        hits.push_back(hit);
    }
}

int main(int argc, char* argv[]) 
{
    auto start = std::chrono::steady_clock::now();
    
    PreProcessing::PreprocessingParams params;

    try
    {
        params = PreProcessing::LoadConfig("../configs/preprocessing_parameters.yaml");
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error when reading the config file: " << ex.what() << '\n';
        std::exit(1);
    }
    
    std::cout << "Starting processing of " << params.dataset_size << " files...\n";
    
    omp_set_num_threads(20);
    
    #pragma omp parallel for
    for (size_t evtid = 0; evtid < params.dataset_size; ++evtid) 
    {
        try 
        {
            std::string hit_file_name = params.input_dir + "event_" + std::to_string(evtid) + "_hits.csv";

            std::vector<PreProcessing::Hit> hits;

            SelectHitsFromCSV(hit_file_name, hits);

            std::vector<GraphSample> graphs;
            
            PreProcessing::ProcessEvent(hits, params, graphs);
            
            for (size_t sectionid = 0; sectionid < graphs.size(); ++sectionid) 
            {
                std::string output_filename = params.output_dir + "/event_" + std::to_string(evtid) 
                                                                + "_section_" + std::to_string(sectionid) + "_graph.pt";

                PreProcessing::SaveGraphSample(graphs[sectionid], output_filename);
            }
        } 
        catch (const std::exception& ex) 
        {
            std::cerr << "Error processing event " << evtid << ": " << ex.what() << std::endl;
            std::exit(1);
        }
    }
    
    std::cout << "Processing completed.\n";

    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Total CPU/GPU time: " << elapsed.count() << " s.\n";

    return 0;
}