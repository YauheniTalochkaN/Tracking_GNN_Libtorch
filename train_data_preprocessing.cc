#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

#include "PreProcessing.hh"

#include "ROOTPlots.hh"

void SelectHitsForTrainingFromCSV(const std::string& hit_file_name, 
                                  const std::string& track_file_name, 
                                  const std::string& truth_file_name,
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
            int hit_id = std::stoi(tokens[0]);
            float x = std::stof(tokens[1]);
            float y = std::stof(tokens[2]);
            float z = std::stof(tokens[3]);
            int sector_id = std::stoi(tokens[4]);
            int row_id = std::stoi(tokens[5]);
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

    std::map<int, int> hit_to_track;
    std::map<int, float> track_to_pt;
    
    std::ifstream truth_file(truth_file_name);

    if (!truth_file.is_open()) 
    {
        throw std::runtime_error("SelectHitsForTraining: Cannot open truth file: " + truth_file_name);
    }
    
    first_line = true;

    while (std::getline(truth_file, line)) 
    {
        if (first_line) 
        {
            first_line = false;
            continue;
        }
        
        auto tokens = ParseCSVLine(line);

        if (tokens.size() >= 2) 
        {
            int hit_id = std::stoi(tokens[0]);
            int track_id = std::stoi(tokens[1]);

            hit_to_track[hit_id] = track_id;
        }
    }

    truth_file.close();
    
    std::ifstream tracks_file(track_file_name);
    if (!tracks_file.is_open()) 
    {
        throw std::runtime_error("SelectHitsForTraining: Cannot open track file: " + track_file_name);
    }
    
    first_line = true;

    while (std::getline(tracks_file, line)) 
    {
        if (first_line) 
        {
            first_line = false;
            continue;
        }
        
        auto tokens = ParseCSVLine(line);

        if (tokens.size() >= 3) 
        {
            int track_id = std::stoi(tokens[0]);
            float pt = std::stof(tokens[2]);

            track_to_pt[track_id] = pt;
        }
    }

    tracks_file.close();
    
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
        
        auto track_it = hit_to_track.find(hit.hit_id);

        if (track_it != hit_to_track.end()) 
        {
            hit.track_id = track_it->second;
            
            auto pt_it = track_to_pt.find(hit.track_id);
            
            if (pt_it != track_to_pt.end()) 
            {
                hit.pt = pt_it->second;
            } 
            else 
            {
                hit.pt = 0.0;
            }
        } 
        else 
        {
            hit.track_id = -1;
            hit.pt = 0.0;
        }
        
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
            std::string truth_file_name = params.input_dir + "event_" + std::to_string(evtid) + "_truth.csv";
            std::string track_file_name = params.input_dir + "event_" + std::to_string(evtid) + "_tracks.csv";

            std::vector<PreProcessing::Hit> hits;

            SelectHitsForTrainingFromCSV(hit_file_name, track_file_name, truth_file_name, hits);

            std::vector<GraphSample> graphs;
            
            PreProcessing::ProcessEvent(hits, params, graphs, true);
            
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