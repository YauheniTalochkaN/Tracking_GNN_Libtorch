#pragma once 

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "GraphDataset.hh"

namespace PreProcessing
{
    struct PreprocessingParams 
    {
        std::string input_dir;
        std::string output_dir;
        size_t dataset_size;
        float dphi_max;
        float z0_max;
        float chi_max;
        float d_min;
        float d_max;
        float pt_min;
        size_t n_phi_sections;
        size_t n_eta_sections;
        float eta_min;
        float eta_max;
        size_t num_rows;
        size_t num_sectors;
        float rmax;
        float zmax;
    };

    struct Hit 
    {
        int hit_id;
        float z;
        float r;
        float phi;
        int row_id;
        int sector_id;
        int track_id;
        float pt;
        int id;
    };

    struct Edge 
    {
        int index_1, index_2;
    };

    PreprocessingParams LoadConfig(const std::string& config_path);
    float CalcDphi(float phi1, float phi2);
    float CalcEta(float r, float z);
    void SplitDetectorSections(const std::vector<Hit>& hits,
                               const std::vector<float>& phi_edges,
                               const std::vector<float>& eta_edges,
                               std::vector<std::vector<Hit>>& hits_sections);
    void SelectSegments(const std::vector<Hit*>& hits1,
                        const std::vector<Hit*>& hits2,
                        float dphi_max, float z0_max,
                        float chi_max, float d_min, float d_max,
                        std::vector<Edge>& segments);
    torch::Tensor GetEdgeFeatures(const std::vector<float>& in_node,
                                  const std::vector<float>& out_node, 
                                  torch::ScalarType Dtype = torch::kFloat32);
    void ProcessEvent(const std::vector<Hit>& hits, 
                      const PreprocessingParams& params, 
                      std::vector<GraphSample>& graphs,
                      bool train = false,
                      torch::ScalarType Dtype = torch::kFloat32, torch::ScalarType Itype = torch::kInt32);
    void SaveGraphSample(const GraphSample& sample, const std::string& filename);
}