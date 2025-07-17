#include "PreProcessing.hh"

PreProcessing::PreprocessingParams PreProcessing::LoadConfig(const std::string& config_path) 
{
    YAML::Node config = YAML::LoadFile(config_path);

    if (!config) 
    {
        throw std::runtime_error("LoadConfig: File " + config_path + " not found.");
    }
    
    PreprocessingParams params;

    params.input_dir = config["input_dir"].as<std::string>();
    params.output_dir = config["output_dir"].as<std::string>();
    params.dataset_size = config["dataset_size"].as<size_t>();
    
    const auto& selection = config["selection"];
    
    params.dphi_max = selection["dphi_max"].as<float>();
    params.z0_max = selection["z0_max"].as<float>();
    params.chi_max = selection["chi_max"].as<float>();
    params.d_min = selection["d_min"].as<float>();
    params.d_max = selection["d_max"].as<float>();
    params.pt_min = selection["pt_min"].as<float>();
    params.n_phi_sections = selection["n_phi_sections"].as<size_t>();
    params.n_eta_sections = selection["n_eta_sections"].as<size_t>();
    params.eta_min = selection["eta_min"].as<float>();
    params.eta_max = selection["eta_max"].as<float>();
    params.num_rows = selection["num_rows"].as<size_t>();
    params.num_sectors = selection["num_sectors"].as<size_t>();
    params.rmax = selection["rmax"].as<float>();
    params.zmax = selection["zmax"].as<float>();

    return params;
}

float PreProcessing::CalcDphi(float phi1, float phi2) 
{
    float dphi = phi2 - phi1;

    if (dphi > M_PI) dphi -= 2 * M_PI;
    if (dphi < -M_PI) dphi += 2 * M_PI;

    return dphi;
}

float PreProcessing::CalcEta(float r, float z) 
{
    return -std::log(std::tan(std::atan2(r, z) / 2.0));
}

void PreProcessing::SplitDetectorSections(const std::vector<Hit>& hits,
                                          const std::vector<float>& phi_edges,
                                          const std::vector<float>& eta_edges,
                                          std::vector<std::vector<Hit>>& hits_sections) 
{    
    hits_sections.clear();
    
    for (size_t i = 0; i < phi_edges.size() - 1; ++i) 
    {
        float phi_min = phi_edges[i];
        float phi_max = phi_edges[i + 1];
        
        std::vector<Hit> phi_hits;

        for (const auto& hit : hits) 
        {
            if (hit.phi > phi_min && hit.phi < phi_max) 
            {
                phi_hits.push_back(hit);
            }
        }
        
        for (size_t j = 0; j < eta_edges.size() - 1; ++j) 
        {
            float eta_min = eta_edges[j];
            float eta_max = eta_edges[j + 1];
            
            std::vector<Hit> sec_hits;

            for (const auto& hit : phi_hits) 
            {
                float eta = CalcEta(hit.r, hit.z);

                if (eta > eta_min && eta < eta_max) 
                {
                    sec_hits.push_back(hit);
                }
            }
            
            hits_sections.push_back(sec_hits);
        }
    }
}

void PreProcessing::SelectSegments(const std::vector<Hit*>& hits1,
                                   const std::vector<Hit*>& hits2,
                                   float dphi_max, float z0_max,
                                   float chi_max, float d_min, float d_max,
                                   std::vector<Edge>& segments) 
{   
    for (size_t i = 0; i < hits1.size(); ++i) 
    {
        for (size_t j = 0; j < hits2.size(); ++j) 
        {            
            float dphi = CalcDphi(hits1[i]->phi, hits2[j]->phi);
            
            float dz = hits2[j]->z - hits1[i]->z;
            float dr = hits2[j]->r - hits1[i]->r;
            
            float z0 = hits1[i]->z - hits1[i]->r * dz / dr;
            float distance = std::sqrt(hits1[i]->r * hits1[i]->r + hits2[j]->r * hits2[j]->r -
                                       2 * hits1[i]->r * hits2[j]->r * std::cos(dphi) + dz * dz);
            float dtheta = std::atan(dz / dr);

            if ((std::fabs(dphi) < dphi_max) && (std::fabs(z0) < z0_max) &&
                (std::fabs(dtheta) < chi_max) && (distance > d_min) && (distance < d_max)) 
            { 
                Edge edge;

                edge.index_1 = hits1[i]->id;
                edge.index_2 = hits2[j]->id;

                segments.push_back(edge);
            }
        }
    }
}

torch::Tensor PreProcessing::GetEdgeFeatures(const std::vector<float>& in_node,
                                             const std::vector<float>& out_node, 
                                             torch::ScalarType Dtype) 
{
    float in_r = in_node[0], in_phi = in_node[1], in_z = in_node[2];
    float out_r = out_node[0], out_phi = out_node[1], out_z = out_node[2];
    
    float in_r3 = std::sqrt(in_r * in_r + in_z * in_z);
    float out_r3 = std::sqrt(out_r * out_r + out_z * out_z);
    
    float in_theta = std::acos(in_z / in_r3);
    float in_eta = -std::log(std::tan(in_theta / 2.0));
    
    float out_theta = std::acos(out_z / out_r3);
    float out_eta = -std::log(std::tan(out_theta / 2.0));
    
    float deta = out_eta - in_eta;
    float dphi = CalcDphi(out_phi, in_phi);
    float dR = std::sqrt(deta * deta + dphi * dphi);
    float dZ = in_z - out_z;
    
    return torch::tensor({deta, dphi, dR, dZ}, torch::TensorOptions().dtype(Dtype));
}

void PreProcessing::ProcessEvent(const std::vector<Hit>& hits, 
                                 const PreprocessingParams& params, 
                                 std::vector<GraphSample>& graphs,
                                 bool train,
                                 torch::ScalarType Dtype, torch::ScalarType Itype) 
{        
    graphs.clear();
    
    std::vector<float> phi_edges;

    for (size_t i = 0; i <= params.n_phi_sections; ++i) 
    {
        phi_edges.push_back(-M_PI + 2 * M_PI * i / params.n_phi_sections);
    }
    
    std::vector<float> eta_edges;

    for (size_t i = 0; i <= params.n_eta_sections; ++i) 
    {
        eta_edges.push_back(params.eta_min + (params.eta_max - params.eta_min) * i / params.n_eta_sections);
    }

    std::vector<std::vector<Hit>> hits_sections;

    SplitDetectorSections(hits, phi_edges, eta_edges, hits_sections);
    
    for (size_t section_id = 0; section_id < hits_sections.size(); ++section_id) 
    {        
        auto& section_hits = hits_sections[section_id];
        
        if (section_hits.empty()) continue;
        
        std::vector<std::vector<float>> node_positions;
        std::vector<torch::Tensor> node_hit_ids, edge_indices, edge_features, edge_labels;
        std::map<int, std::vector<size_t>> row_groups;

        node_positions.resize(section_hits.size());

        for (size_t i = 0; i < section_hits.size(); ++i) 
        {
            std::vector<float> pos = {section_hits[i].r / params.rmax,
                                      section_hits[i].phi / M_PIf,
                                      section_hits[i].z / params.zmax};  
            node_positions[i] = pos;

            section_hits[i].id = i;

            row_groups[section_hits[i].row_id].push_back(i);

            node_hit_ids.push_back(torch::tensor({section_hits[i].hit_id}, torch::TensorOptions().dtype(Itype)));
        }
        
        for (int row = 0; row < static_cast<int>(params.num_rows) - 1L; ++row) 
        {
            if (row_groups.find(row) == row_groups.end() || 
                row_groups.find(row + 1) == row_groups.end()) 
            {
                continue;
            }
            
            std::vector<Hit*> hits1, hits2;

            for (size_t idx : row_groups[row]) 
            {
                hits1.push_back(&section_hits[idx]);
            }

            for (size_t idx : row_groups[row + 1]) 
            {
                hits2.push_back(&section_hits[idx]);
            }
            
            std::vector<Edge> segments;

            SelectSegments(hits1, hits2, params.dphi_max,
                           params.z0_max, params.chi_max,
                           params.d_min, params.d_max,
                           segments);

            for (const auto& segment : segments) 
            {                
                edge_indices.push_back(torch::tensor({segment.index_1, segment.index_2}, torch::TensorOptions().dtype(Itype)));
                
                auto edge_feat = GetEdgeFeatures(node_positions[segment.index_1], node_positions[segment.index_2], Dtype);
                edge_features.push_back(edge_feat);

                if (train)
                {
                    int label = (section_hits[segment.index_1].track_id == section_hits[segment.index_2].track_id &&
                                 section_hits[segment.index_1].track_id != -1 &&
                                 section_hits[segment.index_1].pt >= params.pt_min) ? 1 : 0;
                    edge_labels.push_back(torch::tensor({static_cast<float>(label)}, torch::TensorOptions().dtype(Dtype)));
                }
            }
        }
        
        if (!edge_indices.empty()) 
        {
            torch::Tensor node_attr = torch::zeros({static_cast<long>(node_positions.size()), 
                                                    static_cast<long>(node_positions.at(0).size())}, 
                                                    torch::TensorOptions().dtype(Dtype));

            for (size_t i = 0; i < node_positions.size(); ++i) 
            {
                for (size_t j = 0; j < node_positions.at(0).size(); ++j) 
                {
                    node_attr[i][j] = node_positions[i][j];
                }
            }
            
            torch::Tensor node_hit_id = torch::stack(node_hit_ids, 0);
            torch::Tensor edge_index = torch::stack(edge_indices, 0).transpose(0, 1);
            torch::Tensor edge_attr = torch::stack(edge_features, 0);
            
            if (train)
            {
                torch::Tensor answer = torch::stack(edge_labels, 0);
                graphs.emplace_back(edge_index, node_attr, node_hit_id, edge_attr, answer);
            }
            else
            {
                graphs.emplace_back(edge_index, node_attr, node_hit_id, edge_attr);
            }
        }
    }
}

void PreProcessing::SaveGraphSample(const GraphSample& sample, const std::string& filename) 
{
    torch::jit::script::Module module("GraphSampleContainer");

    module.register_buffer("edge_index", sample.edge_index);
    module.register_buffer("node_attr", sample.node_attr);
    module.register_buffer("node_hit_id", sample.node_hit_id);
    module.register_buffer("edge_attr", sample.edge_attr);
    module.register_buffer("answer", sample.answer);
    
    module.save(filename);
}