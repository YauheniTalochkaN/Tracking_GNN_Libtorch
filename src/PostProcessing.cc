#include "PostProcessing.hh"

std::vector<std::set<int>> PostProcessing::GetTracks(const GraphSample& graph, const float threshold) 
{
    auto binary_answer = (graph.answer >= threshold).to(torch::kInt32);
    
    auto mask = binary_answer.squeeze() == 1;

    auto in_indices = torch::masked_select(graph.edge_index[0], mask);
    auto out_indices = torch::masked_select(graph.edge_index[1], mask);
    
    auto in_hits = graph.node_hit_id.index_select(0, in_indices).squeeze();
    auto out_hits = graph.node_hit_id.index_select(0, out_indices).squeeze();
    
    auto pairs = torch::stack({out_hits, in_hits}, 1);
    auto accessor = pairs.accessor<int, 2>();
    
    std::unordered_map<int, int> parent;
    std::function<int(int)> find = [&parent, &find](int x) -> int 
    {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };

    for (int64_t i = 0; i < pairs.size(0); ++i) 
    {
        int a = accessor[i][0], b = accessor[i][1];

        if (!parent.count(a)) 
        {
            parent[a] = a;
        }

        if (!parent.count(b)) 
        {
            parent[b] = b;
        }

        parent[find(a)] = find(b);
    }
    
    std::unordered_map<int, std::set<int>> temp;

    for (auto& p : parent) 
    {
        temp[find(p.first)].insert(p.first);
    }

    std::vector<std::set<int>> result_vec;

    for (auto& group : temp) 
    {
        result_vec.push_back(std::move(group.second));
    }

    return result_vec;
}