#pragma once

#include <iostream>
#include <map>
#include <set>

#include "GraphDataset.hh"

namespace PostProcessing
{
    std::vector<std::set<int>> GetTracks(const GraphSample& graph, const float threshold = 0.5);
}