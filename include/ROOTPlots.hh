#pragma once

#include <iostream>
#include <algorithm>

#include "GraphDataset.hh"

#include "TROOT.h"
#include "TObject.h"
#include "TSystem.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TPad.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TH1D.h"
#include "TGraph2D.h"
#include "TPolyLine3D.h" 
#include "TStyle.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TString.h"

namespace ROOTPlots
{
    void PlotTrainingLoss(const float* epoch_array, 
                          const float* train_error, 
                          const float* test_error, 
                          const int& num_epochs, 
                          const std::string& output_filename = "Training_loss.root");
    void PlotMetrics(const float* threshold_array, 
                     const float* accuracy_array, 
                     const float* purity_array, 
                     const float* efficiency_array, 
                     const int& num_points, 
                     const std::string& output_filename = "Metrics.root");
    void PlotGraphSample3D(const GraphSample& sample,
                           const float threshold = 0.5,
                           const bool only_true = false, 
                           const std::string& output_filename = "Graph_Visualization.root");
}