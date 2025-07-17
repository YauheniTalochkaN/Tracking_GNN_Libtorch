#include "ROOTPlots.hh"

void ROOTPlots::PlotTrainingLoss(const float* epoch_array, 
                                 const float* train_error, 
                                 const float* test_error, 
                                 const int& num_epochs,
                                 const std::string& output_filename)
{
    TFile* fout = 0;
	fout = new TFile(output_filename.c_str(), "update");
    
    TString topdir = gDirectory->GetPath();

    TCanvas* can = new TCanvas("Training_dynamics", "Training_dynamics", 0, 0, 700, 500);
    can->GetPad(0)->SetLogy();

    TGraph* accuracy_graph = new TGraph(num_epochs, epoch_array, train_error);
    TGraph* purity_graph = new TGraph(num_epochs, epoch_array, test_error);
    accuracy_graph->SetName("train");
	accuracy_graph->GetHistogram()->SetTitle("");
	accuracy_graph->SetLineColor(2);
    accuracy_graph->SetLineWidth(2.5);
    purity_graph->SetName("test");
	purity_graph->GetHistogram()->SetTitle("");
	purity_graph->SetLineColor(1);
    purity_graph->SetLineWidth(2.5);

    TLegend* legend = new TLegend(0.8, 0.75, 0.9, 0.85, NULL, "brNDC");
    //legend->SetHeader("","C");
	legend->SetTextSize(0.035);

    TMultiGraph* mgraph = new TMultiGraph("Training_dynamics","Training dynamics");
    mgraph->Add(accuracy_graph);
    legend->AddEntry(accuracy_graph, "train", "l");
    mgraph->Add(purity_graph);
    legend->AddEntry(purity_graph, "test", "l");

    mgraph->SetName("Training dynamics");
	mgraph->GetHistogram()->SetTitle("");;
    mgraph->GetHistogram()->GetXaxis()->SetTitle("Epoch number");
    mgraph->GetHistogram()->GetYaxis()->SetTitle("Loss");
    mgraph->GetHistogram()->GetXaxis()->SetTitleSize(0.04);
    mgraph->GetHistogram()->GetXaxis()->SetTitleOffset(1.1);
    mgraph->GetHistogram()->GetYaxis()->SetTitleSize(0.04);
    mgraph->GetHistogram()->GetYaxis()->SetTitleOffset(1.5);
    mgraph->GetHistogram()->GetXaxis()->SetRangeUser(epoch_array[0], epoch_array[num_epochs-1]);

    can->cd();
    mgraph->Draw("AL");

    gPad->SetBottomMargin(0.13);
    gPad->SetRightMargin(0.08);
    gPad->SetLeftMargin(0.12);
    gPad->SetTopMargin(0.1);
    gPad->UnZoomed();
	gPad->Modified();
	gPad->Update();

    legend->Draw();
    
    gDirectory->cd(topdir);
	can->Write(0,TObject::kWriteDelete,0);
	can = nullptr;

	fout->Close();
	fout = nullptr;
}

void ROOTPlots::PlotMetrics(const float* threshold_array, 
                            const float* accuracy_array, 
                            const float* purity_array, 
                            const float* efficiency_array, 
                            const int& num_points,
                            const std::string& output_filename)
{
    TFile* fout = 0;
	fout = new TFile(output_filename.c_str(), "update");
    
    TString topdir = gDirectory->GetPath();

    TCanvas* can = new TCanvas("Metrics", "Metrics", 0, 0, 700, 500);

    TGraph* accuracy_graph = new TGraph(num_points, threshold_array, accuracy_array);
    TGraph* purity_graph = new TGraph(num_points, threshold_array, purity_array);
    TGraph* efficiency_graph = new TGraph(num_points, threshold_array, efficiency_array);
    accuracy_graph->SetName("train");
	accuracy_graph->GetHistogram()->SetTitle("");
	accuracy_graph->SetLineColor(1);
    accuracy_graph->SetLineWidth(2.5);
    purity_graph->SetName("test");
	purity_graph->GetHistogram()->SetTitle("");
	purity_graph->SetLineColor(2);
    purity_graph->SetLineWidth(2.5);
    efficiency_graph->SetName("test");
	efficiency_graph->GetHistogram()->SetTitle("");
	efficiency_graph->SetLineColor(3);
    efficiency_graph->SetLineWidth(2.5);

    TLegend* legend = new TLegend(0.4, 0.4, 0.6, 0.6, NULL, "brNDC");
    //legend->SetHeader("","C");
	legend->SetTextSize(0.035);

    TMultiGraph* mgraph = new TMultiGraph("Metrics","Metrics");
    mgraph->Add(accuracy_graph);
    legend->AddEntry(accuracy_graph, "accuracy", "l");
    mgraph->Add(purity_graph);
    legend->AddEntry(purity_graph, "purity", "l");
    mgraph->Add(efficiency_graph);
    legend->AddEntry(efficiency_graph, "efficiency", "l");

    mgraph->SetName("Metrics");
	mgraph->GetHistogram()->SetTitle("");;
    mgraph->GetHistogram()->GetXaxis()->SetTitle("Threshold");
    mgraph->GetHistogram()->GetYaxis()->SetTitle("Metrics");
    mgraph->GetHistogram()->GetXaxis()->SetTitleSize(0.04);
    mgraph->GetHistogram()->GetXaxis()->SetTitleOffset(1.1);
    mgraph->GetHistogram()->GetYaxis()->SetTitleSize(0.04);
    mgraph->GetHistogram()->GetYaxis()->SetTitleOffset(1.5);
    mgraph->GetHistogram()->GetXaxis()->SetRangeUser(threshold_array[0], threshold_array[num_points-1]);

    can->cd();
    mgraph->Draw("AL");

    gPad->SetBottomMargin(0.13);
    gPad->SetRightMargin(0.08);
    gPad->SetLeftMargin(0.12);
    gPad->SetTopMargin(0.1);
    gPad->UnZoomed();
	gPad->Modified();
	gPad->Update();

    legend->Draw();
    
    gDirectory->cd(topdir);
	can->Write(0,TObject::kWriteDelete,0);
	can = nullptr;

	fout->Close();
	fout = nullptr;
}

void ROOTPlots::PlotGraphSample3D(const GraphSample& sample, 
                                  const float threshold,
                                  const bool only_true,
                                  const std::string& output_filename) 
{
    TFile* fout = new TFile(output_filename.c_str(), "recreate");
    
    TString topdir = gDirectory->GetPath();
    
    TCanvas* canvas = new TCanvas("Graph_Visualization", "Graph_Visualization", 0, 0, 800, 600);
    
    auto edge_index_accessor = sample.edge_index.accessor<int, 2>();
    auto node_attr_accessor = sample.node_attr.accessor<float, 2>();
    auto answer_accessor = sample.answer.accessor<float, 2>();
    
    int num_nodes = sample.node_attr.size(0);
    int num_edges = sample.edge_index.size(1);
    int num_answers = sample.answer.size(0);
    
    TGraph2D* nodes_graph = new TGraph2D(num_nodes);
    nodes_graph->SetName("Graph");
    nodes_graph->SetTitle("; X; Y; Z");
    nodes_graph->GetXaxis()->SetTitleSize(0.04);
    nodes_graph->GetXaxis()->SetTitleOffset(1.1);
    nodes_graph->GetYaxis()->SetTitleSize(0.04);
    nodes_graph->GetYaxis()->SetTitleOffset(1.5);
    nodes_graph->GetZaxis()->SetTitleSize(0.04);
    nodes_graph->GetZaxis()->SetTitleOffset(1.2);
    
    for (int i = 0; i < num_nodes; ++i) 
    {
        float r = node_attr_accessor[i][0];
        float phi = M_PIf * node_attr_accessor[i][1]; 
        float z = node_attr_accessor[i][2];
        
        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        
        nodes_graph->SetPoint(i, x, y, z);
    }

    nodes_graph->SetMarkerStyle(kFullCircle);
    nodes_graph->SetMarkerSize(0.6);
    nodes_graph->SetMarkerColor(kBlack);
    
    std::vector<TPolyLine3D*> edge_lines;
    
    for (int i = 0; i < num_edges; ++i) 
    {
        int node1_idx = edge_index_accessor[0][i];
        int node2_idx = edge_index_accessor[1][i];
        
        float r1 = node_attr_accessor[node1_idx][0];
        float phi1 = M_PIf * node_attr_accessor[node1_idx][1];
        float z1 = node_attr_accessor[node1_idx][2];
        
        float r2 = node_attr_accessor[node2_idx][0];
        float phi2 = M_PIf * node_attr_accessor[node2_idx][1];
        float z2 = node_attr_accessor[node2_idx][2];
        
        float x1 = r1 * std::cos(phi1);
        float y1 = r1 * std::sin(phi1);
        
        float x2 = r2 * std::cos(phi2);
        float y2 = r2 * std::sin(phi2);
        
        TPolyLine3D* edge_line = new TPolyLine3D(2);
        
        if (num_answers == num_edges)
        {
            if (answer_accessor[i][0] > threshold) 
            {
                edge_line->SetPoint(0, x1, y1, z1);
                edge_line->SetPoint(1, x2, y2, z2);

                edge_line->SetLineColor(kGreen + 3);
                edge_line->SetLineWidth(2);
            } 
            else 
            {
                if (!only_true)
                {
                    edge_line->SetPoint(0, x1, y1, z1);
                    edge_line->SetPoint(1, x2, y2, z2);

                    edge_line->SetLineColor(kGray + 2); 
                    edge_line->SetLineWidth(1.5);
                }
            }
        }
        else 
        {
            edge_line->SetLineColor(kGray + 2);
            edge_line->SetLineWidth(1.5);
        }

        edge_lines.push_back(edge_line);
    }
    
    canvas->cd();

    nodes_graph->Draw("P");
    
    for (auto& edge_line : edge_lines) 
    {
        edge_line->Draw("same");
    }
    
    gPad->SetBottomMargin(0.13);
    gPad->SetRightMargin(0.08);
    gPad->SetLeftMargin(0.12);
    gPad->SetTopMargin(0.1);
    gPad->UnZoomed();
    gPad->Modified();
    gPad->Update();
    
    gDirectory->cd(topdir);
    canvas->Write(0, TObject::kWriteDelete, 0);
    canvas = nullptr;
    
    fout->Close();
    fout = nullptr;
}