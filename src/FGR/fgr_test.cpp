//
// Created by prajval on 22.06.21.
// This is a test executable to run the FGR method given any model cloud. It randomnly rotates and translates the cloud, adds noise and
// extracts features from fpfh. The features are then used to extract correspondences and then execute FGR
//

//C++ headers
#include <chrono>
#include <iostream>
#include <random>
#include <cstdlib>

// FGR
#include "include/app.h"
#include "include/fgr_utils.h"

int main (int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Please provide path to model" << std::endl;
        return EXIT_FAILURE;
    }

    PointCloudT::Ptr source(new PointCloudT);
    PointCloudT::Ptr target_estimate(new PointCloudT);
    PointCloudT::Ptr target(new PointCloudT);
    PointCloudNT::Ptr source_normal (new PointCloudNT);
    PointCloudNT::Ptr target_normal (new PointCloudNT);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_descriptor (new pcl::PointCloud<pcl::FPFHSignature33> ());
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_descriptor (new pcl::PointCloud<pcl::FPFHSignature33> ());

    // load model
    if (pcl::io::loadPCDFile<PointT>(argv[1], *source) < 0) {
        pcl::console::print_error("Error loading object file!\n");
        return EXIT_FAILURE;
    }

    // center points
    Eigen::Vector4f centroid_cloud;
    pcl::compute3DCentroid(*source, centroid_cloud);
    for(int p=0; p != source->points.size() ; p++){
        source->points[p].x = source->points[p].x - centroid_cloud[0];
        source->points[p].y = source->points[p].y - centroid_cloud[1];
        source->points[p].z = source->points[p].z - centroid_cloud[2];
    }

    // scale to 1m cube
    PointT max_pt;
    PointT min_pt;
    pcl::getMinMax3D(*source,min_pt, max_pt);
    double x_scale = 1 / abs(max_pt.x - min_pt.x);
    double y_scale = 1 / abs(max_pt.y - min_pt.y);
    double z_scale = 1 / abs(max_pt.z - min_pt.z);

    for(int p=0; p != source->points.size() ; p++){
        source->points[p].x =  x_scale * source->points[p].x;
        source->points[p].y =  y_scale * source->points[p].y;
        source->points[p].z =  z_scale * source->points[p].z;
    }

    std::cout<<"number of points: "<<source->points.size() << std::endl;
    // copy source and target
    pcl::copyPointCloud(*source, *target);

    // create ground truth correspondences
    pcl::Correspondences ground_truth_correspondences;
    for(int i=0; i!= source->points.size(); i++){
        pcl::Correspondence temp_corr;
        temp_corr.index_query = i;
        temp_corr.index_match = i;
        ground_truth_correspondences.push_back(temp_corr);
    }

    Eigen::Matrix4d R = applyRandomRotation(*target);
    double s = 1.0;
    Eigen::Matrix4d t = applyRandomTranslation(*target);
    addGaussianNoiseToCloud(*target);
    addOutlierPoints(*target, s, t, R);
    Eigen::Matrix4d gt_trans = t;
    gt_trans.block(0,0,3,3) = R.block(0,0,3,3);


    // Extract features
    extractNormals(source, source_normal);
    extractNormals(target, target_normal);
    extractFPFHdescriptors(source, source_normal, source_descriptor);
    extractFPFHdescriptors(target, target_normal, target_descriptor);
    FILE* fid = fopen("source_features.bin", "wb");
    int nV = source->size(), nDim = 33;
    fwrite(&nV, sizeof(int), 1, fid);
    fwrite(&nDim, sizeof(int), 1, fid);
    for (int v = 0; v < nV; v++) {
        const PointT &pt = source->points[v];
        float xyz[3] = {pt.x, pt.y, pt.z};
        fwrite(xyz, sizeof(float), 3, fid);
        const pcl::FPFHSignature33 &feature = source_descriptor->points[v];
        fwrite(feature.histogram, sizeof(float), 33, fid);
    }
    fclose(fid);

    FILE* fid1 = fopen("target_features.bin", "wb");
    int nV1 = target->size();
    fwrite(&nV1, sizeof(int), 1, fid);
    fwrite(&nDim, sizeof(int), 1, fid);
    for (int v = 0; v < nV1; v++) {
        const PointT &pt = target->points[v];
        float xyz[3] = {pt.x, pt.y, pt.z};
        fwrite(xyz, sizeof(float), 3, fid);
        const pcl::FPFHSignature33 &feature = target_descriptor->points[v];
        fwrite(feature.histogram, sizeof(float), 33, fid);
    }
    fclose(fid1);

    // FGR registration
    fgr::CApp app;

    app.ReadFeature("target_features.bin");
    app.ReadFeature("source_features.bin");
    app.NormalizePoints();
    app.AdvancedMatching();
    app.OptimizePairwise(true);

    Eigen::Matrix4f estimated_tf = app.GetOutputTrans();
    std::cout << "Ground Truth Tf\n" <<gt_trans << std::endl;
    std::cout << "Estimated Tf\n" <<estimated_tf << std::endl;

    pcl::transformPointCloud(*source, *target_estimate, estimated_tf);

    // visualiser
    pcl::visualization::PCLVisualizer visu("FGR");
    visu.setBackgroundColor(0, 0, 0);
    visu.addCoordinateSystem(0.1);
    visu.setSize(1280, 1024); // Visualiser window size
    pcl::visualization::PointCloudColorHandlerCustom<PointT> source_handler(
            source, 0, 255, 255);
    visu.addPointCloud (source,source_handler, "source");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_handler(
            target, 255, 0, 0);
    visu.addPointCloud (target,target_handler, "target");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_est_handler(
            target_estimate, 0, 0, 100);
    visu.addPointCloud (target_estimate,target_est_handler, "target_est_handler");

    // set point size
    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source");
    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target");
    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target_est_handler");

    visu.addCube(-0.5, 0.5,-0.5, 0.5,-0.5, 0.5, 1.0, 1.0, 1.0, "cube");
    visu.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");

    visu.spin();

    return EXIT_SUCCESS;
}