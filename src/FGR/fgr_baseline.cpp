//
// Created by prajval on 06.10.21.
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

    PointCloudT::Ptr source(new PointCloudT);
    PointCloudT::Ptr target(new PointCloudT);
    PointCloudT::Ptr target_estimate(new PointCloudT);

    std::string path_to_model = "/home/prajval/MT_CertifiableRobustPCR/fitting_graphs/datasets/data/stanford_dataset/bunny_model.pcd";
    // load model
    if (pcl::io::loadPCDFile<PointT>(path_to_model, *source) < 0) {
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

    // Apply random rotation, translation, scale and outlier points
    Eigen::Matrix4d R = applyRandomRotation(*target);
    double s = 1.0; // FGR cannot estimate scale
    Eigen::Matrix4d t = applyRandomTranslation(*target);
    addGaussianNoiseToCloud(*target);
    addOutlierPoints(*target, s, t, R);
    Eigen::Matrix4d gt_trans = t;
    gt_trans.block(0,0,3,3) = R.block(0,0,3,3);

    // Generate Outlier correspondences
    pcl::Correspondences outlier_correspondences;
    for(int it=0; it != source->points.size(); it++){
        for(int jt = 0; jt!= target->points.size(); jt++){
            // save only the incorrect corr
            if(it != jt){
                pcl::Correspondence temp_corr;
                temp_corr.index_query = it;
                temp_corr.index_match = jt;
                outlier_correspondences.push_back(temp_corr);
            }
        }
    }

    int num_corr = 100;
    float outlier_ratio = 0.95;
    int num_inlier = floor(num_corr * (1-outlier_ratio));
    int num_outlier = num_corr - num_inlier;

    pcl::Correspondences final_corr;
    for(int it=0; it != num_inlier; it++){
        int randomIndex = rand() % ground_truth_correspondences.size();
        pcl::Correspondence temp_corr = ground_truth_correspondences[randomIndex];
        final_corr.push_back(temp_corr);
    }

    for(int it=0; it != num_outlier; it++){
        int randomIndex = rand() % outlier_correspondences.size();
        pcl::Correspondence temp_corr = outlier_correspondences[randomIndex];
        final_corr.push_back(temp_corr);
    }

    // FGR Stuff here
    fgr::CApp app;
    fgr::Points fgr_source = pclToFGRpointcloud(*source);
    fgr::Points fgr_target = pclToFGRpointcloud(*target);
    app.pointcloud_.push_back(fgr_target);
    app.pointcloud_.push_back(fgr_source);
    app.corres_ = pclToFGRCorr(final_corr);

    app.NormalizePoints();
    app.OptimizePairwise(true);
    Eigen::Matrix4f estimated_tf = app.GetOutputTrans();
    std::cout << "Ground Truth Tf\n" <<gt_trans << std::endl;
    std::cout << "Estimated Tf\n" <<estimated_tf << std::endl;
//
//    // visualiser
//    pcl::transformPointCloud(*source, *target_estimate, estimated_tf);
//    pcl::visualization::PCLVisualizer visu("FGR");
//    visu.setBackgroundColor(0, 0, 0);
//    visu.addCoordinateSystem(0.1);
//    visu.setSize(1280, 1024); // Visualiser window size
//    pcl::visualization::PointCloudColorHandlerCustom<PointT> source_handler(
//            source, 0, 255, 255);
//    visu.addPointCloud (source,source_handler, "source");
//    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_handler(
//            target, 255, 0, 0);
//    visu.addPointCloud (target,target_handler, "target");
//    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_est_handler(
//            target_estimate, 0, 0, 100);
//    visu.addPointCloud (target_estimate,target_est_handler, "target_est_handler");
//
//    // set point size
//    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source");
//    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target");
//    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target_est_handler");
//
//    visu.addCube(-0.5, 0.5,-0.5, 0.5,-0.5, 0.5, 1.0, 1.0, 1.0, "cube");
//    visu.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");
//
//    visu.spin();

    return EXIT_SUCCESS;
}
