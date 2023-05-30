//
// Created by prajval on 06.10.21.
//

#ifndef FGR_BASELINE_FGR_UTILS_H
#define FGR_BASELINE_FGR_UTILS_H

//C++ headers
#include <chrono>
#include <iostream>
#include <random>
#include <cstdlib>

#include <Eigen/Core>
#include "Eigen/Geometry"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/generate.h>
#include <pcl/common/random.h>

#include "app.h"

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::Normal NormalT;
typedef pcl::PointCloud<NormalT> PointCloudNT;

double applyRandomScale(PointCloudT& cloud_in);
Eigen::Matrix4d applyRandomRotation(PointCloudT& cloud_in);
Eigen::Matrix4d applyRandomTranslation(PointCloudT& cloud_in);
void addGaussianNoiseToCloud(PointCloudT& cloud_in);
void addOutlierPoints(PointCloudT& cloud_in, double scale, Eigen::Matrix4d translation, Eigen::Matrix4d rotation);
bool extractNormals(const PointCloudT::Ptr& cloud_in,
                    pcl::PointCloud<NormalT>::Ptr& normal_out,
                    int Kneighbour_normal = 10,
                    bool changeViewPoint = false,
                    std::vector<float> view_point = {0,0,0});
bool extractKeypointDetector(const PointCloudT::Ptr& cloud_in, PointCloudT::Ptr& extracted_keypoints);
bool extractFPFHdescriptors(PointCloudT::Ptr &cloud_in, PointCloudNT::Ptr &normal_in, pcl::PointCloud<pcl::FPFHSignature33>::Ptr &descriptors_out);
fgr::Points pclToFGRpointcloud(PointCloudT& pcl_cloud);
PointCloudT FGRToPClpointcloud(fgr::Points& fgr_cloud);
fgr::Correspondences pclToFGRCorr(pcl::Correspondences pcl_corr);

#endif //FGR_BASELINE_FGR_UTILS_H
