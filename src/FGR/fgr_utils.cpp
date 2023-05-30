//
// Created by prajval on 06.10.21.
//

#include "include/fgr_utils.h"


double applyRandomScale(PointCloudT& cloud_in){
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 generator(rd()); // seed the generator
    std::uniform_real_distribution<double> distribution(0.5,1.5);
    double rand_scale = distribution(generator);
    for(int p=0; p != cloud_in.points.size() ; p++){
        cloud_in.points[p].x = rand_scale * cloud_in.points[p].x;
        cloud_in.points[p].y = rand_scale * cloud_in.points[p].y;
        cloud_in.points[p].z = rand_scale * cloud_in.points[p].z;
    }
    return rand_scale;
};

Eigen::Matrix4d applyRandomRotation(PointCloudT& cloud_in){
    Eigen::Quaternion<double> random_quat = Eigen::Quaternion<double>::UnitRandom();
    Eigen::Matrix3d random_rot = random_quat.toRotationMatrix();
    Eigen::Matrix4d random_tf = Eigen::Matrix4d::Identity();
    random_tf.block(0, 0, 3, 3) = random_rot;
    pcl::transformPointCloud(cloud_in, cloud_in, random_tf);
    return random_tf;
};

Eigen::Matrix4d applyRandomTranslation(PointCloudT& cloud_in){
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 generator(rd()); // seed the generator
    std::uniform_real_distribution<double> distribution(-1.5, 1.5); // define the position range
    Eigen::Matrix4d random_trans = Eigen::Matrix4d::Identity();
    random_trans(0,3) = distribution(generator);
    random_trans(1,3) = distribution(generator);
    random_trans(2,3) = distribution(generator);
    pcl::transformPointCloud(cloud_in, cloud_in, random_trans);
    return random_trans;
};

void addGaussianNoiseToCloud(PointCloudT& cloud_in){
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 generator(rd()); // seed the generator
    std::uniform_real_distribution<double> distribution(-0.01, 0.01); // define the position range
    for(int p=0; p != cloud_in.points.size() ; p++){
        cloud_in.points[p].x =  cloud_in.points[p].x + distribution(generator);
        cloud_in.points[p].y =  cloud_in.points[p].y + distribution(generator);
        cloud_in.points[p].z =  cloud_in.points[p].z + distribution(generator);
    }
};

void addOutlierPoints(PointCloudT& cloud_in, double scale, Eigen::Matrix4d translation, Eigen::Matrix4d rotation){
    PointCloudT outlier_points;
    pcl::common::CloudGenerator<pcl::PointXYZ, pcl::common::NormalGenerator<float> > generator;
    std::uint32_t seed = static_cast<std::uint32_t> (time (nullptr));
    pcl::common::NormalGenerator<float>::Parameters x_params (0, 1, seed++);
    generator.setParametersForX (x_params);
    pcl::common::NormalGenerator<float>::Parameters y_params (0, 1, seed++);
    generator.setParametersForY (y_params);
    pcl::common::NormalGenerator<float>::Parameters z_params (0, 1, seed++);
    generator.setParametersForZ (z_params);
    generator.fill (200, 1, outlier_points);
    PointT max_pt;
    PointT min_pt;
    pcl::getMinMax3D(outlier_points,min_pt, max_pt);
    double x_scale = 1 / abs(max_pt.x - min_pt.x);
    double y_scale = 1 / abs(max_pt.y - min_pt.y);
    double z_scale = 1 / abs(max_pt.z - min_pt.z);

    for(int p=0; p != outlier_points.points.size() ; p++){
        outlier_points.points[p].x =  x_scale * outlier_points.points[p].x;
        outlier_points.points[p].y =  y_scale * outlier_points.points[p].y;
        outlier_points.points[p].z =  z_scale * outlier_points.points[p].z;
    }

    for(int p=0; p != outlier_points.points.size() ; p++){
        outlier_points.points[p].x =  scale * outlier_points.points[p].x;
        outlier_points.points[p].y =  scale * outlier_points.points[p].y;
        outlier_points.points[p].z =  scale * outlier_points.points[p].z;
    }
    pcl::transformPointCloud(outlier_points, outlier_points, rotation);
    pcl::transformPointCloud(outlier_points, outlier_points, translation);

    for(int p=0; p != outlier_points.points.size() ; p++){
        cloud_in.points.push_back(outlier_points.points[p]);
    }
};

bool extractNormals(const PointCloudT::Ptr& cloud_in,
                    pcl::PointCloud<NormalT>::Ptr& normal_out,
                    int Kneighbour_normal /*= 10*/,
                    bool changeViewPoint /*= false*/,
                    std::vector<float> view_point /*= {0,0,0}*/)
{
    if (cloud_in->size() == 0)
        return false;

    pcl::NormalEstimationOMP<PointT, NormalT> norm_est;
    norm_est.setKSearch(Kneighbour_normal);
    norm_est.setInputCloud(cloud_in);
    if(changeViewPoint){
        norm_est.setViewPoint(view_point[0], view_point[1], view_point[2]);
    }

    norm_est.compute(*normal_out);
    return true;
};

bool extractKeypointDetector(const PointCloudT::Ptr& cloud_in, PointCloudT::Ptr& extracted_keypoints)
{

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    // Compute keypoints
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
    iss_detector.setSearchMethod(tree);
    iss_detector.setSalientRadius(0.005);
    iss_detector.setNonMaxRadius(0.001);
    iss_detector.setThreshold21(0.975);
    iss_detector.setThreshold32(0.975);
    iss_detector.setMinNeighbors(1);
    iss_detector.setNumberOfThreads(8);
    iss_detector.setInputCloud(cloud_in);
    iss_detector.compute(*extracted_keypoints);

    return true;
};


bool extractFPFHdescriptors(PointCloudT::Ptr &cloud_in, PointCloudNT::Ptr &normal_in, pcl::PointCloud<pcl::FPFHSignature33>::Ptr &descriptors_out){
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud (cloud_in);
    fpfh.setInputNormals (normal_in);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_object (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod (tree_object);
    fpfh.setKSearch(10);
    // Compute the features
    fpfh.compute (*descriptors_out);
    return true;
}

fgr::Points pclToFGRpointcloud(PointCloudT& pcl_cloud){
    fgr::Points fgr_cloud;
    for(int it=0; it!= pcl_cloud.points.size(); it++){
        Eigen::Vector3f pt;
        pt = {pcl_cloud.points[it].x, pcl_cloud.points[it].y, pcl_cloud.points[it].z};
        fgr_cloud.push_back(pt);
    }
    return fgr_cloud;
}


PointCloudT FGRToPClpointcloud(fgr::Points& fgr_cloud){
    PointCloudT pcl_cloud;
    for(int it=0; it!= fgr_cloud.size(); it++){
        PointT pt;
        pt = {fgr_cloud[it].x(), fgr_cloud[it].y(), fgr_cloud[it].z()};
        pcl_cloud.points.push_back(pt);
    }
    return pcl_cloud;
}

fgr::Correspondences pclToFGRCorr(pcl::Correspondences pcl_corr){
    fgr::Correspondences fgr_corr;
    for(int it=0; it!= pcl_corr.size(); it++){
        std::pair<int, int> corr_temp;
        corr_temp.first = pcl_corr[it].index_query;
        corr_temp.second = pcl_corr[it].index_match;
        fgr_corr.push_back(corr_temp);
    }
    return fgr_corr;
}
