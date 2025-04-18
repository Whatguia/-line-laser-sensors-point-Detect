#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>
#include <cmath>
#include<pcl/common/angles.h>
#include <pcl/common/distances.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <vector>
#include <cmath>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// 改进的坡口角度检测
void detectGroovePoints(const PointCloudT::Ptr& cloud,
	PointCloudT::Ptr& groove_points,
	int window_size = 10,
	float angle_threshold = 30.0,
	float min_angle = 20.0)
{
	if (cloud->size() < 2 * window_size + 1)
		return;

	// 首先确保点云是有序的(按线激光扫描顺序)
	// 如果没有顺序，需要先对点云进行排序

	for (size_t i = window_size; i < cloud->size() - window_size; ++i)
	{
		// 前向窗口点云
		PointCloudT::Ptr front_cloud(new PointCloudT);
		for (int j = 1; j <= window_size; ++j)
			front_cloud->push_back(cloud->points[i - j]);

		// 后向窗口点云
		PointCloudT::Ptr back_cloud(new PointCloudT);
		for (int j = 1; j <= window_size; ++j)
			back_cloud->push_back(cloud->points[i + j]);

		// 对前后窗口分别进行PCA分析获取主方向
		pcl::PCA<PointT> pca_front;
		pca_front.setInputCloud(front_cloud);
		Eigen::Vector3f front_dir = pca_front.getEigenVectors().col(0);

		pcl::PCA<PointT> pca_back;
		pca_back.setInputCloud(back_cloud);
		Eigen::Vector3f back_dir = pca_back.getEigenVectors().col(0);

		// 计算两方向夹角(0-90度)
		float angle = pcl::rad2deg(acos(abs(front_dir.normalized().dot(back_dir.normalized()))));

		// 角度在指定范围内则认为是坡口点
		if (angle >= min_angle && angle <= angle_threshold)
		{
			groove_points->push_back(cloud->points[i]);
		}
	}
}

// 改进的角点检测(结合曲率和PCA)
void detectCornerPoints(const PointCloudT::Ptr& cloud,
	PointCloudT::Ptr& corner_points,
	int k_neighbors = 9,
	float curvature_threshold = 10.01,
	float angle_threshold =18.0)
{
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	ne.setSearchMethod(tree);
	ne.setInputCloud(cloud);
	ne.setKSearch(k_neighbors);
	ne.compute(*normals);

	for (size_t i = k_neighbors; i < cloud->size() - k_neighbors; ++i)
	{
		float curvature = normals->points[i].curvature;

		// 前向局部点云
		PointCloudT::Ptr front_cloud(new PointCloudT);
		for (int j = 1; j <= k_neighbors / 2; ++j)
			front_cloud->push_back(cloud->points[i - j]);

		// 后向局部点云
		PointCloudT::Ptr back_cloud(new PointCloudT);
		for (int j = 1; j <= k_neighbors / 2; ++j)
			back_cloud->push_back(cloud->points[i + j]);

		// PCA分析
		pcl::PCA<PointT> pca_front;
		pca_front.setInputCloud(front_cloud);
		Eigen::Vector3f front_dir = pca_front.getEigenVectors().col(0);

		pcl::PCA<PointT> pca_back;
		pca_back.setInputCloud(back_cloud);
		Eigen::Vector3f back_dir = pca_back.getEigenVectors().col(0);

		float angle = pcl::rad2deg(acos(abs(front_dir.normalized().dot(back_dir.normalized()))));

		// 曲率高或角度大的点被认为是角点
		if (curvature > curvature_threshold || angle > angle_threshold)
		{
			corner_points->push_back(cloud->points[i]);
		}
	}
}

int main66(int argc, char** argv)
{
	// 加载点云
	PointCloudT::Ptr cloud(new PointCloudT);
	if (pcl::io::loadPCDFile<PointT>("3.pcd", *cloud) == -1)
	{
		std::cerr << "Couldn't read file line_cloud.pcd" << std::endl;
		return -1;
	}

	// 预处理 - 降采样(根据实际需要调整)
	PointCloudT::Ptr filtered_cloud(new PointCloudT);
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.1f, 0.1f, 0.1f); // 更精细的采样
	sor.filter(*filtered_cloud);

	// 检测特征点
	PointCloudT::Ptr corner_points(new PointCloudT);    // 角点
	PointCloudT::Ptr groove_points(new PointCloudT);     // 坡口点

	detectCornerPoints(filtered_cloud, corner_points);
	detectGroovePoints(filtered_cloud, groove_points, 3, 22.0, 17.0); // 专门检测30-60度之间的坡口
	//detectGroovePoints(filtered_cloud, groove_points, 5, 60.0, 5.0); // 专门检测30-60度之间的坡口
	// 可视化
	pcl::visualization::PCLVisualizer viewer("Improved Feature Points Viewer");
	viewer.setBackgroundColor(0, 0, 0);

	// 原始点云 - 白色
	viewer.addPointCloud<PointT>(filtered_cloud, "original_cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
		1.0, 1.0, 1.0, "original_cloud");

	// 角点 - 红色
	if (!corner_points->empty())
	{
		viewer.addPointCloud<PointT>(corner_points, "corner_points");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
			1.0, 0.0, 0.0, "corner_points");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
			5, "corner_points");
	}

	// 坡口点 - 黄色(50度坡口)
	if (!groove_points->empty())
	{
		viewer.addPointCloud<PointT>(groove_points, "groove_points");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
			1.0, 1.0, 0.0, "groove_points");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
			5, "groove_points");
	}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
	}

	return 0;
}