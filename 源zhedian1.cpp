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

// �Ľ����¿ڽǶȼ��
void detectGroovePoints(const PointCloudT::Ptr& cloud,
	PointCloudT::Ptr& groove_points,
	int window_size = 10,
	float angle_threshold = 30.0,
	float min_angle = 20.0)
{
	if (cloud->size() < 2 * window_size + 1)
		return;

	// ����ȷ�������������(���߼���ɨ��˳��)
	// ���û��˳����Ҫ�ȶԵ��ƽ�������

	for (size_t i = window_size; i < cloud->size() - window_size; ++i)
	{
		// ǰ�򴰿ڵ���
		PointCloudT::Ptr front_cloud(new PointCloudT);
		for (int j = 1; j <= window_size; ++j)
			front_cloud->push_back(cloud->points[i - j]);

		// ���򴰿ڵ���
		PointCloudT::Ptr back_cloud(new PointCloudT);
		for (int j = 1; j <= window_size; ++j)
			back_cloud->push_back(cloud->points[i + j]);

		// ��ǰ�󴰿ڷֱ����PCA������ȡ������
		pcl::PCA<PointT> pca_front;
		pca_front.setInputCloud(front_cloud);
		Eigen::Vector3f front_dir = pca_front.getEigenVectors().col(0);

		pcl::PCA<PointT> pca_back;
		pca_back.setInputCloud(back_cloud);
		Eigen::Vector3f back_dir = pca_back.getEigenVectors().col(0);

		// ����������н�(0-90��)
		float angle = pcl::rad2deg(acos(abs(front_dir.normalized().dot(back_dir.normalized()))));

		// �Ƕ���ָ����Χ������Ϊ���¿ڵ�
		if (angle >= min_angle && angle <= angle_threshold)
		{
			groove_points->push_back(cloud->points[i]);
		}
	}
}

// �Ľ��Ľǵ���(������ʺ�PCA)
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

		// ǰ��ֲ�����
		PointCloudT::Ptr front_cloud(new PointCloudT);
		for (int j = 1; j <= k_neighbors / 2; ++j)
			front_cloud->push_back(cloud->points[i - j]);

		// ����ֲ�����
		PointCloudT::Ptr back_cloud(new PointCloudT);
		for (int j = 1; j <= k_neighbors / 2; ++j)
			back_cloud->push_back(cloud->points[i + j]);

		// PCA����
		pcl::PCA<PointT> pca_front;
		pca_front.setInputCloud(front_cloud);
		Eigen::Vector3f front_dir = pca_front.getEigenVectors().col(0);

		pcl::PCA<PointT> pca_back;
		pca_back.setInputCloud(back_cloud);
		Eigen::Vector3f back_dir = pca_back.getEigenVectors().col(0);

		float angle = pcl::rad2deg(acos(abs(front_dir.normalized().dot(back_dir.normalized()))));

		// ���ʸ߻�Ƕȴ�ĵ㱻��Ϊ�ǽǵ�
		if (curvature > curvature_threshold || angle > angle_threshold)
		{
			corner_points->push_back(cloud->points[i]);
		}
	}
}

int main66(int argc, char** argv)
{
	// ���ص���
	PointCloudT::Ptr cloud(new PointCloudT);
	if (pcl::io::loadPCDFile<PointT>("3.pcd", *cloud) == -1)
	{
		std::cerr << "Couldn't read file line_cloud.pcd" << std::endl;
		return -1;
	}

	// Ԥ���� - ������(����ʵ����Ҫ����)
	PointCloudT::Ptr filtered_cloud(new PointCloudT);
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.1f, 0.1f, 0.1f); // ����ϸ�Ĳ���
	sor.filter(*filtered_cloud);

	// ���������
	PointCloudT::Ptr corner_points(new PointCloudT);    // �ǵ�
	PointCloudT::Ptr groove_points(new PointCloudT);     // �¿ڵ�

	detectCornerPoints(filtered_cloud, corner_points);
	detectGroovePoints(filtered_cloud, groove_points, 3, 22.0, 17.0); // ר�ż��30-60��֮����¿�
	//detectGroovePoints(filtered_cloud, groove_points, 5, 60.0, 5.0); // ר�ż��30-60��֮����¿�
	// ���ӻ�
	pcl::visualization::PCLVisualizer viewer("Improved Feature Points Viewer");
	viewer.setBackgroundColor(0, 0, 0);

	// ԭʼ���� - ��ɫ
	viewer.addPointCloud<PointT>(filtered_cloud, "original_cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
		1.0, 1.0, 1.0, "original_cloud");

	// �ǵ� - ��ɫ
	if (!corner_points->empty())
	{
		viewer.addPointCloud<PointT>(corner_points, "corner_points");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
			1.0, 0.0, 0.0, "corner_points");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
			5, "corner_points");
	}

	// �¿ڵ� - ��ɫ(50���¿�)
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