#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <vector>
#include <cmath>
#include <pcl/common/distances.h>
#include<pcl/common/angles.h>
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct FeaturePoints {
	PointCloudT::Ptr endpoints;    // �˵�
	PointCloudT::Ptr breakpoints;  // �ϵ�
	PointCloudT::Ptr corners;      // �ǵ�
	PointCloudT::Ptr inflections;  // �յ�
	PointCloudT::Ptr sharp_points; // �����

	FeaturePoints() :
		endpoints(new PointCloudT),
		breakpoints(new PointCloudT),
		corners(new PointCloudT),
		inflections(new PointCloudT),
		sharp_points(new PointCloudT) {}
};

class LineFeatureExtractor {
public:
	LineFeatureExtractor() :
		break_distance_(5),
		angle_threshold_(0.01),
		curvature_threshold_(15),
		window_size_(5) {}

	FeaturePoints extract(const PointCloudT::Ptr& cloud) {
		FeaturePoints features;

		// Ԥ�����������
		sortPointCloud(cloud);

		// ����������
		detectEndpoints(cloud, features.endpoints);
		detectBreakpoints(cloud, features.breakpoints);
		detectCorners(cloud, features.corners);
		detectInflections(cloud, features.inflections);
		detectSharpPoints(cloud, features.sharp_points);

		return features;
	}

private:
	float break_distance_;     // �ϵ������ֵ
	float angle_threshold_;    // �ǵ�Ƕ���ֵ(��)
	float curvature_threshold_;// ������ֵ
	int window_size_;          // ��ⴰ�ڴ�С

	// �������򣨼����������
	void sortPointCloud(const PointCloudT::Ptr& cloud) {
		std::sort(cloud->begin(), cloud->end(),
			[](const PointT& a, const PointT& b) {
			return a.x < b.x; // ����ɨ�跽�����
		});
	}

	// �˵���
	void detectEndpoints(const PointCloudT::Ptr& cloud,
		PointCloudT::Ptr& endpoints) {
		if (cloud->empty()) return;

		// ��β����Ϊ��ѡ�˵�
		endpoints->push_back(cloud->front());
		endpoints->push_back(cloud->back());

		// ��֤�Ƿ�Ϊ��ʵ�˵�
		auto validateEndpoint = [&](int index) {
			const int check_size = 3;
			if (index < check_size || index >= cloud->size() - check_size)
				return true;

			// ���ǰ����ܶ�
			float front_dist = pcl::euclideanDistance(
				cloud->points[index],
				cloud->points[index - check_size]);

			float back_dist = pcl::euclideanDistance(
				cloud->points[index],
				cloud->points[index + check_size]);

			return (front_dist > 2 * back_dist) || (back_dist > 2 * front_dist);
		};

		if (validateEndpoint(0)) endpoints->push_back(cloud->front());
		if (validateEndpoint(cloud->size() - 1)) endpoints->push_back(cloud->back());
	}

	// �ϵ���
	void detectBreakpoints(const PointCloudT::Ptr& cloud,
		PointCloudT::Ptr& breakpoints) {
		for (size_t i = 1; i < cloud->size(); ++i) {
			float dist = pcl::euclideanDistance(cloud->points[i], cloud->points[i - 1]);
			if (dist > break_distance_) {
				breakpoints->push_back(cloud->points[i - 1]);
				breakpoints->push_back(cloud->points[i]);
			}
		}
	}

	// �ǵ��⣨���ڽǶȱ仯��
	void detectCorners(const PointCloudT::Ptr& cloud,
		PointCloudT::Ptr& corners) {
		const int k = window_size_;
		std::vector<float> angles(cloud->size(), 0.0);

		for (int i = k; i < cloud->size() - k; ++i) {
			Eigen::Vector2f prev_vec(
				cloud->points[i].x - cloud->points[i - k].x,
				cloud->points[i].y - cloud->points[i - k].y);

			Eigen::Vector2f next_vec(
				cloud->points[i + k].x - cloud->points[i].x,
				cloud->points[i + k].y - cloud->points[i].y);

			float angle = acos(prev_vec.normalized().dot(next_vec.normalized()));
			angles[i] = pcl::rad2deg(angle);
		}

		// Ѱ�Ҿֲ����ֵ
		for (int i = k + 1; i < cloud->size() - k - 1; ++i) {
			if (angles[i] > angle_threshold_/* &&
				angles[i] > angles[i - 1] &&
				angles[i] > angles[i + 1]*/) {
				corners->push_back(cloud->points[i]);
			}
		}
	}

	// �յ��⣨�������ʱ仯��
	void detectInflections(const PointCloudT::Ptr& cloud,
		PointCloudT::Ptr& inflections) {
		std::vector<float> curvatures(cloud->size(), 0.0);

		for (int i = 1; i < cloud->size() - 1; ++i) {
			Eigen::Vector2f dx(
				cloud->points[i + 1].x - cloud->points[i - 1].x,
				cloud->points[i + 1].y - cloud->points[i - 1].y);

			Eigen::Vector2f dxx(
				cloud->points[i + 1].x - 2 * cloud->points[i].x + cloud->points[i - 1].x,
				cloud->points[i + 1].y - 2 * cloud->points[i].y + cloud->points[i - 1].y);

			float curvature = dx.norm() > 0 ? dxx.norm() / pow(dx.norm(), 5) : 0;
			curvatures[i] = curvature;
		}

		// ������ʷ��ű仯
		for (int i = 2; i < cloud->size() - 2; ++i) {
			if (curvatures[i] * curvatures[i - 1] < 0 ||
				curvatures[i] * curvatures[i + 1] < 0) {
				inflections->push_back(cloud->points[i]);
			}
		}
	}

	// ������⣨�����ʣ�
	void detectSharpPoints(const PointCloudT::Ptr& cloud,
		PointCloudT::Ptr& sharp_points) {
		std::vector<float> curvatures(cloud->size(), 0.0);

		for (int i = window_size_; i < cloud->size() - window_size_; ++i) {
			PointT left = cloud->points[i - window_size_];
			PointT right = cloud->points[i + window_size_];

			float dx = right.x - left.x;
			float dy = right.y - left.y;
			float ds = sqrt(dx*dx + dy * dy);

			float curvature = ds > 0 ? 2 * abs(dy) / ds : 0;
			curvatures[i] = curvature;
		}

		for (size_t i = 0; i < curvatures.size(); ++i) {
			if (curvatures[i] > curvature_threshold_) {
				sharp_points->push_back(cloud->points[i]);
			}
		}
	}
};

// ���ӻ�����
void visualizeFeatures(const PointCloudT::Ptr& cloud,
	const FeaturePoints& features) {
	pcl::visualization::PCLVisualizer viewer("Line Features");
	viewer.setBackgroundColor(0, 0, 0);

	// ԭʼ���ƣ���ɫ��
	viewer.addPointCloud<PointT>(cloud, "cloud");
	viewer.setPointCloudRenderingProperties(
		pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "cloud");

	// �˵㣨��ɫ��
	if (!features.endpoints->empty()) {
		viewer.addPointCloud<PointT>(features.endpoints, "endpoints");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "endpoints");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "endpoints");
	}

	// �ϵ㣨��ɫ��
	if (!features.breakpoints->empty()) {
		viewer.addPointCloud<PointT>(features.breakpoints, "breakpoints");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "breakpoints");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "breakpoints");
	}

	// �ǵ㣨��ɫ��
	if (!features.corners->empty()) {
		viewer.addPointCloud<PointT>(features.corners, "corners");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "corners");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "corners");
	}

	// �յ㣨��ɫ��
	if (!features.inflections->empty()) {
		viewer.addPointCloud<PointT>(features.inflections, "inflections");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 0, "inflections");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "inflections");
	}

	// ����㣨Ʒ�죩
	if (!features.sharp_points->empty()) {
		viewer.addPointCloud<PointT>(features.sharp_points, "sharp_points");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 1, "sharp_points");
		viewer.setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "sharp_points");
	}

	while (!viewer.wasStopped()) {
		viewer.spinOnce(100);
	}
}

int main(int argc, char** argv) {
	// 1. ���ص���
	PointCloudT::Ptr cloud(new PointCloudT);
	pcl::io::loadPCDFile<PointT>("3.pcd", *cloud);

	// 2. ������ȡ
	LineFeatureExtractor extractor;
	FeaturePoints features = extractor.extract(cloud);

	// 3. ���ӻ�
	visualizeFeatures(cloud, features);

	// 4. ����������ѡ��
	pcl::io::savePCDFile("endpoints.pcd", *features.endpoints);
	pcl::io::savePCDFile("breakpoints.pcd", *features.breakpoints);
	pcl::io::savePCDFile("corners.pcd", *features.corners);
	pcl::io::savePCDFile("inflections.pcd", *features.inflections);
	pcl::io::savePCDFile("sharp_points.pcd", *features.sharp_points);

	return 0;
}