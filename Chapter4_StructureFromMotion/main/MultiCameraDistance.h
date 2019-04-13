/*****************************************************************************
*   ExploringSfMWithOpenCV
******************************************************************************
*   by Roy Shilkrot, 5th Dec 2012
*   http://www.morethantechnical.com/
******************************************************************************
*   Ch4 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

#ifndef MULTI_CAMERA_DISTANCE_H
#define MULTI_CAMERA_DISTANCE_H

// STL
#include <vector>

// Boost
#include <utility>

// OpenCV
#include <opencv2/opencv.hpp>

// Original
#include "main/IDistance.h"
#include "calib3d/Triangulation.h"
#include "feature2d/IFeatureMatcher.h"
#include "calib3d/FindCameraMatrices.h"

struct CamParams {
	cv::Mat K;
	cv::Mat_<double> Kinv;
	cv::Mat K32f;
	cv::Mat camMat;
	cv::Mat distCoeff;
	cv::Mat distCoeff32f; 
};

class MultiCameraDistance  : public IDistance {	
protected:
	// Feature Points
	std::vector<std::vector<cv::KeyPoint> > m_imgPts;
	std::vector<std::vector<cv::KeyPoint> > m_imgPtsGood;
	
	// Images
	std::vector<std::string> m_imgNames;
	std::vector<cv::Mat_<cv::Vec3b> > m_originalImgs;
	std::vector<cv::Mat> m_convertedImgs;

	// Matching Matrix
	std::map<std::pair<int,int> ,std::vector<cv::DMatch> > m_matchesMatrix;

	// Pose Matrices
	std::map<int,cv::Matx34d> m_poseMats;

	// Camera Parameters
	CamParams m_camPar;

	// Point Cloud
	std::vector<CloudPoint> pcloud;
	std::vector<cv::Vec3b> pointCloudRGB;
	std::vector<cv::KeyPoint> correspImg1Pt; //TODO: remove
	
	cv::Ptr<IFeatureMatcher> feature_matcher;
	
	bool features_matched;
public:
	bool use_rich_features;
	bool use_gpu;

	std::vector<cv::Point3d> getPointCloud() { 
		return CloudPointsToPoints(pcloud); 
	}

	const cv::Mat& get_im_orig(int frame_num) {
		return m_originalImgs[frame_num]; 
	}

	const std::vector<cv::KeyPoint>& getcorrespImg1Pt() { 
		return correspImg1Pt; 
	}

	const std::vector<cv::Vec3b>& getPointCloudRGB() { if(pointCloudRGB.size()==0) { GetRGBForPointCloud(pcloud,pointCloudRGB); } return pointCloudRGB; }
	std::vector<cv::Matx34d> getCameras() { 
		std::vector<cv::Matx34d> v; 
		for(std::map<int ,cv::Matx34d>::const_iterator it = m_poseMats.begin(); 
				it != m_poseMats.end();
				++it ) {
			v.push_back( it->second );
		}
		return v;
    }

	void GetRGBForPointCloud(
		const std::vector<struct CloudPoint>& pcloud,
		std::vector<cv::Vec3b>& RGBforCloud
		);

	MultiCameraDistance(
		const std::vector<cv::Mat>& imgs_, 
		const std::vector<std::string>& imgs_names_, 
		const std::string& imgs_path_);	
	virtual void OnlyMatchFeatures(int strategy = STRATEGY_USE_FEATURE_MATCH);	
//	bool CheckCoherentRotation(cv::Mat_<double>& R);
};

#endif // MULTI_CAMERA_DISTANCE_H