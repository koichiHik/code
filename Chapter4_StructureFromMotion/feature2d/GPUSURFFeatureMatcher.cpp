/*
 *  GPUSURFFeatureMatcher.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 6/13/12.
 *
 */

// STD
#include <iostream>

// STL
#include <set>

// OpenCV
#include <opencv2/features2d/features2d.hpp>

// Original
#include "feature2d/GPUSURFFeatureMatcher.h"
#include "calib3d/FindCameraMatrices.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

GPUSURFFeatureMatcher::GPUSURFFeatureMatcher(
	vector<cv::Mat>& imgs_,
	vector<std::vector<cv::KeyPoint> >& imgpts_
	) : m_imgPts(imgpts_),use_ratio_test(true)
{
	m_extractor = new gpu::SURF_GPU();
	
	std::cout << " -------------------- extract feature points for all images (GPU) -------------------\n";
	
	m_imgPts.resize(imgs_.size());
	m_descriptorsOnGpu.resize(imgs_.size());

	CV_PROFILE("extract",
	for(int img_i=0;img_i<imgs_.size();img_i++) {
		GpuMat _m; _m.upload(imgs_[img_i]);
		(*m_extractor)(_m,GpuMat(),m_imgPts[img_i],m_descriptorsOnGpu[img_i]);
		cout << ".";
	}
	)
}	

void GPUSURFFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches) {

	const vector<KeyPoint>& imgpts1 = m_imgPts[idx_i];
	const vector<KeyPoint>& imgpts2 = m_imgPts[idx_j];
	const GpuMat& descriptors_1 = m_descriptorsOnGpu[idx_i];
	const GpuMat& descriptors_2 = m_descriptorsOnGpu[idx_j];
	
	std::vector< DMatch > good_matches_,very_good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	
	//cout << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << endl;
	//cout << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << endl;
	
	keypoints_1 = imgpts1;
	keypoints_2 = imgpts2;
	
	if(descriptors_1.empty()) {
		CV_Error(0,"descriptors_1 is empty");
	}
	if(descriptors_2.empty()) {
		CV_Error(0,"descriptors_2 is empty");
	}
	
	//matching descriptor vectors using Brute Force matcher
	BruteForceMatcher_GPU<L2<float> > matcher;
	std::vector< DMatch > matches_;
	if (matches == NULL) {
		matches = &matches_;
	}
	if (matches->size() == 0) {
		cout << "match " << descriptors_1.rows << " vs. " << descriptors_2.rows << " ...";

		if(use_ratio_test) {
			vector<vector<DMatch> > knn_matches;
			GpuMat trainIdx,distance,allDist;
			CV_PROFILE("match", 
				matcher.knnMatchSingle(descriptors_1,descriptors_2,trainIdx,distance,allDist,2); 
				matcher.knnMatchDownload(trainIdx,distance,knn_matches);
			)

			(*matches).clear();

			//ratio test
			for(int i=0;i<knn_matches.size();i++) {
				if(knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) {
					(*matches).push_back(knn_matches[i][0]);
				}
			}
			cout << "kept " << (*matches).size() << " features after ratio test"<<endl;
		} else {
			CV_PROFILE("match",matcher.match( descriptors_1, descriptors_2, *matches );)
		}
	}
}
