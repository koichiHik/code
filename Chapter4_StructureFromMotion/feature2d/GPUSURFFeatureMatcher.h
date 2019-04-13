/*
 *  GPUSURFFeatureMatcher.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 6/13/12.
 *
 */

#ifndef GPU_SURF_FEATURE_MATCHER_H
#define GPU_SURF_FEATURE_MATCHER_H

// OpenCV
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>

// Original
#include "feature2d/IFeatureMatcher.h"

class GPUSURFFeatureMatcher : public IFeatureMatcher {
public:
	GPUSURFFeatureMatcher(std::vector<cv::Mat>& imgs, 
					   std::vector<std::vector<cv::KeyPoint> >& imgpts);
	
	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches = NULL);
	
	std::vector<cv::KeyPoint> GetImagePoints(int idx) {
		return m_imgPts[idx]; 
	}

private:
	cv::Ptr<cv::gpu::SURF_GPU> m_extractor;
	std::vector<cv::gpu::GpuMat> m_descriptorsOnGpu;
	std::vector<std::vector<cv::KeyPoint> >& m_imgPts;
	bool use_ratio_test;
};

#endif // GPU_SURF_FEATURE_MATCHER_H