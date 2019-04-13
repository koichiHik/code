/*
 *  GPUSURFFeatureMatcher.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 6/13/12.
 *
 */

#include "IFeatureMatcher.h"
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>

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
	//std::vector<cv::gpu::GpuMat> m_imgsOnGpu;
	//std::vector<cv::gpu::GpuMat> imgPtsOnGpu;
	std::vector<std::vector<cv::KeyPoint> >& m_imgPts;
	bool use_ratio_test;
};
