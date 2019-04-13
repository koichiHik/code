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

// Original
#include "main/MultiCameraDistance.h"
#include "feature2d/RichFeatureMatcher.h"
#include "feature2d/OFFeatureMatcher.h"
#include "feature2d/GPUSURFFeatureMatcher.h"

MultiCameraDistance::MultiCameraDistance(
	const std::vector<cv::Mat>& imgs_, 
	const std::vector<std::string>& imgs_names_, 
	const std::string& imgs_path_) :
m_imgNames(imgs_names_),
features_matched(false),
use_rich_features(true),
use_gpu(true)
{		
	std::cout << "=========================== Load Images ===========================\n";
	//ensure images are CV_8UC3
	for (unsigned int i=0; i<imgs_.size(); i++) {
		m_originalImgs.push_back(cv::Mat_<cv::Vec3b>());
		if (!imgs_[i].empty()) {
			if (imgs_[i].type() == CV_8UC1) {
				cvtColor(imgs_[i], m_originalImgs[i], CV_GRAY2BGR);
			} else if (imgs_[i].type() == CV_32FC3 || imgs_[i].type() == CV_64FC3) {
				imgs_[i].convertTo(m_originalImgs[i],CV_8UC3,255.0);
			} else {
				imgs_[i].copyTo(m_originalImgs[i]);
			}
		}
		
		m_convertedImgs.push_back(cv::Mat());
		cvtColor(m_originalImgs[i], m_convertedImgs[i], CV_BGR2GRAY);
		
		m_imgPts.push_back(std::vector<cv::KeyPoint>());
		m_imgPtsGood.push_back(std::vector<cv::KeyPoint>());
		std::cout << ".";
	}
	std::cout << std::endl;
		
	//load calibration matrix
	cv::FileStorage fs;
	if(fs.open(imgs_path_+ "\\out_camera_data.yml",cv::FileStorage::READ)) {
		fs["camera_matrix"] >> m_camPar.camMat;
		fs["distortion_coefficients"] >> m_camPar.distCoeff;
	} else {
		//no calibration matrix file - mockup calibration
		cv::Size imgs_size = imgs_[0].size();
		double max_w_h = MAX(imgs_size.height,imgs_size.width);
		m_camPar.camMat  = (
			cv::Mat_<double>(3,3) << max_w_h, 0, imgs_size.width/2.0,
															 0, max_w_h, imgs_size.height/2.0,
															 0,	0, 1);
		m_camPar.distCoeff = cv::Mat_<double>::zeros(1,4);
	}
	
	m_camPar.K = m_camPar.camMat;
	invert(m_camPar.K, m_camPar.Kinv); //get inverse of camera matrix

	m_camPar.distCoeff.convertTo(m_camPar.distCoeff32f, CV_32FC1);
	m_camPar.K.convertTo(m_camPar.K32f, CV_32FC1);
}

void MultiCameraDistance::OnlyMatchFeatures(int strategy) 
{
	if(features_matched) return;
	
	if (use_rich_features) {
		if (use_gpu) {
			feature_matcher = new GPUSURFFeatureMatcher(m_convertedImgs, m_imgPts);
		} else {
			feature_matcher = new RichFeatureMatcher(m_convertedImgs, m_imgPts);
		}
	} else {
		feature_matcher = new OFFeatureMatcher(use_gpu, m_convertedImgs, m_imgPts);
	}	

	if(strategy & STRATEGY_USE_OPTICAL_FLOW)
		use_rich_features = false;

	int loop1_top = m_convertedImgs.size() - 1;
	int loop2_top = m_convertedImgs.size();
	int frame_num_i = 0;
	//#pragma omp parallel for schedule(dynamic)
	
	//if (use_rich_features) {
	//	for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
	//		for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
	//		{
	//			std::vector<cv::KeyPoint> fp,fp1;
	//			std::cout << "------------ Match " << imgs_names[frame_num_i] << ","<<imgs_names[frame_num_j]<<" ------------\n";
	//			std::vector<cv::DMatch> matches_tmp;
	//			feature_matcher->MatchFeatures(frame_num_i,frame_num_j,&matches_tmp);
	//			
	//			//#pragma omp critical
	//			{
	//				matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;
	//			}
	//		}
	//	}
	//} else {
#pragma omp parallel for
		for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
			for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
			{
				std::cout << "------------ Match " << m_imgNames[frame_num_i] << ","<<m_imgNames[frame_num_j]<<" ------------\n";
				std::vector<cv::DMatch> matches_tmp;
				feature_matcher->MatchFeatures(frame_num_i,frame_num_j,&matches_tmp);
				m_matchesMatrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;

				std::vector<cv::DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
				m_matchesMatrix[std::make_pair(frame_num_j,frame_num_i)] = matches_tmp_flip;
			}
		}
	//}

	features_matched = true;
}

void MultiCameraDistance::GetRGBForPointCloud(
	const std::vector<struct CloudPoint>& _pcloud,
	std::vector<cv::Vec3b>& RGBforCloud
	) 
{
	RGBforCloud.resize(_pcloud.size());
	for (unsigned int i=0; i<_pcloud.size(); i++) {
		unsigned int good_view = 0;
		std::vector<cv::Vec3b> point_colors;
		for(; good_view < m_originalImgs.size(); good_view++) {
			if(_pcloud[i].imgpt_for_img[good_view] != -1) {
				int pt_idx = _pcloud[i].imgpt_for_img[good_view];
				if(pt_idx >= m_imgPts[good_view].size()) {
					std::cerr << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << m_imgPts[good_view].size() << std::endl;
					continue;
				}
				cv::Point _pt = m_imgPts[good_view][pt_idx].pt;
				assert(good_view < m_originalImgs.size() && _pt.x < m_originalImgs[good_view].cols && _pt.y < m_originalImgs[good_view].rows);
				
				point_colors.push_back(m_originalImgs[good_view].at<cv::Vec3b>(_pt));
				
//				std::stringstream ss; ss << "patch " << good_view;
//				imshow_250x250(ss.str(), imgs_orig[good_view](cv::Range(_pt.y-10,_pt.y+10),cv::Range(_pt.x-10,_pt.x+10)));
			}
		}
//		cv::waitKey(0);
		cv::Scalar res_color = cv::mean(point_colors);
		RGBforCloud[i] = (cv::Vec3b(res_color[0],res_color[1],res_color[2])); //bgr2rgb
		if(good_view == m_convertedImgs.size()) //nothing found.. put red dot
			RGBforCloud.push_back(cv::Vec3b(255,0,0));
	}
}
