/*
 *  SfMUpdateListener.h
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 10/7/12.
 *
 */

#ifndef SFM_UPDATE_LISTENER_H
#define SFM_UPDATE_LISTENER_H

// STL
#include <vector>

// OpenCV
#include <opencv2/core/core.hpp>

class SfMUpdateListener
{
public:
	virtual void update(std::vector<cv::Point3d> pcld,
						std::vector<cv::Vec3b> pcldrgb, 
						std::vector<cv::Point3d> pcld_alternate,
						std::vector<cv::Vec3b> pcldrgb_alternate, 
						std::vector<cv::Matx34d> cameras) = 0;
};

#endif // SFM_UPDATE_LISTENER_H
