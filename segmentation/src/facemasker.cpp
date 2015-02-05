/*
Copyright (c) 2013-2014, Gregory P. Meyer
                         University of Illinois Board of Trustees
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <dip/segmentation/facemasker.h>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

namespace dip {

FaceMasker::~FaceMasker() {
  if (boundary_ != NULL)
    delete [] boundary_;
  if (distances_ != NULL)
    delete [] distances_;
  if (min_sizes_ != NULL)
    delete [] min_sizes_;
  if (max_sizes_ != NULL)
    delete [] max_sizes_;
}

void FaceMasker::Run(int max_difference, int min_depth, int max_depth,
                     float min_face_size, float max_face_size,
                     int window_size, int width, int height,
                     float focal_length, const Depth *depth) {
  width_ = width;
  height_ = height;
  window_size_ = window_size;

  if (size_ < (width_ * height_)) {
    size_ = width_ * height_;

    if (boundary_ != NULL)
      delete [] boundary_;
    if (distances_ != NULL)
      delete [] distances_;
    if (min_sizes_ != NULL)
      delete [] min_sizes_;
    if (max_sizes_ != NULL)
      delete [] max_sizes_;

    boundary_ = new bool[size_];
    distances_ = new unsigned int[size_];
    min_sizes_ = new float[size_];
    max_sizes_ = new float[size_];
  }

  memset(boundary_, 0, sizeof(bool) * size_);

  #pragma omp parallel for
  for (int y = 1; y < height_ - 1; y++) {
    for (int x = 1; x < width_ - 1; x++) {
      int i = x + y * width_;

      if ((DIFF(depth[i], depth[i - 1]) > max_difference) ||
          (DIFF(depth[i], depth[i - width]) > max_difference)) {
        boundary_[i] = true;
      }
    }
  }
  cv::Mat boundary_bool(height, width, CV_8UC1, boundary_);
  cv::Mat boundary_mat(height, width, CV_8UC1, cvScalar(0));
  boundary_mat.setTo(255, boundary_bool);
  imshow("boundary", boundary_mat);

  //distance计算每个像素距离最近的边缘的像素距离
  distance_.Run(width_, height_, boundary_, distances_);

  memset(min_sizes_, 0, sizeof(float) * size_);
  memset(max_sizes_, 0, sizeof(float) * size_);

  Mat tmp_mask(height_, width_, CV_8UC1);
  tmp_mask.setTo(0);
  unsigned char * pmask = tmp_mask.data;

  #pragma omp parallel for
  for (int y = 1; y < height_ - 1; y++) {
    for (int x = 1; x < width_ - 1; x++) {
      int i = x + y * width_;

      if ((depth[i] > min_depth) && (depth[i] < max_depth)) {
        float min_size = (min_face_size * focal_length) / depth[i];
        float max_size = (max_face_size * focal_length) / depth[i];

        int mean_radius = (int)((min_size + max_size) / 4.0f);
        int difference = (int)((max_size - min_size) / 2.0f);

		//判断平均半径处，左边、右边、上边三个方向，
		//要求这三个方向的半径处距离最近边缘的距离不能太远，否则当前点不可能作为头部中心
        if (x > mean_radius) {
          if (distances_[i - mean_radius] > difference)
            continue;
        }

        if (x < (width_ - mean_radius)) {
          if (distances_[i + mean_radius] > difference)
            continue;
        }

        if (y > mean_radius) {
          if (distances_[i - mean_radius * width_] > difference)
            continue;
        }

		//此处的最小大小和最大大小表示以该点为中心的人脸，可能的最大像素大小和最小像素大小
        min_sizes_[i] = min_size;
        max_sizes_[i] = max_size;
		pmask[i] = 255;
      }
    }
  }

  imshow("tmp_mask", tmp_mask);
  waitKey(1);
}

Mat FaceMasker::generateMask(const Mat& src) {
  Mat mask = Mat::zeros(src.size(), CV_8U);
  //将当前输入窗口缩放到与训练所用窗口大小的尺度上
  float scale = (float)src.cols / (float)width_;
  float inv_scale = 1.0f / scale;
  float half_window = window_size_ / 2.0f;
  float scaled_window_size = window_size_ * inv_scale;

  int rows = (int)(src.rows - half_window);
  int cols = (int)(src.cols - half_window);

  #pragma omp parallel for
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      int Y = (int)((y + half_window) * inv_scale);
      int X = (int)((x + half_window) * inv_scale);

      if ((Y < height_) && (X < width_)) {
        int i = X + Y * width_;
		//只将有可能为头部内部的点置为255
        if ((scaled_window_size >= min_sizes_[i]) &&
            (scaled_window_size <= max_sizes_[i])) {
          mask.at<unsigned char>(y, x) = 255;
        }
      }
    }
  }

  return mask;
}

} // namespace dip
