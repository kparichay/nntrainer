/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	input_layer.cpp
 * @date	14 May 2020
 * @brief	This is Input Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <random>

#include <input_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace nntrainer {

const std::string InputLayer::type = "input";

int InputLayer::initialize(Manager &manager) {
  output_dim = input_dim;

  rng.seed(getSeed());

  float translation_factor = 0.09;
  flip_dist = std::uniform_real_distribution<float>(0.0, 1.0);
  translate_dist = std::uniform_real_distribution<float>(-translation_factor,
                                                         translation_factor);

  affine_transform_mat = cv::Mat::zeros(2, 3, CV_32FC1);
  affine_transform_mat.at<float>(0, 0) = 1;
  affine_transform_mat.at<float>(1, 1) = 1;

  // Made for 3 channel input
  input_mat =
    cv::Mat::zeros(input_dim[0].height(), input_dim[0].width(), CV_32FC3);
  output_mat =
    cv::Mat::zeros(input_dim[0].height(), input_dim[0].width(), CV_32FC3);

  return ML_ERROR_NONE;
}

void InputLayer::setProperty(const PropertyType type,
                             const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::normalization:
    if (!value.empty()) {
      status = setBoolean(normalization, value);
      throw_status(status);
    }
    break;
  case PropertyType::standardization:
    if (!value.empty()) {
      status = setBoolean(standardization, value);
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void InputLayer::forwarding() {
  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  if (normalization)
    input_.normalization_i();
  if (standardization)
    input_.standardization_i();

  if (augmentation) {
    unsigned int batch = input_dim[0].batch();
    for (unsigned int b = 0; b < batch; b++) {

      /** random translation */
      float translate_x = translate_dist(rng) * input_dim[0].width();
      float translate_y = translate_dist(rng) * input_dim[0].height();
      affine_transform_mat.at<cv::Vec2f>(0, 0)[2] = translate_x;
      affine_transform_mat.at<cv::Vec2f>(1, 0)[2] = translate_y;

      for (unsigned int c = 0; c < input_dim[0].channel(); c++)
        for (unsigned int h = 0; h < input_dim[0].height(); h++)
          for (unsigned int w = 0; w < input_dim[0].width(); w++)
            input_mat.at<cv::Vec3f>(h, w)[c] = input_.getValue(b, c, h, w);

      cv::warpAffine(input_mat, output_mat, affine_transform_mat,
          output_mat.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

      if (flip_dist(rng) < 0.5) {
        cv::flip(output_mat, input_mat, 1);
      } else {
        output_mat.copyTo(input_mat);
      }

      for (unsigned int c = 0; c < input_dim[0].channel(); c++)
        for (unsigned int h = 0; h < input_dim[0].height(); h++)
          for (unsigned int w = 0; w < input_dim[0].width(); w++)
            hidden_.setValue(b, c, h, w, input_mat.at<cv::Vec3f>(h, w)[c]);
    }
  } else {
    hidden_ = input_;
  }
}

void InputLayer::calcDerivative() {
  throw exception::not_supported(
    "calcDerivative for input layer is not supported");
}

void InputLayer::setTrainable(bool train) {
  if (train)
    throw exception::not_supported("Input layer does not support training");

  Layer::setTrainable(false);
}

} /* namespace nntrainer */
