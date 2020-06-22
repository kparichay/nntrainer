/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	rnn_layer.h
 * @date	19 June 2020
 * @brief	This is Recurrent Neural Network Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __RNN_LAYER_H__
#define __RNN_LAYER_H__
#ifdef __cplusplus

#include <fstream>
#include <iostream>
#include <layer.h>
#include <optimizer.h>
#include <tensor.h>
#include <vector>

namespace nntrainer {

/**
 * @class   RNNLayer
 * @brief   Recurrent Neural Network Layer
 */
class RNNLayer : public Layer {
public:
  /**
   * @brief     Constructor of RNN Layer
   */
  RNNLayer();

  /**
   * @brief     Destructor of RNN Layer
   */
  ~RNNLayer();

  /**
   * @brief     Read RNN parameters and data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save RNN parameters and data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file);

  /**
   * @brief     forward propagation with input
   * @param[in] in Input Tensor from upper layer
   * @retval    output from forward prop
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     back propagation
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    derivative for the next layer
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     initialize layer
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(bool last);

  void setUnit(unsigned int u) { unit = u; };

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     Optimizer Setter
   * @param[in] opt Optimizer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setOptimizer(Optimizer &opt);

private:
  unsigned int unit;
  std::shared_ptr<FullyConnectedLayer> op_u_x;
  std::shared_ptr<FullyConnectedLayer> op_w_st_1;
  std::shared_ptr<FullyConnectedLayer> op_v_st;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RNN_LAYER_H__ */
