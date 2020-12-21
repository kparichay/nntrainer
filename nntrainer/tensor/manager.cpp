// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	manager.cpp
 * @date	2 Dec 2020
 * @brief	This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <functional>
#include <vector>

#include <manager.h>

namespace nntrainer {

/**
 * @brief     Add weight to be tracked and updated with nntrainer
 */
void Manager::trackWeight(std::reference_wrapper<Weight> w) {
  std::vector<std::reference_wrapper<Weight>> temp = {w};
  weights.emplace_back(temp);
}

/**
 * @brief     Add weights to be tracked and updated with nntrainer
 */
void Manager::trackWeights(std::vector<Weight> &ws) {
  std::vector<std::reference_wrapper<Weight>> layer_weights;
  layer_weights.reserve(ws.size());

  size_t weight_size = 0;

  for (auto &w : ws) {
    layer_weights.emplace_back(std::ref(w));
    if (w.getTrainable())
      weight_size += w.getDim().getDataLen();
  }

  weights.push_back(layer_weights);

  max_weight_size = std::max(max_weight_size, weight_size);
}

/**
 * @brief Allocate and initialize the weight variable
 */
void Manager::initialize() {
  Tensor shared_grad;
  if (max_weight_size > 0 && enable_gradient_memory_opt)
    shared_grad = Tensor(max_weight_size);

  for (auto &l_w : weights) {
    size_t offset = 0;
    for (auto &w : l_w) {
      Weight &weight = w.get();
      if (weight.getTrainable() && enable_gradient_memory_opt) {
        weight.initialize(
          shared_grad.getSharedDataTensor(weight.getDim(), offset));
        offset += weight.getDim().getDataLen();
      } else {
        weight.initialize();
      }
    }
  }
}

/**
 * @brief Track the inputs/ouputs of the layer
 * @note This assumes are layers are being tracked in sorted order of execution
 */
void Manager::TrackLayerInOuts(const std::string layer_name,
                               const std::vector<TensorDim> &input_dim,
                               bool trainable) {
  int cnt = 0;
  auto base_name = layer_name + ":InOut";

  size_t inout_derivative_size = 0;

  std::vector<std::shared_ptr<Var_Grad>> in_out;
  in_out.reserve(input_dim.size());

  for (auto const &dim : input_dim) {
    in_out.emplace_back(std::make_shared<Var_Grad>(
      dim, trainable, base_name + std::to_string(cnt++)));
    if (trainable)
      inout_derivative_size += dim.getDataLen();
  }

  in_outs.push_back(in_out);

  max_derivative_size = std::max(max_derivative_size, inout_derivative_size);
}

void Manager::untrackLayerInOuts(const std::string layer_name) {
  auto var_name = layer_name + ":InOut" + std::to_string(0);

  for (unsigned int cnt = 0; cnt < in_outs.size(); cnt ++) {
    if (!in_outs[cnt].empty() && in_outs[cnt][0]->getName() == var_name) {
      in_outs.erase(in_outs.begin() + cnt);
      break;
    }
  }
}

/**
 * @brief Initialize the inputs/outputs for the layer
 */
void Manager::initializeInOuts(bool trainable) {
  // Tensor shared_deriv;
  // if (enable_derivative_memory_opt) {
  //   max_derivative_size = 0;

  //   size_t in_derivative_size = 0, out_derivative_size = 0;
  //   for (unsigned int i = 0; i < in_outs.size(); i++) {
  //     auto &l_io = in_outs[i];
  //     in_derivative_size = 0;
  //     for (auto &io : l_io) {
  //       if (io->getTrainable())
  //         in_out_derivative_size += io->getDim().getDataLen();
  //     }

  //     max_derivative_size = max(max_derivative_size, in_derivative_size + out_derivative_size);
  //     out_derivative_size = in_derivative_size;
  //   }

  //   shared_deriv = Tensor(max_derivative_size);
  // }

  Tensor shared_deriv;
  if (max_derivative_size > 0 && enable_derivative_memory_opt)
    shared_deriv = Tensor(max_derivative_size);

  for (auto &l_io : in_outs) {
    size_t offset = 0;
    for (auto &io : l_io) {
      if (io->getTrainable() && enable_derivative_memory_opt) {
        io->initialize(shared_deriv.getSharedDataTensor(io->getDim(), offset),
                       trainable);
        offset += io->getDim().getDataLen() * trainable;
      } else {
        io->initialize(Tensor(), trainable);
      }
    }
  }
}

/**
 * @brief Set the batch size for the inputs/outputs of the layers
 */
void Manager::setBatchSize(unsigned int batch) {
  if (!in_outs.empty() && !in_outs[0].empty()) {
    max_derivative_size /= in_outs[0][0]->getDim().batch();
    max_derivative_size *= batch;
  }
  for (auto &in_out : in_outs)
    for (auto &vg : in_out)
      vg->setBatchSize(batch);
}

} // namespace nntrainer
