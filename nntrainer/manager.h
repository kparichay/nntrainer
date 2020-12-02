// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	manager.h
 * @date	30 Nov 2020
 * @brief	This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __MANAGER_H__
#define __MANAGER_H__
#ifdef __cplusplus

#include <functional>
#include <vector>

#include <weight.h>

namespace nntrainer {

/**
 * @class   Manager
 * @brief   manager of nntrainer
 */
class Manager {
public:
  /**
   * @brief     Constructor of Manager
   */
  Manager() : enable_gradient_memory_opt(true) {}

  /**
   * @brief     Destructor of Manager
   */
  ~Manager() {}

  /**
   * @brief     Add weight to be tracked and updated with nntrainer
   *
   * @param w   Weight to be tracked
   */
  void trackWeight(std::reference_wrapper<Weight> w) {
    weights.emplace_back(w);
  }

  /**
   * @brief     Add weights to be tracked and updated with nntrainer
   *
   * @param ws  Weights to be tracked
   */
  void trackWeights(std::vector<Weight> &ws) {
    weights.reserve(weights.size() + ws.size());
    for (auto &w : ws)
      weights.emplace_back(std::ref(w));
  }

  /**
   * @brief     Get weights tracked with nntrainer
   *
   * @retval    list of weight references
   */
  std::vector<std::reference_wrapper<Weight>> getWeightRefs() {
    return weights;
  }

  /**
   * @brief Enable gradient memory sharing based optimization
   * @param opt True to enable, else false
   */
  void setGradientMemoryOptimization(bool opt) {
    enable_gradient_memory_opt = opt;
  }

  /**
   * @brief Allocate and initialize the weight variable
   */
  void initialize() {
    for (auto &weight : weights)
      weight.get().initialize();
  }

  void reset() { weights.clear(); }

private:
  // TODO: ensure that names of these weights are unique
  std::vector<std::reference_wrapper<Weight>> weights;

  bool enable_gradient_memory_opt; /**< share memory among all the gradients */
};

// /**
//  * @brief Helper func for weight creation which are tracked by nntrainer
//  *
//  * @retval create weight
//  */
// template <typename... Args>
// Weight createWeight(Manager &manager, Args... args) {
//   Weight w = Weight(args...);
//   manager.trackWeight(std::forward<Args...>args);
//   return ;
// }

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
