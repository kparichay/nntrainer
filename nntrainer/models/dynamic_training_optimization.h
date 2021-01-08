// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   dynamic_training_optimization.h
 * @date   4 January 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Dynamic Training Optimization for Neural Network
 *
 */

#ifndef __DYNAMIC_TRAINING_OPT_H__
#define __DYNAMIC_TRAINING_OPT_H__
#ifdef __cplusplus

#include <random>
#include <vector>

#include <layer_internal.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

/**
 * @class   DynamicTraining Optimization
 * @brief   Dynamic Training Optimization
 */
class DynamicTrainingOptimization {
public:
  /**
   * @brief     Constructor of DynamicFineTuning Optimization
   */
  DynamicTrainingOptimization(int threshold_ = 1, int skip_n_iter = 1) :
    threshold(threshold_),
    enabled(false),
    epsilon(1e-7),
    skip_n_iterations(skip_n_iter) {
    reduce_op = reduceByNorm;
    calc_ratio_op = ratioUsingDerivative;
    rng.seed(getSeed());
    dist = std::uniform_real_distribution<float>(0.0, 1.0);
  }

  /**
   * @brief     Set threshold for optimization
   */
  void setThreshold(float threshold_) { threshold = threshold_; };

  float getThreshold() { return threshold; };

  /**
   * @brief     Set the reduce operation for dynamic optimization
   */
  void setOp(std::string op) {
    if (op == dft_opt_max)
      reduce_op = reduceByMax;
    else if (op == dft_opt_norm)
      reduce_op = reduceByNorm;
    else
      enabled = false;
  };

  /**
   * @brief     Enable the optimization
   */
  void enable() { enabled = true; }

  /**
   * @brief     Disable the optimization
   */
  void disable() { enabled = false; }

  /**
   * @brief     Set the mode for optimization
   */
  void setMode(std::string mode_) {
    calc_ratio_mode = mode_;
    if (mode_ == dft_opt_mode_derivative)
      calc_ratio_op = ratioUsingDerivative;
    else if (mode_ == dft_opt_mode_gradient)
      calc_ratio_op = ratioUsingGradient;
    else
      enabled = false;
  }

  /**
   * @brief     Check if the derivative mode is used for optimization
   */
  bool isDerivativeMode() {
    if (enabled && calc_ratio_mode == dft_opt_mode_derivative)
      return true;
    return false;
  }

  /**
   * @brief     Check if the gradient mode is used for optimization
   */
  bool isGradientMode() {
    if (enabled && calc_ratio_mode == dft_opt_mode_gradient)
      return true;
    return false;
  }

  /**
   * @brief     Set initial iterations to skip from optimization
   */
  void setSkipIterations(int skip_n_iter) { skip_n_iterations = skip_n_iter; }

  /**
   * @brief     Check if the given weights can skip updating
   * @note true if should be applied, else false
   */
  bool checkIfApply(const std::vector<Weight> &weights,
                    const std::shared_ptr<Var_Grad> input,
                    const std::shared_ptr<Var_Grad> output,
                    const std::shared_ptr<Optimizer> opt, int iteration);

  /**
   * @brief     Check if the given weight can skip updating
   * @note true if should be applied, else false
   */
  bool checkIfApply(const Weight &weight,
                    const std::shared_ptr<Var_Grad> &input,
                    const std::shared_ptr<Var_Grad> &output,
                    const std::shared_ptr<Optimizer> &opt, int iteration);

  /**< Different types of reduce operations */
  static const std::string dft_opt_max;
  static const std::string dft_opt_norm;

  /**< Different types of optimization modes */
  static const std::string dft_opt_mode_gradient;
  static const std::string dft_opt_mode_derivative;

private:
  std::mt19937 rng; /**< random number generator */
  std::uniform_real_distribution<float>
    dist;                      /**< uniform random distribution */
  float threshold;             /**< threshold to decide when to skip updating */
  bool enabled;                /**< if optimization is enabled */
  float epsilon;               /**< epsilon to skip overflow */
  int skip_n_iterations;       /**< skip initial iterations from optimization */
  std::string calc_ratio_mode; /**< the mode to calc the ratio */

  std::function<float(Tensor const &)>
    reduce_op; /**< operation to reduce update ratio to value */
  std::function<float(const Weight &, const std::shared_ptr<Var_Grad> &,
                      const std::shared_ptr<Var_Grad> &,
                      std::function<float(Tensor const &)> reduce_op)>
    calc_ratio_op; /**< calculate the ratio of update to the weight */

  /**
   * @brief   Calculate the ratio of update to the weight using derivative
   */
  static float
  ratioUsingDerivative(const Weight &weight,
                       const std::shared_ptr<Var_Grad> &input,
                       const std::shared_ptr<Var_Grad> &output,
                       std::function<float(Tensor const &)> reduce_op);

  /**
   * @brief   Calculate the ratio of update to the weight using gradient
   */
  static float
  ratioUsingGradient(const Weight &weight,
                     const std::shared_ptr<Var_Grad> &input,
                     const std::shared_ptr<Var_Grad> &output,
                     std::function<float(Tensor const &)> reduce_op);

  /**
   * @brief   Check if the update should be applied or skipped
   * @note true if should be applied, else false
   */
  bool checkIfApply(float reduced_ratio, float learning_rate);

  /**
   * @brief     Operation to decide if update should be skipped
   * @note      Calculate l0 norm of the tensor
   */
  static float reduceByMax(Tensor const &ratio);

  /**
   * @brief     Operation to decide if update should be skipped
   * @note      Calcalate l2 norm of the tensor averaged by its size
   */
  static float reduceByNorm(Tensor const &ratio);
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __DYNAMIC_TRAINING_OPT_H__ */
