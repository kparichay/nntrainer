// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_node.cpp
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer node for network graph
 */

#include <app_context.h>
#include <layer_factory.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <time_dist.h>

#include <common_properties.h>

namespace nntrainer {

LayerNode::LayerNode(std::shared_ptr<nntrainer::LayerV1> l, size_t idx) :
  LayerNode(nullptr, l, idx) {}

LayerNode::LayerNode(std::unique_ptr<nntrainer::Layer> &&l, size_t idx) :
  LayerNode(std::move(l), nullptr, idx) {}

LayerNode::LayerNode(std::unique_ptr<nntrainer::Layer> &&layer_v2,
                     std::shared_ptr<nntrainer::LayerV1> layer_v1, size_t idx) :
  layerv1(layer_v1),
  layer(std::move(layer_v2)),
  index(idx),
  finalized(false),
  flatten(false),
  distribute(false),
  activation_type(ActivationType::ACT_NONE),
  props(new std::tuple<props::Name, props::Trainable>(props::Name(),
                                                      props::Trainable())) {
  if (layerv1 && layerv1->getType() == TimeDistLayer::type) {
    distribute = true;
  } else if (layer && layer->getType() == TimeDistLayer::type) {
    distribute = true;
  }
}

/**
 * @brief Destroy the Layer Node object
 *
 */
LayerNode::~LayerNode() = default;

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(const std::string &type,
                const std::vector<std::string> &properties) {
  auto &ac = nntrainer::AppContext::Global();
  return createLayerNode(ac.createObject<nntrainer::LayerV1>(type), properties);
}

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(std::shared_ptr<nntrainer::LayerV1> layer,
                const std::vector<std::string> &properties) {
  auto lnode = std::make_unique<LayerNode>(layer);
  if (lnode->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Error setting layer properties.");

  return lnode;
}

int LayerNode::setProperty(std::vector<std::string> properties) {
  int status = ML_ERROR_NONE;

  properties = loadProperties(properties, *props.get());
  std::vector<std::string> remainder;

  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < properties.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(properties[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key);

    if (value.empty()) {
      ml_logd("value is empty for layer: %s, key: %s, value: %s",
              getName().c_str(), key.c_str(), value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(static_cast<nntrainer::LayerV1::PropertyType>(type), value);
    } catch (...) {
      remainder.push_back(properties[i]);
    }
  }

  status = getLayer()->setProperty(remainder);
  return status;
}

void LayerNode::setProperty(const nntrainer::LayerV1::PropertyType type,
                            const std::string &value) {
  int status = ML_ERROR_NONE;
  using PropertyType = nntrainer::LayerV1::PropertyType;

  switch (type) {
  case PropertyType::input_shape: {
    if (getNumInputs() > 1) {
      throw std::invalid_argument("input_shape keyword is only for one input");
    }

    if (getNumInputs() == 0) {
      input_dim.resize(1);
      // FIXME: input_layers information is currently unavailable
      // so input_dim and input_layers are inconsistent with this
      // fix this when debugging with layerV2
    }

    TensorDim &in_dim = input_dim[0];
    if (!value.empty()) {
      unsigned int cache_batch_size = 1;
      /** cache original value of batch size */
      if (in_dim.batch()) {
        cache_batch_size = in_dim.batch();
        in_dim.batch(1);
      }
      status = in_dim.setTensorDim(value.c_str());
      if (in_dim.batch() > 1) {
        ml_logw("Batch size set with input dimension %d is ignored."
                "Set batchsize property for the model to update batchsize.",
                in_dim.batch());
      }
      /** set back to cache value of dimension */
      in_dim.batch(cache_batch_size);
      throw_status(status);
    }
    // FIXME: this throws input_dim is still needed by LayerV1. It is in core
    // structure of LayerV1 and cant be directly moved to LayerV2 for now.
    // This throw allows setting input_dim for LayerV1 as well.
    throw std::invalid_argument("Setting input_dim for LayerV1 as well.");
  } break;
  case PropertyType::name:
    if (!value.empty()) {
      std::get<props::Name>(*props).set(value);
    }
    break;
  case PropertyType::flatten:
    if (!value.empty()) {
      status = setBoolean(flatten, value);
      throw_status(status);
    }
    break;
  case PropertyType::distribute:
    if (!value.empty()) {
      status = setBoolean(distribute, value);
      throw_status(status);
      if (distribute) {
        auto &ac = nntrainer::AppContext::Global();
        std::shared_ptr<nntrainer::LayerV1> dlayer =
          ac.createObject<nntrainer::LayerV1>(TimeDistLayer::type);
        std::static_pointer_cast<TimeDistLayer>(dlayer)->setDistLayer(layerv1);
        layerv1 = dlayer;
      }
    }
    break;
  case PropertyType::input_layers:
    if (!value.empty()) {
      static const std::regex reg("\\,+");
      std::vector<std::string> split_layers = split(value, reg);

      layerv1->setNumInputs(split_layers.size());
      input_layers = split_layers;
    }
    break;
  case PropertyType::output_layers:
    if (!value.empty()) {
      static const std::regex reg("\\,+");
      std::vector<std::string> split_layers = split(value, reg);

      layerv1->setNumOutputs(split_layers.size());
      output_layers = split_layers;
    }
    break;
  default:
    throw std::invalid_argument("Unknown property.");
  }
}

const std::string LayerNode::getName() const noexcept {
  auto &name = std::get<props::Name>(*props);
  return name.empty() ? "" : name.get();
}

std::ostream &operator<<(std::ostream &out, const LayerNode &l) {
  out << "[" << l.getName() << '/' << l.getType() << "]\n";
  auto print_vector = [&out](const std::vector<std::string> &layers,
                             const std::string &title) {
    out << title << "[" << layers.size() << "] ";
    for (auto &layer : layers) {
      out << layer << ' ';
    }
    out << '\n';
  };

  print_vector(l.input_layers, " input_layers");
  print_vector(l.output_layers, "output_layers");
  /// comment intended here,
  // print_vector(l.getObject()->input_layers, " input_layers");
  // print_vector(l.getObject()->output_layers, "output_layers");
  return out;
}

std::string LayerNode::getDistLayerType() const {
  if (distribute)
    return std::static_pointer_cast<TimeDistLayer>(layerv1)->getDistLayerType();
  else
    throw std::runtime_error(
      "Get distribution layer type for non-distributed layer");
}

ActivationType LayerNode::getActivationType() {
  return getLayer()->getActivationType();
}

const std::string LayerNode::getType() const { return getLayer()->getType(); }

std::shared_ptr<nntrainer::LayerV1> &LayerNode::getObject() {
  return getLayer();
}

const std::shared_ptr<nntrainer::LayerV1> &LayerNode::getObject() const {
  return getLayer();
}

bool LayerNode::getTrainable() const noexcept {
  return std::get<props::Trainable>(*props);
}

const std::shared_ptr<nntrainer::LayerV1> &LayerNode::getLayer() const {
  if (distribute)
    return std::static_pointer_cast<TimeDistLayer>(layerv1)->getDistLayer();
  else
    return layerv1;
}

std::shared_ptr<nntrainer::LayerV1> &LayerNode::getLayer() {
  if (distribute)
    return std::static_pointer_cast<TimeDistLayer>(layerv1)->getDistLayer();
  else
    return layerv1;
}

void LayerNode::updateInputLayers(const std::string &from,
                                  const std::string &to) {
  for (unsigned int idx = 0; idx < input_layers.size(); ++idx) {
    if (istrequal(input_layers[idx], from)) {
      input_layers[idx] = to;
    }
  }
}

void LayerNode::updateInputLayers(const unsigned int idx,
                                  const std::string &to) {
  if (idx >= input_layers.size())
    throw std::out_of_range("Out of range for input_layers");
  input_layers[idx] = to;
}

void LayerNode::exportTo(Exporter &exporter,
                         const ExportMethods &method) const {
  exporter.saveResult(*props, method, this);
  if (layerv1 == nullptr) {
    /// have layer_v2 implementation
  } else {
    layerv1->export_to(exporter, method);
  }
}

void LayerNode::read(std::ifstream &file) {
  if (layerv1 == nullptr) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      run_context.getWeight(i).read(file);
    }
  } else {
    getLayer()->read(file);
  }
}

void LayerNode::save(std::ofstream &file) const {
  if (layerv1 == nullptr) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      run_context.getWeight(i).save(file);
    }
  } else {
    getLayer()->save(file);
  }
}

/**
 * @brief     Finalize creating the layer node
 */
void LayerNode::finalize() {
  /** Create init context right before finalize */
  init_context = InitLayerContext(input_dim);
#if LAYER_V2
  layer->finalize(init_context);
#endif
  finalized = true;
}

/**
 * @brief     Forward Propagation of a layer
 */
void LayerNode::forwarding(bool training) {
  layer->forwarding(run_context, training);
}

/**
 * @brief     calc the derivative to be passed to the previous layer
 */
void LayerNode::calcDerivative() { layer->calcDerivative(run_context); }

/**
 * @brief     Calculate the derivative of a layer
 */
void LayerNode::calcGradient() { layer->calcGradient(run_context); }

/**
 * @brief Set the batch for the layer
 */
void LayerNode::setBatch(unsigned int batch) {
  if (finalized) {
    run_context.setBatch(batch);
    layer->setBatch(run_context, batch);
  } else {
    init_context.setBatch(batch);
    layer->setBatch(init_context, batch);
  }
}

/**
 * @brief   If the current layer can support in-place
 */
bool LayerNode::supportInPlace() const { return layer->supportInPlace(); }

/**
 * @brief  check if this layer requires label to be passed
 */
bool LayerNode::requireLabel() const { return layer->requireLabel(); }

}; // namespace nntrainer
