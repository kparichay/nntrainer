class TensorBase  {
  TensorBase () {
    allocateMemory();
  }
  ....
}

class TensorBlas : public TensorBase {
  TensorBlas () {
    allocateMemory();
  }
  ....
}

class TensorCuda : public TensorBase {
}

makeTensor (shape, Delegate) {
  std::shared_ptr<TensorBase> tensor;
  if (Delegate == blas)
    tensor = std::make_shared<TensorBlas> TensorBlas();
  .....

  return tensor;
}


class layer {
  init () {
    Tensor ();
  }

  set/get property

  std::shared_ptr<delegateLayer> delLayer;
  delegateProperties
}

class fc_layer : public layer {
  init (delegate) {
    Tensor ();
    switch(delegate) {
      delLayer = ....
    }
  }

  forward () {
    delLayer -> forward(delegateProperties);
  }
}

class delegate_layer {
}

class fc_layer_cpu : delegate_layer {
  forward(...., user_data);
  backward(....., user_data);
}

class fc_layer_blas : delegate_layer {
  forward();
  backward();
}

class custom_layer : public layer {
  init (forward_cb, backward_cb) {
    set these functions forward/backward cb
  }
}

class LayerFactory {
  create(type);
}

class Network {
  std::vector<layer>;
}
