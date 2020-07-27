// V1

class Node {
  inputLayers[];
  outputLayers[];
  outputType(broadcast, 1 to 1)
}

class Tensor;

class layer : Node;

class model {

  addLayer();
  removeLayer();

  graph <layer>;
}

// V2
class Node {
  inputLayers[];
  outputLayers[];
  outputType(broadcast, 1 to 1)
}

globalDictionaryCompute {opName, computeFunc}

class ops {
  list of input shared_ptr tensors;
  list of output shared_ptr tensors;
  virtual compute() {
  }
  type;
}

class Tensor {
  // this is lazy tensor itself. Tensor not exposed outside.
  Tensor() {
    set shape, and info;
    no allocation of memory;
  }

  init() {
    allocateMemory();
  }

  run() {
    actually runs;
  }

  std::shared_ptr<ops> multiply;
  std::shared_ptr<ops> multiply_i;
  std::vector<std::shared_ptr<ops>> operations;
}

class layer {
  constructor(list of ops - in_ops) {
    create output tensors;
    create weight tensors;
    std::make_shared<ops> layer_forward(in_ops.tensors, output tensors, function, layer_forward);

    create derivative tensors
    create gradient tensors;
    std::make_shared<ops> layer_backward(gradient.tensors, output tensors, function, layer_forward);
  }

  // case 1
  forward() {
    Tensor out = in.multiply(w);
  }

  // case 2
  ops forward;
  ops backward;
}

class model {

  addLayer();
  removeLayer();

  graph <ops>;
}
