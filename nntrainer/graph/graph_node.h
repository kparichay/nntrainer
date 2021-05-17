// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   graph_node.h
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the graph node interface for c++ API
 */

#ifndef __GRAPH_NODE_H__
#define __GRAPH_NODE_H__

#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace nntrainer {

/**
 * @class   Layer Base class for the graph node
 * @brief   Base class for all layers
 */
class GraphNode {
public:
  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~GraphNode() = default;

  /**
   * @brief     Get index of the node
   *
   */
  virtual size_t getIndex() = 0;

  /**
   * @brief     Set index of the node
   *
   */
  virtual void setIndex(size_t) = 0;

  /**
   * @brief     Get the Name of the underlying object
   *
   * @return std::string Name of the underlying object
   * @note name of each node in the graph must be unique
   */
  virtual std::string getName() noexcept = 0;

  /**
   * @brief     Set the Name of the underlying object
   *
   * @param[in] std::string Name for the underlying object
   * @note name of each node in the graph must be unique, and caller must ensure
   * that
   */
  virtual int setName(const std::string &name) = 0;

  /**
   * @brief     Get the Type of the underlying object
   *
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

  /**
   * @brief     Copy the graph
   * @param[in] from Graph Object to copy
   * @retval    Graph Object copyed
   */
  // virtual GraphNode &copy(const GraphNode &from) = 0;
};

/**
 * @brief   Iterator for GraphNode which return LayerNode object upon realize
 *
 * @note    This does not include the complete list of required functions. Add
 * them as per need.
 */
template <typename T>
class GraphNodeIterator : public std::iterator<std::random_access_iterator_tag,
                                               std::shared_ptr<GraphNode>> {
  const std::shared_ptr<GraphNode> *p; /** underlying object of GraphNode */

public:
  /**
   * @brief   iterator_traits types definition
   *
   * @note    these are not requried to be explicitly defined now, but maintains
   *          forward compatibility for c++17 and later
   *
   * @note    valute_type, pointer and reference are different from standard
   * iterator
   */
  typedef std::shared_ptr<T> value_type;
  typedef std::random_access_iterator_tag iterator_category;
  typedef std::ptrdiff_t difference_type;
  typedef std::shared_ptr<T> *pointer;
  typedef std::shared_ptr<T> &reference;

  /**
   * @brief Construct a new Graph Node Iterator object
   *
   * @param x underlying object of GraphNode
   */
  GraphNodeIterator(const std::shared_ptr<GraphNode> *x) : p(x) {}

  /**
   * @brief reference operator
   *
   * @return value_type
   * @note this is different from standard iterator
   */
  value_type operator*() const { return std::static_pointer_cast<T>(*p); }

  /**
   * @brief pointer operator
   *
   * @return value_type
   * @note this is different from standard iterator
   */
  value_type operator->() const { return std::static_pointer_cast<T>(*p); }

  /**
   * @brief == comparison operator override
   *
   * @param lhs iterator lhs
   * @param rhs iterator rhs
   * @return true if match
   * @return false if mismatch
   */
  friend bool operator==(GraphNodeIterator const &lhs,
                         GraphNodeIterator const &rhs) {
    return lhs.p == rhs.p;
  }

  /**
   * @brief != comparison operator override
   *
   * @param lhs iterator lhs
   * @param rhs iterator rhs
   * @return true if mismatch
   * @return false if match
   */
  friend bool operator!=(GraphNodeIterator const &lhs,
                         GraphNodeIterator const &rhs) {
    return lhs.p != rhs.p;
  }

  /**
   * @brief override for ++ operator
   *
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator++() {
    p += 1;
    return *this;
  }

  /**
   * @brief override for operator++
   *
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator++(int) {
    GraphNodeIterator temp(p);
    p += 1;
    return temp;
  }

  /**
   * @brief override for -- operator
   *
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator--() {
    p -= 1;
    return *this;
  }

  /**
   * @brief override for operator--
   *
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator--(int) {
    GraphNodeIterator temp(p);
    p -= 1;
    return temp;
  }

  /**
   * @brief override for subtract operator
   *
   * @param offset offset to subtract
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator-(const difference_type offset) const {
    return GraphNodeIterator(p - offset);
  }

  /**
   * @brief override for subtract operator
   *
   * @param other iterator to subtract
   * @return difference_type
   */
  difference_type operator-(const GraphNodeIterator &other) const {
    return p - other.p;
  }

  /**
   * @brief override for subtract and return result operator
   *
   * @param offset offset to subtract
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator-=(const difference_type offset) {
    p -= offset;
    return *this;
  }

  /**
   * @brief override for add operator
   *
   * @param offset offset to add
   * @return GraphNodeIterator
   */
  GraphNodeIterator operator+(const difference_type offset) const {
    return GraphNodeIterator(p + offset);
  }

  /**
   * @brief override for add and return result operator
   *
   * @param offset offset to add
   * @return GraphNodeIterator&
   */
  GraphNodeIterator &operator+=(const difference_type offset) {
    p += offset;
    return *this;
  }
};

/**
 * @brief   Reverse Iterator for GraphNode which return LayerNode object upon
 * realize
 *
 * @note    This just extends GraphNodeIterator and is limited by its
 * functionality.
 */
template <typename T_iterator>
class GraphNodeReverseIterator : public std::reverse_iterator<T_iterator> {
public:
  /**
   * @brief Construct a new Graph Node Reverse Iterator object
   *
   * @param iter Iterator
   */
  explicit GraphNodeReverseIterator(T_iterator iter) :
    std::reverse_iterator<T_iterator>(iter) {}

  /**
   *  @brief reference operator
   *
   * @return T_iterator::value_type
   * @note this is different from standard iterator
   */
  typename T_iterator::value_type operator*() const {
    auto temp = std::reverse_iterator<T_iterator>::current - 1;
    return *temp;
  }

  /**
   *  @brief pointer operator
   *
   * @return T_iterator::value_type
   * @note this is different from standard iterator
   */
  typename T_iterator::value_type operator->() const {
    auto temp = std::reverse_iterator<T_iterator>::current - 1;
    return *temp;
  }
};

/**
 * @brief     Iterators to traverse the graph
 */
template <class T> using graph_iterator = GraphNodeIterator<T>;

/**
 * @brief     Iterators to traverse the graph
 */
template <class T>
using graph_reverse_iterator = GraphNodeReverseIterator<GraphNodeIterator<T>>;

} // namespace nntrainer
#endif // __GRAPH_NODE_H__
