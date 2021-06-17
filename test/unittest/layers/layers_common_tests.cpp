// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file layer_common_tests.cpp
 * @date 15 June 2021
 * @brief Common test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <layers_common_tests.h>

#include <layer_devel.h>

constexpr unsigned SAMPLE_TRIES = 10;

LayerSementics::~LayerSementics() {}

void LayerSementics::SetUp() {
  auto f = std::get<0>(GetParam());
  layer = std::move(f({}));
  std::tie(std::ignore, expected_type, valid_properties, invalid_properties,
           options) = GetParam();
}

void LayerSementics::TearDown() {}

TEST_P(LayerSementics, createFromAppContext_pn) {}

TEST_P(LayerSementics, setProperties_p) {
  /// @todo check if setProperties does not collide with layerNode designated
  /// properties
}

TEST_P(LayerSementics, setPropertiesValidWithInvalid_n) {}

TEST_P(LayerSementics, setPropertiesValidInvalidOnly_n) {}

TEST_P(LayerSementics, finalizeTwice_p) {}

TEST_P(LayerGoldenTest, HelloWorld) { EXPECT_TRUE(true); }
