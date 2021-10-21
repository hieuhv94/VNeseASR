/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "models/asr/nn/Conv1d.h"
#include "models/asr/nn/Identity.h"
#include "models/asr/nn/LayerNorm.h"
#include "models/asr/nn/Linear.h"
#include "models/asr/nn/LocalNorm.h"
#include "models/asr/nn/Relu.h"
#include "models/asr/nn/Residual.h"
#include "models/asr/nn/Sequential.h"
#include "models/asr/nn/TDSBlock.h"

// We need to include the backend for the Cereal serirlization implementation.
#if W2L_INFERENCE_BACKEND == fbgemm
#include "models/asr/nn/backend/fbgemm/fbgemm.h"
#endif
