# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for EgoBabyLLaVA."""

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
# Bracketed token strings, not credentials — silence ruff's secret heuristic.
DEFAULT_IMAGE_TOKEN = "<image>"  # noqa: S105
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"  # noqa: S105
DEFAULT_IM_START_TOKEN = "<im_start>"  # noqa: S105
DEFAULT_IM_END_TOKEN = "<im_end>"  # noqa: S105
IMAGE_PLACEHOLDER = "<image-placeholder>"
