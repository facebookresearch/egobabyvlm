% Copyright (c) Meta Platforms, Inc. and affiliates.
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.

% Bridge: load (rgb, depth_raw), project, save (depth_uint16, rgb_undist).
inputs = load(input_mat);
[imgDepthProj, imgRgbUd] = project_depth_map(inputs.imgDepthRaw, inputs.imgRgb);
imgDepthProj = uint16(imgDepthProj * 1000.0);
save('-v7', output_mat, 'imgDepthProj', 'imgRgbUd');
