#  Copyright (C) 2025 Marius Maryniak
#  Copyright (C) 2025 Alexander Roß
#
#  This file is part of aviary-models.
#
#  aviary-models is free software: you can redistribute it and/or modify it under the terms of the
#  GNU General Public License as published by the Free Software Foundation,
#  either version 3 of the License, or (at your option) any later version.
#
#  aviary-models is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with aviary-models.
#  If not, see <https://www.gnu.org/licenses/>.

from math import ceil

import torch


class SlidingWindowInference:
    def __init__(self,
                 window_size,
                 batch_size,
                 model_receptive_field,
                 overlap=0.5,
                 downweight_edges=True):
        self.window_size = window_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.downweight_edges = downweight_edges
        self.model_receptive_field = model_receptive_field

        self.model_halve_n_times = 5

    @staticmethod
    def get_batch_stats(batch):
        B = H = W = device = None
        for value in batch.values():
            assert value.dim() <= 4
            if value.dim() == 4:
                B, _, H, W = value.shape
                device = value.device
                break
        assert B is not None
        assert H is not None
        assert W is not None
        assert device is not None

        return B, H, W, device

    def get_sliding_window_params(self, H, W, device):
        if self.model_receptive_field is not None:
            buffer = ((self.model_receptive_field + 2 ** self.model_halve_n_times - 1)
                      // 2 ** self.model_halve_n_times * 2 ** self.model_halve_n_times)
            kernel_size = self.window_size + buffer
            kernel_size = min(kernel_size, ((min(H, W) + 2 ** self.model_halve_n_times - 1)
                      // 2 ** self.model_halve_n_times * 2 ** self.model_halve_n_times))
            stride = self.window_size

            patch_pixel_weights = torch.ones((kernel_size, kernel_size), device=device, dtype=torch.float32)
            if self.downweight_edges:
                patch_pixel_weights[:buffer // 2, :] = 1e-6
                patch_pixel_weights[-buffer // 2:, :] = 1e-6
                patch_pixel_weights[:, :buffer // 2] = 1e-6
                patch_pixel_weights[:, -buffer // 2:] = 1e-6
        else:
            kernel_size = self.window_size
            stride = round(self.window_size * self.overlap)
            patch_pixel_weights = torch.ones((kernel_size, kernel_size), device=device, dtype=torch.float32)
            if self.downweight_edges:
                indices = torch.stack(torch.meshgrid(torch.arange(kernel_size,
                                                                  dtype=patch_pixel_weights.dtype,
                                                                  device=patch_pixel_weights.device),
                                                     torch.arange(kernel_size,
                                                                  dtype=patch_pixel_weights.dtype,
                                                                  device=patch_pixel_weights.device), indexing='ij'))
                center_index = (kernel_size - 1) / 2
                distances = torch.maximum((indices[0] - center_index).abs(), (indices[1] - center_index).abs())
                patch_pixel_weights = (
                            1 - (distances - distances.min()) / (distances.max() - distances.min()) * (1 - 1e-6))

        return kernel_size, stride, patch_pixel_weights

    @staticmethod
    def align_sliding_window_params(H, W, kernel_size, init_stride):
        n_patches_y = max(ceil((H - kernel_size) / init_stride + 1), 1)
        n_patches_x = max(ceil((W - kernel_size) / init_stride + 1), 1)
        stride_y = ceil((H - kernel_size) / (n_patches_y - 1)) if n_patches_y > 1 else 1
        stride_x = ceil((W - kernel_size) / (n_patches_x - 1)) if n_patches_x > 1 else 1

        assert stride_y <= init_stride
        assert stride_x <= init_stride

        stride = (stride_y, stride_x)

        padded_H = (n_patches_y - 1) * stride_y + kernel_size
        padded_W = (n_patches_x - 1) * stride_x + kernel_size

        assert padded_H >= H
        assert padded_W >= W

        return stride, padded_H, padded_W, n_patches_y, n_patches_x

    @staticmethod
    def make_patches(batch, kernel_size, stride, n_patches_per_item,
                     value_padding_H=0, value_padding_W=0):
        patched_batch = {}

        for key, value in batch.items():
            if value.dim() == 4:
                B, channels, _, _ = value.shape
                dtype = value.dtype
                if dtype == torch.int32:
                    value = value.view(torch.float32)
                value = torch.nn.functional.pad(value, (0, value_padding_W, 0, value_padding_H))
                unfolded = torch.nn.functional.unfold(value, kernel_size=kernel_size, stride=stride)
                if dtype == torch.int32:
                    unfolded = unfolded.view(torch.int32)
                patches = unfolded.reshape(B, channels, kernel_size, kernel_size, n_patches_per_item).moveaxis(-1, 1)
                patches = patches.reshape(B * n_patches_per_item, channels, kernel_size, kernel_size)
            else:
                patches = value.repeat_interleave(n_patches_per_item, dim=0)

            patched_batch[key] = patches

        return patched_batch

    @staticmethod
    def reassemble_patches(preds, patch_pixel_weights, kernel_size, stride, n_patches_per_item,
                           B, padded_H, padded_W, H, W):
        pred = {}
        for key, value in preds.items():
            assert value.dim() == 4
            unfolded_value = value.reshape(B, n_patches_per_item, value.shape[1], kernel_size, kernel_size)
            unfolded_value = unfolded_value.moveaxis(1, -1)
            pixel_weights = patch_pixel_weights.reshape(1, 1, kernel_size, kernel_size, 1).expand_as(unfolded_value)
            pixel_weights = pixel_weights.reshape(B, value.shape[1] * kernel_size * kernel_size, n_patches_per_item)
            unfolded_value = unfolded_value.reshape(B, value.shape[1] * kernel_size * kernel_size, n_patches_per_item)

            unfolded_value = unfolded_value * pixel_weights
            divisor = torch.nn.functional.fold(pixel_weights,
                                               output_size=(padded_H, padded_W),
                                               kernel_size=kernel_size,
                                               stride=stride)

            refolded_value = torch.nn.functional.fold(unfolded_value,
                                                      output_size=(padded_H, padded_W),
                                                      kernel_size=kernel_size,
                                                      stride=stride)

            final_value = refolded_value / divisor
            pred[key] = final_value[:, :, :H, :W]

        return pred

    def __call__(self, model, batch):
        B, H, W, device = self.get_batch_stats(batch)

        kernel_size, init_stride, patch_pixel_weights = self.get_sliding_window_params(H, W,device)

        stride, padded_H, padded_W, n_patches_y, n_patches_x = self.align_sliding_window_params(H,
                                                                                                W,
                                                                                                kernel_size,
                                                                                                init_stride)
        value_padding_H = padded_H - H
        value_padding_W = padded_W - W

        n_patches_per_item = n_patches_x * n_patches_y
        n_batches = ceil((B * n_patches_per_item) / self.batch_size)

        patched_batch = self.make_patches(batch, kernel_size, stride, n_patches_per_item,
                                          value_padding_H=value_padding_H, value_padding_W=value_padding_W)

        chunked_patch_values = [torch.chunk(value, n_batches, dim=0) for value in patched_batch.values()]
        chunks = [dict(zip(patched_batch.keys(), values)) for values in zip(*chunked_patch_values)]

        chunked_preds = []
        for chunk in chunks:
            chunked_preds.append(model(chunk))

        patched_preds = {key: torch.cat([pred[key] for pred in chunked_preds], dim=0) for key in chunked_preds[0].keys()}
        preds = self.reassemble_patches(patched_preds, patch_pixel_weights, kernel_size, stride, n_patches_per_item,
                                       B, padded_H, padded_W, H, W)

        return preds