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

#  ruff: noqa: D101, D102, D107, N803, N806

from math import ceil

import torch


class SlidingWindowInference:

    def __init__(
        self,
        window_size: int,
        batch_size: int,
        overlap: float = .5,
        downweight_edges: bool = True,
    ) -> None:
        self._window_size = window_size
        self._batch_size = batch_size
        self._overlap = overlap
        self._downweight_edges = downweight_edges

    @staticmethod
    def get_batch_stats(
        batch: dict[str, torch.Tensor],
    ) -> tuple[int, int, int, torch.device]:
        B = H = W = device = None

        for value in batch.values():
            if value.dim() == 4:  # noqa: PLR2004
                B, _, H, W = value.shape
                device = value.device
                break

        return B, H, W, device

    def get_sliding_window_params(
        self,
        device: torch.device,
    ) -> tuple[int, int, torch.Tensor]:
        kernel_size = self._window_size
        stride = round(self._window_size * self._overlap)
        patch_pixel_weights = torch.ones(
            size=(kernel_size, kernel_size),
            dtype=torch.float32,
            device=device,
        )

        if self._downweight_edges:
            indices = torch.stack(
                torch.meshgrid(
                    torch.arange(
                        kernel_size,
                        dtype=patch_pixel_weights.dtype,
                        device=patch_pixel_weights.device,
                    ),
                    torch.arange(
                        kernel_size,
                        dtype=patch_pixel_weights.dtype,
                        device=patch_pixel_weights.device,
                    ),
                    indexing='ij',
                ),
            )

            center_index = (kernel_size - 1) / 2
            distances = torch.maximum((indices[0] - center_index).abs(), (indices[1] - center_index).abs())
            patch_pixel_weights = (
                1 - (distances - distances.min()) / (distances.max() - distances.min()) * (1 - 1e-6)
            )

        return kernel_size, stride, patch_pixel_weights

    @staticmethod
    def align_sliding_window_params(
        H: int,
        W: int,
        kernel_size: int,
        init_stride: int,
    ) -> tuple[int, int, int, int, int]:
        n_patches_y = max(ceil((H - kernel_size) / init_stride + 1), 1)
        n_patches_x = max(ceil((W - kernel_size) / init_stride + 1), 1)
        stride_y = ceil((H - kernel_size) / (n_patches_y - 1)) if n_patches_y > 1 else 1
        stride_x = ceil((W - kernel_size) / (n_patches_x - 1)) if n_patches_x > 1 else 1

        stride = (stride_y, stride_x)

        padded_H = (n_patches_y - 1) * stride_y + kernel_size
        padded_W = (n_patches_x - 1) * stride_x + kernel_size

        return stride, padded_H, padded_W, n_patches_y, n_patches_x

    @staticmethod
    def make_patches(
        batch: dict[str, torch.Tensor],
        kernel_size: int,
        stride: int,
        n_patches_per_item: int,
        value_padding_H: int = 0,
        value_padding_W: int = 0,
    ) -> dict[str, torch.Tensor]:
        patched_batch = {}

        for key, value in batch.items():
            if value.dim() == 4:  # noqa: PLR2004
                B, channels, _, _ = value.shape
                dtype = value.dtype

                if dtype == torch.int32:
                    value = value.view(torch.float32)  # noqa: PLW2901

                value = torch.nn.functional.pad(  # noqa: PLW2901
                    input=value,
                    pad=(0, value_padding_W, 0, value_padding_H),
                )
                unfolded = torch.nn.functional.unfold(
                    input=value,
                    kernel_size=kernel_size,
                    stride=stride,
                )

                if dtype == torch.int32:
                    unfolded = unfolded.view(torch.int32)

                patches = unfolded.reshape(B, channels, kernel_size, kernel_size, n_patches_per_item).moveaxis(-1, 1)
                patches = patches.reshape(B * n_patches_per_item, channels, kernel_size, kernel_size)
            else:
                patches = value.repeat_interleave(n_patches_per_item, dim=0)

            patched_batch[key] = patches

        return patched_batch

    @staticmethod
    def reassemble_patches(
        preds: dict[str, torch.Tensor],
        patch_pixel_weights: torch.Tensor,
        kernel_size: int,
        stride: int,
        n_patches_per_item: int,
        B: int,
        padded_H: int,
        padded_W: int,
        H: int,
        W: int,
    ) -> dict[str, torch.Tensor]:
        pred = {}

        for key, value in preds.items():
            unfolded_value = value.reshape(B, n_patches_per_item, value.shape[1], kernel_size, kernel_size)
            unfolded_value = unfolded_value.moveaxis(1, -1)
            pixel_weights = patch_pixel_weights.reshape(1, 1, kernel_size, kernel_size, 1).expand_as(unfolded_value)
            pixel_weights = pixel_weights.reshape(B, value.shape[1] * kernel_size * kernel_size, n_patches_per_item)
            unfolded_value = unfolded_value.reshape(B, value.shape[1] * kernel_size * kernel_size, n_patches_per_item)

            unfolded_value = unfolded_value * pixel_weights
            divisor = torch.nn.functional.fold(
                input=pixel_weights,
                output_size=(padded_H, padded_W),
                kernel_size=kernel_size,
                stride=stride,
            )

            refolded_value = torch.nn.functional.fold(
                input=unfolded_value,
                output_size=(padded_H, padded_W),
                kernel_size=kernel_size,
                stride=stride,
            )

            final_value = refolded_value / divisor
            pred[key] = final_value[:, :, :H, :W]

        return pred

    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        B, H, W, device = self.get_batch_stats(batch)

        kernel_size, init_stride, patch_pixel_weights = self.get_sliding_window_params(device)

        stride, padded_H, padded_W, n_patches_y, n_patches_x = self.align_sliding_window_params(
            H=H,
            W=W,
            kernel_size=kernel_size,
            init_stride=init_stride,
        )
        value_padding_H = padded_H - H
        value_padding_W = padded_W - W

        n_patches_per_item = n_patches_x * n_patches_y
        n_batches = ceil((B * n_patches_per_item) / self._batch_size)

        patched_batch = self.make_patches(
            batch=batch,
            kernel_size=kernel_size,
            stride=stride,
            n_patches_per_item=n_patches_per_item,
            value_padding_H=value_padding_H,
            value_padding_W=value_padding_W,
        )

        chunked_patch_values = [
            torch.chunk(value, n_batches, dim=0)
            for value in patched_batch.values()
        ]
        chunks = [
            dict(zip(patched_batch.keys(), values, strict=True))
            for values in zip(*chunked_patch_values, strict=True)
        ]

        chunked_preds = []

        for chunk in chunks:
            chunked_preds.append(model(chunk))  # noqa: PERF401

        patched_preds = {
            key: torch.cat([pred[key] for pred in chunked_preds], dim=0)
            for key in chunked_preds[0]
        }

        return self.reassemble_patches(
            preds=patched_preds,
            patch_pixel_weights=patch_pixel_weights,
            kernel_size=kernel_size,
            stride=stride,
            n_patches_per_item=n_patches_per_item,
            B=B,
            padded_H=padded_H,
            padded_W=padded_W,
            H=H,
            W=W,
        )
