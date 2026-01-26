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
    ) -> tuple[tuple[int, int], int, int]:
        n_patches_y = max(ceil((H - kernel_size) / init_stride + 1), 1)
        n_patches_x = max(ceil((W - kernel_size) / init_stride + 1), 1)
        stride_y = ceil((H - kernel_size) / (n_patches_y - 1)) if n_patches_y > 1 else 1
        stride_x = ceil((W - kernel_size) / (n_patches_x - 1)) if n_patches_x > 1 else 1

        stride = (stride_y, stride_x)

        return stride, n_patches_y, n_patches_x

    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        B, H, W, device = self.get_batch_stats(batch)

        kernel_size, init_stride, patch_pixel_weights = self.get_sliding_window_params(device)

        stride, n_patches_y, n_patches_x = self.align_sliding_window_params(
            H=H,
            W=W,
            kernel_size=kernel_size,
            init_stride=init_stride,
        )

        preds = {}
        accumulated_pixel_weights = torch.zeros(
            size=(H, W),
            dtype=patch_pixel_weights.dtype,
            device=device,
        )

        for y in range(n_patches_y):
            for x in range(n_patches_x):
                start_y = y * stride[0]
                start_x = x * stride[1]
                end_y = start_y + kernel_size
                end_x = start_x + kernel_size

                patch = {}
                padding_x = 0
                padding_y = 0

                for key, value in batch.items():
                    if value.dim() == 4:  # noqa: PLR2004
                        patch_value = value[:, :, start_y:end_y, start_x:end_x]
                        padding_x = kernel_size - patch_value.shape[3]
                        padding_y = kernel_size - patch_value.shape[2]
                        patch_value = torch.nn.functional.pad(patch_value, (0, padding_x, 0, padding_y))
                        patch[key] = patch_value
                    else:
                        patch[key] = value

                pred = model(patch)

                for key, value in pred.items():
                    if key not in preds:
                        preds[key] = torch.zeros(
                            size=(B, value.shape[1], H, W),
                            dtype=value.dtype,
                            device=device,
                        )

                    patch_pred = value * patch_pixel_weights

                    preds[key][:, :, start_y:end_y, start_x:end_x] += (
                        patch_pred[:, :, :kernel_size-padding_y, :kernel_size-padding_x]
                    )
                    accumulated_pixel_weights[start_y:end_y, start_x:end_x] += (
                        patch_pixel_weights[:kernel_size-padding_y, :kernel_size-padding_x]
                    )

        for key, value in preds.items():
            preds[key] = value / accumulated_pixel_weights

        return preds
