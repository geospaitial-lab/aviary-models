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

#  ruff: noqa: D101, D102, D107, N806

import math
import warnings

import torch

warnings.filterwarnings('ignore', message='xFormers')


class PyramidRescaler(torch.nn.Module):

    def __init__(
        self,
        num_channels: int,
        scales: list[int],
    ) -> None:
        super().__init__()

        for scale in scales:
            assert math.log2(scale).is_integer()  # noqa: S101

        self.upscale_steps = [int(math.log2(scale)) for scale in scales]  # can be negative for downscaling
        self.rescale_operations = torch.nn.ModuleList()

        for n_steps in self.upscale_steps:
            step_operations = []

            if n_steps > 0:
                for _ in range(n_steps - 1):
                    step_operations.append(
                        torch.nn.ConvTranspose2d(
                            in_channels=num_channels,
                            out_channels=num_channels,
                            kernel_size=2,
                            stride=2,
                            bias=False,
                        ),
                    )
                    step_operations.append(torch.nn.BatchNorm2d(num_channels))
                    step_operations.append(torch.nn.GELU())

                step_operations.append(
                    torch.nn.ConvTranspose2d(
                        in_channels=num_channels,
                        out_channels=num_channels,
                        kernel_size=2,
                        stride=2,
                    ),
                )

            if n_steps == 0:
                step_operations.append(torch.nn.Identity())

            if n_steps < 0:
                for _ in range(-n_steps):
                    step_operations.append(  # noqa: PERF401
                        torch.nn.MaxPool2d(
                            kernel_size=2,
                            stride=2,
                        ),
                    )

            self.rescale_operations.append(torch.nn.Sequential(*step_operations))

    def forward(
        self,
        inputs: tuple[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            rescale_op(x)
            for rescale_op, x
            in zip(self.rescale_operations, inputs, strict=True)
        ]


class PPM(torch.nn.Module):

    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        pooling_scales: list[int],
        dropout_rate: float = .1,
    ) -> None:
        super().__init__()

        self.pooling_blocks = torch.nn.ModuleList()

        for scale in pooling_scales:
            self.pooling_blocks.append(torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(scale),
                torch.nn.Conv2d(
                    in_channels=num_channels_in,
                    out_channels=num_channels_out,
                    kernel_size=1,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(num_channels_out),
                torch.nn.ReLU(),
            ))

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_channels_in + len(pooling_scales) * num_channels_out,
                out_channels=num_channels_out,
                kernel_size=3,
                padding='same',
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_channels_out),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        pooled_x = [
            torch.nn.functional.interpolate(
                input=pooling_block(x),
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            for pooling_block in self.pooling_blocks
        ]

        return self.conv_block(torch.cat([x, *pooled_x], dim=1))


class FPNBlock(torch.nn.Module):

    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
    ) -> None:
        super().__init__()

        self.lateral_connection = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_channels_in,
                out_channels=num_channels_out,
                kernel_size=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_channels_out),
            torch.nn.ReLU(),
        )
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_channels_out,
                out_channels=num_channels_out,
                kernel_size=3,
                padding='same',
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_channels_out),
            torch.nn.ReLU(),
        )

    def forward(
        self,
        lateral_input: torch.Tensor,
        top_down_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lateral_connection = self.lateral_connection(lateral_input)
        top_down_connection = torch.nn.functional.interpolate(
            input=top_down_input,
            size=lateral_connection.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        top_down_output = lateral_connection + top_down_connection
        feature_map_output = self.conv_block(top_down_output)
        return top_down_output, feature_map_output


class FPN(torch.nn.Module):

    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        num_levels: int = 4,
        ppm_scales: list[int] | None = None,
        dropout_rate: float = .1,
    ) -> None:
        super().__init__()

        if ppm_scales is not None:
            self.top_path = PPM(
                num_channels_in=num_channels_in,
                num_channels_out=num_channels_out,
                pooling_scales=ppm_scales,
                dropout_rate=dropout_rate,
            )
        else:
            self.top_path = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=num_channels_in,
                    out_channels=num_channels_out,
                    kernel_size=1,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(num_channels_out),
                torch.nn.ReLU(),
            )

        self.fpn_blocks = torch.nn.ModuleList(
            [
                FPNBlock(
                    num_channels_in=num_channels_in,
                    num_channels_out=num_channels_out,
                )
                for _ in range(num_levels - 1)
            ],
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        outputs = []
        x = self.top_path(inputs[-1])
        outputs.append(x)

        for feature_map, fpn_block in zip(reversed(inputs[:-1]), self.fpn_blocks, strict=True):
            x, out_feature_map = fpn_block(feature_map, x)
            outputs.append(out_feature_map)

        return outputs


class FuseBlock(torch.nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        dropout_rate: float = .1,
    ) -> None:
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_channels * num_layers,
                out_channels=num_channels,
                kernel_size=3,
                padding='same',
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(dropout_rate),
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        target_size = inputs[-1].shape[2:]
        scaled_features = [inputs[-1]]

        for feature_map in reversed(inputs[:-1]):
            scaled_features.append(  # noqa: PERF401
                torch.nn.functional.interpolate(
                    input=feature_map,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                ),
            )

        return self.conv_block(torch.cat(scaled_features, dim=1))


class UPerNet(torch.nn.Module):

    def __init__(
        self,
        num_backbone_features: int,
        num_classes: int,
        intermediate_layers: list[int],
        pyramid_scales: list[int],
        num_channels_fpn: int,
        ppm_scales: list[int],
    ) -> None:
        super().__init__()

        self.intermediate_layers = intermediate_layers

        self.pyramid_rescaler = PyramidRescaler(num_channels=num_backbone_features, scales=pyramid_scales)
        self.fpn = FPN(
            num_channels_in=num_backbone_features,
            num_channels_out=num_channels_fpn,
            num_levels=len(intermediate_layers),
            ppm_scales=ppm_scales,
            dropout_rate=0.,
        )
        self.fuse_block = FuseBlock(
            num_channels=num_channels_fpn,
            num_layers=len(intermediate_layers),
            dropout_rate=0.,
        )

        self.class_head = torch.nn.Conv2d(num_channels_fpn, num_classes, kernel_size=1)

    def forward(
        self,
        inputs: tuple[torch.Tensor],
    ) -> torch.Tensor:
        feature_pyramid = self.pyramid_rescaler(inputs)
        fpn_features = self.fpn(feature_pyramid)
        fused_features = self.fuse_block(fpn_features)

        return self.class_head(fused_features)


class DINOUperNet(torch.nn.Module):

    def __init__(
        self,
        backbone_name: str,
        landcover_ckpt: dict,
        solar_ckpt: dict,
        landcover_out_name: str = 'sursentia_landcover',
        solar_out_name: str = 'sursentia_solar',
    ) -> None:
        super().__init__()

        self._backbone = torch.hub.load('facebookresearch/dinov2', backbone_name, verbose=False)

        self.intermediate_layers = None
        self._landcover_upernet = None

        if landcover_ckpt is not None:
            hyperparameters = landcover_ckpt['hyperparameters']
            self._landcover_upernet = UPerNet(
                num_backbone_features=self._backbone.num_features,
                num_classes=hyperparameters['num_classes'],
                intermediate_layers=hyperparameters['intermediate_layers'],
                pyramid_scales=hyperparameters['pyramid_scales'],
                num_channels_fpn=hyperparameters['num_channels_fpn'],
                ppm_scales=hyperparameters['ppm_scales'],
            )
            self._landcover_upernet.load_state_dict(landcover_ckpt['state_dict'])
            self.intermediate_layers = hyperparameters['intermediate_layers']

        self._solar_upernet = None
        if solar_ckpt is not None:
            hyperparameters = solar_ckpt['hyperparameters']
            self._solar_upernet = UPerNet(
                num_backbone_features=self._backbone.num_features,
                num_classes=hyperparameters['num_classes'],
                intermediate_layers=hyperparameters['intermediate_layers'],
                pyramid_scales=hyperparameters['pyramid_scales'],
                num_channels_fpn=hyperparameters['num_channels_fpn'],
                ppm_scales=hyperparameters['ppm_scales'],
            )
            self._solar_upernet.load_state_dict(solar_ckpt['state_dict'])
            self.intermediate_layers = hyperparameters['intermediate_layers']

        self._landcover_out_name = landcover_out_name
        self._solar_out_name = solar_out_name

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = inputs['tensor']
        _, _, H, W = x.shape

        out_dict = {}

        with torch.no_grad():
            features = self._backbone.get_intermediate_layers(
                x,
                n=self.intermediate_layers,
                reshape=True,
                norm=True,
                return_class_token=False,
            )

            if self._landcover_upernet is not None:
                landcover_logits = self._landcover_upernet(features)
                landcover_logits = torch.nn.functional.interpolate(
                    input=landcover_logits,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                )
                out_dict[self._landcover_out_name] = landcover_logits

            if self._solar_upernet is not None:
                solar_logits = self._solar_upernet(features)
                solar_logits = torch.nn.functional.interpolate(
                    input=solar_logits,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                )
                out_dict[self._solar_out_name] = solar_logits

        return out_dict
