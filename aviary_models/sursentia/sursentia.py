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

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic
import torch
from aviary import AviaryUserError, RasterChannel, ChannelName
from aviary.tile import register_tiles_processor
from huggingface_hub import hf_hub_download

from aviary_models.sursentia.model import DINOUperNet
from aviary_models.sursentia.sliding_window_inference import SlidingWindowInference

if TYPE_CHECKING:
    from aviary import Tiles

_PACKAGE = 'aviary-models'


class Device(Enum):
    CPU = 'cpu'
    GPU = 'cuda'

    def to_torch(self) -> torch.Device:
        return torch.device(self.value)


class SursentiaVersion(Enum):
    V1_0 = '1.0'


class SursentiaConfig(pydantic.BaseModel):
    """Configuration for the `from_config` class method of `Sursentia`

    Create the configuration from a config file:
        - Use null instead of None
        - Use false or true instead of False or True

    Example:
        You can create the configuration from a config file.

        ``` yaml title="config.yaml"
        package: 'aviary-models'
        name: 'Sursentia'
        config:
          r_channel_name: 'r'
          g_channel_name: 'g'
          b_channel_name: 'b'
          landcover_channel_name: 'sursentia_landcover'
          solar_channel_name: 'sursentia_solar'
          batch_size: 1
          version: '1.0'
          device: 'cpu'
          cache_dir_path: 'cache'
          remove_channels: true
        ```

    Attributes:
        r_channel_name: Channel name of the red channel -
            defaults to `ChannelName.R`
        g_channel_name: Channel name of the green channel -
            defaults to `ChannelName.G`
        b_channel_name: Channel name of the blue channel -
            defaults to `ChannelName.B`
        landcover_channel_name: Channel name of the landcover channel -
            defaults to 'sursentia_landcover'
        solar_channel_name: Channel name of the solar channel -
            defaults to 'sursentia_solar'
        batch_size: Batch size for sliding window inference -
            defaults to 1
        version: Version of the model -
            defaults to '1.0'  # todo
        device: Device to run the model on -
            defaults to 'cpu'  # todo
        cache_dir_path: Path to the cache directory of the model -
            defaults to 'cache'
        remove_channels: If True, the channels are removed -
            defaults to True
    """
    r_channel_name: ChannelName | str = ChannelName.R
    g_channel_name: ChannelName | str = ChannelName.G
    b_channel_name: ChannelName | str = ChannelName.B
    landcover_channel_name: str = 'landcover'
    solar_channel_name: str = 'solar'
    batch_size: int = 1
    version: SursentiaVersion = SursentiaVersion.V1_0
    device: Device = Device.CPU
    cache_dir_path: Path = Path('cache')
    remove_channels: bool = True


@register_tiles_processor(config_class=SursentiaConfig)
class Sursentia:
    """TODO

    Implements the `TilesProcessor` protocol.
    """
    _HF_HUB_MODEL_PATHS = {
        SursentiaVersion.V1_0: {
            'landcover': 'models/v1_0/sursentia_landcover.ckpt',
            'solar': 'models/v1_0/sursentia_solar.ckpt',
        }
    }
    _HF_HUB_REPO = 'geospaitial-lab/sursentia'

    def __init__(
        self,
        r_channel_name: ChannelName | str = ChannelName.R,
        g_channel_name: ChannelName | str = ChannelName.G,
        b_channel_name: ChannelName | str = ChannelName.B,
        landcover_channel_name: str | None = 'sursentia_landcover',
        solar_channel_name: str | None = 'sursentia_solar',
        batch_size: int = 1,
        version: SursentiaVersion = SursentiaVersion.V1_0,
        device: Device = Device.CPU,
        cache_dir_path: Path = Path('cache'),
        remove_channels: bool = True,
    ) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError as error:
            message = (
                'Missing dependency! '
                'To use Sursentia, you need to install torch.'
            )
            raise ImportError(message) from error

        self._r_channel_name = r_channel_name
        self._g_channel_name = g_channel_name
        self._b_channel_name = b_channel_name
        self._landcover_channel_name = landcover_channel_name
        self._solar_channel_name = solar_channel_name
        self._batch_size = batch_size
        self._version = version
        self._device = device.to_torch()
        self._cache_dir_path = cache_dir_path
        self._remove_channels = remove_channels

        if self._device.type == "cpu":
            os.environ["XFORMERS_DISABLED"] = "1"
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

        torch.hub.set_dir(self._cache_dir_path)

        if self._version not in self._HF_HUB_MODEL_PATHS:
            message = (
                f'Unsupported version: {self._version}. '
                'Supported versions: '
                f'{list(self._HF_HUB_MODEL_PATHS.keys())}'
            )
            raise AviaryUserError(message)

        if self._landcover_channel_name is None and self._solar_channel_name is None:
            message = (
                'At least one of `landcover_channel_name` or `solar_channel_name` '
                'must be specified.'
            )
            raise AviaryUserError(message)

        self._hf_hub_model_paths = self._HF_HUB_MODEL_PATHS[self._version]

        self._landcover_ckpt = None
        if self._landcover_channel_name is not None:
            landcover_ckpt_path = hf_hub_download(
                repo_id=self._HF_HUB_REPO,
                filename=self._hf_hub_model_paths['landcover'],
                local_dir=self._cache_dir_path,
            )
            self._landcover_ckpt = torch.load(landcover_ckpt_path, weights_only=False)

        self._solar_ckpt = None
        if self._solar_channel_name is not None:
            solar_ckpt_path = hf_hub_download(
                repo_id=self._HF_HUB_REPO,
                filename=self._hf_hub_model_paths['solar'],
                local_dir=self._cache_dir_path,
            )
            self._solar_ckpt = torch.load(solar_ckpt_path, weights_only=False)

        if self._landcover_ckpt is not None and self._solar_ckpt is not None:
            assert self._landcover_ckpt['hyperparameters']['arch'] == self._solar_ckpt['hyperparameters']['arch']
            assert self._landcover_ckpt['hyperparameters']['backbone_name'] == self._solar_ckpt['hyperparameters']['backbone_name']
            assert self._landcover_ckpt['hyperparameters']['intermediate_layers'] == self._solar_ckpt['hyperparameters']['intermediate_layers']
            assert self._landcover_ckpt['hyperparameters']['pyramid_scales'] == self._solar_ckpt['hyperparameters']['pyramid_scales']
            assert self._landcover_ckpt['hyperparameters']['num_channels_fpn'] == self._solar_ckpt['hyperparameters']['num_channels_fpn']
            assert self._landcover_ckpt['hyperparameters']['ppm_scales'] == self._solar_ckpt['hyperparameters']['ppm_scales']
            assert self._landcover_ckpt['hyperparameters']['patch_size'] == self._solar_ckpt['hyperparameters']['patch_size']

            backbone_name = self._landcover_ckpt['hyperparameters']["backbone_name"]
            patch_size = self._landcover_ckpt['hyperparameters']['patch_size']
        elif self._landcover_ckpt is not None:
            backbone_name = self._landcover_ckpt['hyperparameters']["backbone_name"]
            patch_size = self._landcover_ckpt['hyperparameters']['patch_size']
        elif self._solar_ckpt is not None:
            backbone_name = self._solar_ckpt['hyperparameters']["backbone_name"]
            patch_size = self._solar_ckpt['hyperparameters']['patch_size']
        else:
            raise AviaryUserError("No checkpoint found")

        self._model = DINOUperNet(
            backbone_name=backbone_name,
            landcover_ckpt=self._landcover_ckpt,
            solar_ckpt=self._solar_ckpt,
            landcover_out_name=self._landcover_channel_name,
            solar_out_name=self._solar_channel_name,
        )
        self._model.requires_grad_(False)
        self._model.eval()
        self._model.to(self._device)
        self._model.to(self._dtype)
        # self._model.compile()

        self._sliding_window_inference = SlidingWindowInference(
            window_size=patch_size,
            batch_size=self._batch_size,
            model_receptive_field=None,
            overlap=0.5,
            downweight_edges=True,
        )

    @classmethod
    def from_config(
        cls,
        config: SursentiaConfig,
    ) -> Sursentia:
        """Creates the Sursentia model from the configuration.

        Parameters:
            config: Configuration

        Returns:
            Sursentia
        """
        config = config.model_dump()
        return cls(**config)

    def __call__(
        self,
        tiles: Tiles,
    ) -> Tiles:
        """Runs the Sursentia model.

        Parameters:
            tiles: Tiles

        Returns:
            Tiles
        """
        channel_names = [
            self._r_channel_name,
            self._g_channel_name,
            self._b_channel_name,
        ]
        inputs = tiles.to_composite_raster(channel_names=channel_names)
        inputs = torch.from_numpy(inputs).to(self._dtype).to(self._device).permute(0, 3, 1, 2)
        inputs = {"tensor": inputs}

        logits_dict = self._sliding_window_inference(self._model, inputs)

        for key, value in logits_dict.items():
            value = value.cpu().numpy()
            value = np.argmax(value, axis=1)
            value = value.astype(np.uint8)

            data = list(value)
            out_channel_name = key
            buffer_size = tiles[channel_names[0]].buffer_size
            preds_channel = RasterChannel(
                data=data,
                name=out_channel_name,
                buffer_size=buffer_size,
                copy=False,
            )

            tiles = tiles.append(
                channels=preds_channel,
                inplace=True,
            )

        if self._remove_channels:
            channel_names = set(channel_names)
            tiles = tiles.remove(
                channel_names=channel_names,
                inplace=True,
            )

        return tiles
