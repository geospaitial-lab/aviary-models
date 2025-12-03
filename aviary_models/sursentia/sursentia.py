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

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic
import torch
from aviary import (
    AviaryUserError,
    ChannelName,
    RasterChannel,
)
from aviary.tile import (
    NormalizeProcessor,
    StandardizeProcessor,
    register_tiles_processor,
)
from aviary.tile import (
    SequentialCompositeProcessor as SequentialCompositeTilesProcessor,
)
from aviary.vector import (
    MapFieldProcessor,
    register_vector_processor,
)
from aviary.vector import (
    SequentialCompositeProcessor as SequentialCompositeVectorProcessor,
)

from aviary_models.sursentia.model import DINOUperNet
from aviary_models.sursentia.sliding_window_inference import SlidingWindowInference

if TYPE_CHECKING:
    from aviary import (
        Tiles,
        Vector,
    )


class Device(Enum):
    """
    Attributes:
        CPU: CPU device
        GPU: GPU device
    """
    CPU = 'cpu'
    GPU = 'gpu'

    def to_torch(self) -> torch.Device:
        """Converts the device to the torch device.

        Returns:
            Torch device
        """
        mapping = {
            Device.CPU: torch.device('cpu'),
            Device.GPU: torch.device('cuda'),
        }
        return mapping[self]


class SursentiaVersion(Enum):
    """
    Attributes:
        V1_0: Version 1.0
    """
    V1_0 = '1.0'


class SursentiaConfig(pydantic.BaseModel):
    """Configuration for the `from_config` class method of `Sursentia`

    Create the configuration from a config file:
        - Use '1.0' instead of `SursentiaVersion.V1_0`
        - Use 'cpu' or 'gpu' instead of `Device.CPU` or `Device.GPU`
        - Use null instead of None
        - Use false or true instead of False or True

    Example:
        You can create the configuration from a config file.

        ``` yaml title="config.yaml"
        package: 'aviary_models'
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
        landcover_channel_name: Channel name of the landcover channel (if None, the landcover head of the model
            is not used) -
            defaults to 'sursentia_landcover'
        solar_channel_name: Channel name of the solar panels channel (if None, the solar head of the model
            is not used) -
            defaults to 'sursentia_solar'
        batch_size: Batch size of the sliding window inference -
            defaults to 1
        version: Version of the model (`V1_0`) -
            defaults to `SursentiaVersion.V1_0`
        device: Device to run the model on (`CPU` or `GPU`) -
            defaults to `Device.CPU`
        cache_dir_path: Path to the cache directory of the model -
            defaults to 'cache'
        remove_channels: If True, the channels are removed -
            defaults to True
    """
    r_channel_name: ChannelName | str = ChannelName.R
    g_channel_name: ChannelName | str = ChannelName.G
    b_channel_name: ChannelName | str = ChannelName.B
    landcover_channel_name: str | None = 'sursentia_landcover'
    solar_channel_name: str | None = 'sursentia_solar'
    batch_size: int = 1
    version: SursentiaVersion = SursentiaVersion.V1_0
    device: Device = Device.CPU
    cache_dir_path: Path = Path('cache')
    remove_channels: bool = True


@register_tiles_processor(config_class=SursentiaConfig)
class Sursentia:
    """Tiles processor that uses the Sursentia model to predict landcover and solar panels.

    Model input channels:
        - `ChannelName.R`: Red channel, raster channel, ground sampling distance of 0.1 to 0.5 meters per pixel,
            standardized values with a mean of 0.392 and a standard deviation of 0.198
        - `ChannelName.G`: Green channel, raster channel, ground sampling distance of 0.1 to 0.5 meters per pixel,
            standardized values with a mean of 0.423 and a standard deviation of 0.173
        - `ChannelName.B`: Blue channel, raster channel, ground sampling distance of 0.1 to 0.5 meters per pixel,
            standardized values with a mean of 0.373 and a standard deviation of 0.157
        - Use the `SursentiaPreprocessor` to preprocess the input channels

    Model output channels:
        - 'sursentia_landcover': Landcover channel, raster channel, ground sampling distance of the input channels,
            the values are 0 (buildings), 1 (buildings with green roofs), 2 (impervious surfaces),
            3 (non-impervious surfaces), and 4 (water bodies)
        - 'sursentia_solar': Solar panels channel, raster channel, ground sampling distance of the input channels,
            the values are 0 (background) and 1 (solar panels)
        - Use the `SursentiaMapFieldProcessor` to map the fields of the layers

    Additional dependencies:
        Sursentia requires the `sursentia` dependency group and `torch`.
        `xformers` is required for GPU inference.

    Implements the `TilesProcessor` protocol.
    """
    _HF_HUB_MODEL_PATHS = {  # noqa: RUF012
        SursentiaVersion.V1_0: {
            'landcover': 'models/v1_0/sursentia_landcover.ckpt',
            'solar': 'models/v1_0/sursentia_solar.ckpt',
        },
    }
    _HF_HUB_REPO = 'geospaitial-lab/sursentia'

    def __init__(  # noqa: PLR0915
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
        """
        Parameters:
            r_channel_name: Channel name of the red channel
            g_channel_name: Channel name of the green channel
            b_channel_name: Channel name of the blue channel
            landcover_channel_name: Channel name of the landcover channel (if None, the landcover head of the model
                is not used)
            solar_channel_name: Channel name of the solar panels channel (if None, the solar head of the model
                is not used)
            batch_size: Batch size of the sliding window inference
            version: Version of the model (`V1_0`)
            device: Device to run the model on (`CPU` or `GPU`)
            cache_dir_path: Path to cache directory of the model
            remove_channels: If True, the channels are removed
        """
        try:
            import torch  # noqa: PLC0415
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError as error:
            message = (
                'Missing dependencies! '
                'To use Sursentia, you need to install the Sursentia dependency group '
                '(pip install geospaitial-lab-aviary-models[sursentia]) and torch.'
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
        self._dtype = torch.float16
        self._cache_dir_path = cache_dir_path
        self._remove_channels = remove_channels

        if self._landcover_channel_name is None and self._solar_channel_name is None:
            message = (
                'Invalid landcover_channel_name / solar_channel_name! '
                'At least one of the channel names must be specified.'
            )
            raise AviaryUserError(message)

        if self._device.type == 'cpu':
            self._dtype = torch.float32
        else:
            try:
                import xformers  # noqa: F401, PLC0415
            except ImportError as error:
                message = (
                    'Missing dependencies! '
                    'To use Sursentia, you need to install the Sursentia dependency group '
                    '(pip install geospaitial-lab-aviary-models[sursentia]) and torch.'
                )
                raise ImportError(message) from error

        torch.hub.set_dir(self._cache_dir_path)

        if self._version not in self._HF_HUB_MODEL_PATHS:
            message = (
                'Invalid version! '
                'The version does not exist.'
            )
            raise AviaryUserError(message)

        hf_hub_model_paths = self._HF_HUB_MODEL_PATHS[self._version]

        landcover_ckpt = None

        if self._landcover_channel_name is not None:
            landcover_ckpt_path = hf_hub_download(
                repo_id=self._HF_HUB_REPO,
                filename=hf_hub_model_paths['landcover'],
                local_dir=self._cache_dir_path,
            )
            landcover_ckpt = torch.load(
                landcover_ckpt_path,
                map_location=self._device,
                weights_only=False,
            )

        solar_ckpt = None

        if self._solar_channel_name is not None:
            solar_ckpt_path = hf_hub_download(
                repo_id=self._HF_HUB_REPO,
                filename=hf_hub_model_paths['solar'],
                local_dir=self._cache_dir_path,
            )
            solar_ckpt = torch.load(
                solar_ckpt_path,
                map_location=self._device,
                weights_only=False,
            )

        ckpt = landcover_ckpt if landcover_ckpt is not None else solar_ckpt

        if ckpt is None:
            message = 'Invalid checkpoint!'
            raise AviaryUserError(message)

        backbone_name = ckpt['hyperparameters']['backbone_name']
        patch_size = ckpt['hyperparameters']['patch_size']

        self._model = DINOUperNet(
            backbone_name=backbone_name,
            landcover_ckpt=landcover_ckpt,
            solar_ckpt=solar_ckpt,
            landcover_out_name=self._landcover_channel_name,
            solar_out_name=self._solar_channel_name,
        )
        self._model.requires_grad_(requires_grad=False)
        self._model.eval()
        self._model.to(self._device)
        self._model.to(self._dtype)

        self._sliding_window_inference = SlidingWindowInference(
            window_size=patch_size,
            batch_size=self._batch_size,
            overlap=.5,
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
        inputs = {'tensor': inputs}

        logits_dict = self._sliding_window_inference(
            model=self._model,
            batch=inputs,
        )

        for channel_name, logits in logits_dict.items():
            preds = np.argmax(logits.cpu().numpy(), axis=1).astype(np.uint8)

            data = list(preds)
            buffer_size = tiles[channel_names[0]].buffer_size
            preds_channel = RasterChannel(
                data=data,
                name=channel_name,
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


class SursentiaPreprocessorConfig(pydantic.BaseModel):
    """Configuration for the `from_config` class method of `SursentiaPreprocessor`

    Create the configuration from a config file:
        - Use null instead of None

    Example:
        You can create the configuration from a config file.

        ``` yaml title="config.yaml"
        package: 'aviary_models'
        name: 'SursentiaPreprocessor'
        config:
          r_channel_name: 'r'
          g_channel_name: 'g'
          b_channel_name: 'b'
          new_r_channel_name: null
          new_g_channel_name: null
          new_b_channel_name: null
          max_num_threads: null
        ```

    Attributes:
        r_channel_name: Channel name of the red channel -
            defaults to `ChannelName.R`
        g_channel_name: Channel name of the green channel -
            defaults to `ChannelName.G`
        b_channel_name: Channel name of the blue channel -
            defaults to `ChannelName.B`
        new_r_channel_name: New channel name of the red channel -
            defaults to None
        new_g_channel_name: New channel name of the green channel -
            defaults to None
        new_b_channel_name: New channel name of the blue channel -
            defaults to None
        max_num_threads: Maximum number of threads -
            defaults to None
    """
    r_channel_name: ChannelName | str = ChannelName.R
    g_channel_name: ChannelName | str = ChannelName.G
    b_channel_name: ChannelName | str = ChannelName.B
    new_r_channel_name: ChannelName | str | None = None
    new_g_channel_name: ChannelName | str | None = None
    new_b_channel_name: ChannelName | str | None = None
    max_num_threads: int | None = None


@register_tiles_processor(config_class=SursentiaPreprocessorConfig)
class SursentiaPreprocessor:
    """Tiles processor that preprocesses the input channels of the Sursentia model.

    Implements the `TilesProcessor` protocol.
    """
    _MIN_VALUE = 0.
    _MAX_VALUE = 255.
    _R_MEAN_VALUE = .392
    _G_MEAN_VALUE = .423
    _B_MEAN_VALUE = .373
    _R_STD_VALUE = .198
    _G_STD_VALUE = .173
    _B_STD_VALUE = .157

    def __init__(
        self,
        r_channel_name: ChannelName | str = ChannelName.R,
        g_channel_name: ChannelName | str = ChannelName.G,
        b_channel_name: ChannelName | str = ChannelName.B,
        new_r_channel_name: ChannelName | str | None = None,
        new_g_channel_name: ChannelName | str | None = None,
        new_b_channel_name: ChannelName | str | None = None,
        max_num_threads: int | None = None,
    ) -> None:
        """
        Parameters:
            r_channel_name: Channel name of the red channel
            g_channel_name: Channel name of the green channel
            b_channel_name: Channel name of the blue channel
            new_r_channel_name: New channel name of the red channel
            new_g_channel_name: New channel name of the green channel
            new_b_channel_name: New channel name of the blue channel
            max_num_threads: Maximum number of threads
        """
        self._r_channel_name = r_channel_name
        self._g_channel_name = g_channel_name
        self._b_channel_name = b_channel_name
        self._new_r_channel_name = new_r_channel_name
        self._new_g_channel_name = new_g_channel_name
        self._new_b_channel_name = new_b_channel_name
        self._max_num_threads = max_num_threads

        self._r_normalize_processor = NormalizeProcessor(
            channel_name=self._r_channel_name,
            min_value=self._MIN_VALUE,
            max_value=self._MAX_VALUE,
            new_channel_name=self._new_r_channel_name,
            max_num_threads=self._max_num_threads,
        )
        self._g_normalize_processor = NormalizeProcessor(
            channel_name=self._g_channel_name,
            min_value=self._MIN_VALUE,
            max_value=self._MAX_VALUE,
            new_channel_name=self._new_g_channel_name,
            max_num_threads=self._max_num_threads,
        )
        self._b_normalize_processor = NormalizeProcessor(
            channel_name=self._b_channel_name,
            min_value=self._MIN_VALUE,
            max_value=self._MAX_VALUE,
            new_channel_name=self._new_b_channel_name,
            max_num_threads=self._max_num_threads,
        )
        self._r_standardize_processor = StandardizeProcessor(
            channel_name=self._new_r_channel_name if self._new_r_channel_name is not None else self._r_channel_name,
            mean_value=self._R_MEAN_VALUE,
            std_value=self._R_STD_VALUE,
            new_channel_name=self._new_r_channel_name,
            max_num_threads=self._max_num_threads,
        )
        self._g_standardize_processor = StandardizeProcessor(
            channel_name=self._new_g_channel_name if self._new_g_channel_name is not None else self._g_channel_name,
            mean_value=self._G_MEAN_VALUE,
            std_value=self._G_STD_VALUE,
            new_channel_name=self._new_g_channel_name,
            max_num_threads=self._max_num_threads,
        )
        self._b_standardize_processor = StandardizeProcessor(
            channel_name=self._new_b_channel_name if self._new_b_channel_name is not None else self._b_channel_name,
            mean_value=self._B_MEAN_VALUE,
            std_value=self._B_STD_VALUE,
            new_channel_name=self._new_b_channel_name,
            max_num_threads=self._max_num_threads,
        )
        self._sursentia_preprocessor = SequentialCompositeTilesProcessor(
            tiles_processors=[
                self._r_normalize_processor,
                self._g_normalize_processor,
                self._b_normalize_processor,
                self._r_standardize_processor,
                self._g_standardize_processor,
                self._b_standardize_processor,
            ],
        )

    @classmethod
    def from_config(
        cls,
        config: SursentiaPreprocessorConfig,
    ) -> SursentiaPreprocessor:
        """Creates a sursentia preprocessor from the configuration.

        Parameters:
            config: Configuration

        Returns:
            Sursentia preprocessor
        """
        config = config.model_dump()
        return cls(**config)

    def __call__(
        self,
        tiles: Tiles,
    ) -> Tiles:
        """Preprocesses the input channels.

        Parameters:
            tiles: Tiles

        Returns:
            Tiles
        """
        return self._sursentia_preprocessor(tiles=tiles)


class SursentiaMapFieldProcessorConfig(pydantic.BaseModel):
    """Configuration for the `from_config` class method of `SursentiaMapFieldProcessor`

    Create the configuration from a config file:
        - Use null instead of None

    Example:
        You can create the configuration from a config file.

        ``` yaml title="config.yaml"
        package: 'aviary_models'
        name: 'SursentiaMapFieldProcessor'
        config:
          field: 'my_field'
          landcover_layer_name: 'sursentia_landcover'
          solar_layer_name: 'sursentia_solar'
          new_landcover_layer_name: null
          new_solar_layer_name: null
        ```

    Attributes:
        field: Field
        landcover_layer_name: Layer name of the landcover layer (if None, the landcover layer is not used) -
            defaults to 'sursentia_landcover'
        solar_layer_name: Layer name of the solar layer (if None, the solar layer is not used) -
            defaults to 'sursentia_solar'
        new_landcover_layer_name: New layer name of the landcover layer -
            defaults to None
        new_solar_layer_name: New layer name of the solar layer -
            defaults to None
    """
    field: str
    landcover_layer_name: str | None = 'sursentia_landcover'
    solar_layer_name: str | None = 'sursentia_solar'
    new_landcover_layer_name: str | None = None
    new_solar_layer_name: str | None = None


@register_vector_processor(config_class=SursentiaMapFieldProcessorConfig)
class SursentiaMapFieldProcessor:
    """Vector processor that maps the fields of the layers of the Sursentia model.

    Landcover mapping:
        - 0: 'Gebäude'
        - 1: 'Gründach'
        - 2: 'versiegelte Fläche'
        - 3: 'nicht versiegelte Fläche'
        - 4: 'Gewässer'

    Solar mapping:
        - 0: 'Hintergrund'
        - 1: 'Solaranlage'

    Implements the `VectorProcessor` protocol.
    """
    _LANDCOVER_MAPPING = {  # noqa: RUF012
        0: 'Gebäude',
        1: 'Gründach',
        2: 'versiegelte Fläche',
        3: 'nicht versiegelte Fläche',
        4: 'Gewässer',
    }
    _SOLAR_MAPPING = {  # noqa: RUF012
        0: 'Hintergrund',
        1: 'Solaranlage',
    }

    def __init__(
        self,
        field: str,
        landcover_layer_name: str | None = 'sursentia_landcover',
        solar_layer_name: str | None = 'sursentia_solar',
        new_landcover_layer_name: str | None = None,
        new_solar_layer_name: str | None = None,
    ) -> None:
        """
        Parameters:
            field: Field
            landcover_layer_name: Layer name of the landcover layer (if None, the landcover layer is not used)
            solar_layer_name: Layer name of the solar layer (if None, the solar layer is not used)
            new_landcover_layer_name: New layer name of the landcover layer
            new_solar_layer_name: New layer name of the solar layer
        """
        self._field = field
        self._landcover_layer_name = landcover_layer_name
        self._solar_layer_name = solar_layer_name
        self._new_landcover_layer_name = new_landcover_layer_name
        self._new_solar_layer_name = new_solar_layer_name

        if self._landcover_layer_name is None and self._solar_layer_name is None:
            message = (
                'Invalid landcover_layer_name / solar_layer_name! '
                'At least one of the layer names must be specified.'
            )
            raise AviaryUserError(message)

        vector_processors = []

        if self._landcover_layer_name is not None:
            vector_processors.append(
                MapFieldProcessor(
                    layer_name=self._landcover_layer_name,
                    field=self._field,
                    mapping=self._LANDCOVER_MAPPING,
                    new_layer_name=self._new_landcover_layer_name,
                ),
            )

        if self._solar_layer_name is not None:
            vector_processors.append(
                MapFieldProcessor(
                    layer_name=self._solar_layer_name,
                    field=self._field,
                    mapping=self._SOLAR_MAPPING,
                    new_layer_name=self._new_solar_layer_name,
                ),
            )

        self._sursentia_map_field_processor = SequentialCompositeVectorProcessor(
            vector_processors=vector_processors,
        )

    @classmethod
    def from_config(
        cls,
        config: SursentiaMapFieldProcessorConfig,
    ) -> SursentiaMapFieldProcessor:
        """Creates a sursentia map field processor from the configuration.

        Parameters:
            config: Configuration

        Returns:
            Sursentia map field processor
        """
        config = config.model_dump()
        return cls(**config)

    def __call__(
        self,
        vector: Vector,
    ) -> Vector:
        """Maps the fields of the layers.

        Parameters:
            vector: Vector

        Returns:
            Vector
        """
        return self._sursentia_map_field_processor(vector=vector)
