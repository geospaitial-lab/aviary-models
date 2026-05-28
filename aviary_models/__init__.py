#  Copyright (C) 2025-2026 Marius Maryniak
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

import contextlib

from .sursentia.sursentia import (
    Device,
    Sursentia,
    SursentiaConfig,
    SursentiaMapFieldProcessor,
    SursentiaMapFieldProcessorConfig,
    SursentiaPreprocessor,
    SursentiaPreprocessorConfig,
    SursentiaVersion,
)

__all__ = [
    'Device',
    'Sursentia',
    'SursentiaConfig',
    'SursentiaMapFieldProcessor',
    'SursentiaMapFieldProcessorConfig',
    'SursentiaPreprocessor',
    'SursentiaPreprocessorConfig',
    'SursentiaVersion',
    '__version__',
]

for name in __all__:
    obj = globals().get(name)

    if obj and hasattr(obj, '__module__') and obj.__module__ != 'builtins':
        with contextlib.suppress(AttributeError, TypeError):
            obj.__module__ = __name__

__version__ = '0.2.0'
