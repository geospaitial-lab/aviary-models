<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://www.github.com/geospaitial-lab/aviary/raw/main/docs/assets/aviary_logo_white.svg">
  <img alt="aviary" src="https://www.github.com/geospaitial-lab/aviary/raw/main/docs/assets/aviary_logo_black.svg" width="30%">
</picture>

</div>

<div align="center">

[![CI][CI Badge]][CI]
[![Docs][Docs Badge]][Docs]

</div>

<div align="center">

[![PyPI version][PyPI version Badge]][PyPI]
[![Python version][Python version Badge]][PyPI]

</div>

<div align="center">

[![Chat][Chat Badge]][Chat]

</div>

  [CI Badge]: https://img.shields.io/github/actions/workflow/status/geospaitial-lab/aviary-models/ci.yaml?branch=main&color=black&label=CI&logo=GitHub
  [CI]: https://www.github.com/geospaitial-lab/aviary-models/actions/workflows/ci.yaml
  [Docs Badge]: https://img.shields.io/github/actions/workflow/status/geospaitial-lab/aviary-models/docs.yaml?branch=main&color=black&label=Docs&logo=materialformkdocs&logoColor=white
  [Docs]: https://geospaitial-lab.github.io/aviary-models
  [PyPI version Badge]: https://img.shields.io/pypi/v/geospaitial-lab-aviary-models?color=black&label=PyPI&logo=PyPI&logoColor=white
  [Python version Badge]: https://img.shields.io/pypi/pyversions/geospaitial-lab-aviary-models?color=black&label=Python&logo=Python&logoColor=white
  [PyPI]: https://www.pypi.org/project/geospaitial-lab-aviary-models
  [Chat Badge]: https://img.shields.io/matrix/geospaitial-lab-aviary%3Amatrix.org?color=black&label=Chat&logo=matrix
  [Chat]: https://matrix.to/#/#geospaitial-lab-aviary:matrix.org

aviary-models is a plugin package containing models for [aviary].

| Name      | Description                         | Input Channels             | Docs        |
|:----------|:------------------------------------|:---------------------------|:------------|
| Sursentia | Predicts landcover and solar panels | R, G, B (0.1 to 0.5 m/px)  | [Sursentia] |

  [aviary]: https://github.com/geospaitial-lab/aviary
  [Sursentia]: https://geospaitial-lab.github.io/aviary-models/api_reference/sursentia

---

## Installation

Each model has its own dependency group and additional dependencies.

### Installation with pip

```
pip install geospaitial-lab-aviary-models
```

Note that aviary and aviary-models require Python 3.10 or later.

### Installation with uv

```
uv pip install geospaitial-lab-aviary-models
```

Note that aviary and aviary-models require Python 3.10 or later.

---

## Documentation

The documentation is available at [geospaitial-lab.github.io/aviary-models].

  [geospaitial-lab.github.io/aviary-models]: https://geospaitial-lab.github.io/aviary-models

---

## License

aviary-models is licensed under the [GPL-3.0 license].

  [GPL-3.0 license]: LICENSE.md
