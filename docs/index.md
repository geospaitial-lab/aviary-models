<style>
  @media screen and (min-width: 76.25em) {
    .md-sidebar--primary { visibility: hidden }
  }
  .md-sidebar--secondary { visibility: hidden }
</style>

aviary-models is a plugin package containing models for [aviary :material-arrow-top-right:][aviary].

| Name      | Description                         | Input Channels             | Docs        |
|:----------|:------------------------------------|:---------------------------|:------------|
| Sursentia | Predicts landcover and solar panels | R, G, B (0.1 to 0.5 m/px)  | [Sursentia] |

  [aviary]: https://github.com/geospaitial-lab/aviary
  [Sursentia]: api_reference/sursentia.md

---

## Installation

Each model has its own dependency group and additional dependencies.

=== "pip"

    ```
    pip install geospaitial-lab-aviary-models
    ```

=== "uv"

    ```
    uv pip install geospaitial-lab-aviary-models
    ```

Note that aviary and aviary-models require Python 3.10 or later.

---

## License

aviary-models is licensed under the [GPL-3.0 license :material-arrow-top-right:][GPL-3.0 license].

  [GPL-3.0 license]: https://github.com/geospaitial-lab/aviary-models/blob/main/LICENSE.md
