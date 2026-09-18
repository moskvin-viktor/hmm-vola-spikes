from pydantic import BaseModel, ConfigDict, model_validator


class LayerConfig(BaseModel):
    """Hyperparameter search space for one GaussianHMM layer."""

    model_config = ConfigDict(frozen=True)

    min_components: int
    max_components: int
    covariance_type: str
    init_params: str
    n_iter: int = 100


class HMMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    covariance_type: str
    random_seed: int
    init_params: str
    n_fits: int
    n_iter: int = 100
    tol: float
    max_components: int


class LayeredHMMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_layers: int
    n_fits: int
    random_seed: int
    tol: float
    layers: list[LayerConfig]

    @model_validator(mode="after")
    def _num_layers_matches_layers(self) -> "LayeredHMMConfig":
        if self.num_layers != len(self.layers):
            raise ValueError(
                f"num_layers ({self.num_layers}) does not match len(layers) "
                f"({len(self.layers)})"
            )
        return self


class HierarchicalHMMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_fits: int
    random_seed: int
    tol: float
    top_layer: LayerConfig
    sub_layer: LayerConfig
