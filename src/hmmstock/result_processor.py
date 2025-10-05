import pandas as pd
from .models.markov_model import MarkovModel

class ModelResultProcessor:
    def __init__(self, model: MarkovModel, original_data: pd.DataFrame):
        self.model = model
        self.original_data = original_data

    def get_state_labeled_data(self) -> pd.DataFrame:
        states = self.model.predict_states()
        if states is None:
            return pd.DataFrame()

        if isinstance(states, pd.DataFrame):
            state_info = states
        else:
            state_info = pd.DataFrame(
                {f"regime_layer0": states},
                index=self.original_data.index[-len(states) :],
            )

        merged_df = self.original_data.merge(
            state_info, left_index=True, right_index=True, how="left"
        )
        return merged_df

    def get_transition_matrices(self) -> list[pd.DataFrame]:
        matrices = []
        models = []
        if hasattr(self.model, 'models') and self.model.models:
            models = self.model.models
        elif hasattr(self.model, 'model') and self.model.model:
            models = [self.model.model]
        
        if not models:
            return []

        for layer_idx, model in enumerate(models):
            if model is None:
                continue
            
            trans_df = pd.DataFrame(
                model.transmat_,
                index=[f"VS{layer_idx}_{i}" for i in range(model.n_components)],
                columns=[f"VS{layer_idx}_{i}" for i in range(model.n_components)],
            )
            matrices.append(trans_df)
        return matrices
