from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict
from .metrics import LogLikelihoodWithEntropy
from .datamanager import default_split
import joblib

class HMMStockModel:
    def __init__(self, data_dict: Dict[str, pd.DataFrame], config_path: str,
                 evaluation_metric=None, train_test_splitter=None):
        self.cfg = OmegaConf.load(config_path)

        self.data_dict = data_dict
        self.max_components = self.cfg.get("max_components", 3)
        self.n_fits = self.cfg.get("n_fits", 100)
        self.random_seed = self.cfg.get("random_seed", 13)
        self.evaluation_metric = evaluation_metric() if evaluation_metric else LogLikelihoodWithEntropy()
        self.splitter = train_test_splitter or default_split
        self.hmm_config = self.cfg.hmm_config
        self.state_labeled_data = {}
        self.models = {}
        self.states = {}

    def _compute_log_returns(self, df):
        if "normalized_returns" in df.columns:
            return df["normalized_returns"].dropna().values.reshape(-1, 1)
        else:
            raise ValueError("Returns column missing in DataFrame.")

    def save_models(self, path):
        for ticker, model in self.models.items():
            joblib.dump(model, f"{path}/{ticker}_hmm.pkl")

    def load_models(self, path):
        for ticker in self.data_dict.keys():
            self.models[ticker] = joblib.load(f"{path}/{ticker}_hmm.pkl")
    
    def _fit_hmm(self, X_features):
        if len(X_features) < 20:
            return None

        best_overall_score = -np.inf
        best_model = None

        X_train, X_validate = self.splitter(X_features)
        np.random.seed(self.random_seed)

        for n_components in range(2, self.max_components + 1):
            for idx in range(self.n_fits):
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.hmm_config.covariance_type,
                    random_state=idx,
                    init_params=self.hmm_config.init_params,
                    n_iter=self.hmm_config.n_iter,
                    tol=self.hmm_config.tol
                )
                try:
                    model.fit(X_train)
                    base_score = self.evaluation_metric.evaluate(model, X_validate)

                    final_score = base_score  # You can include model.means_ separation weighting here
                    if final_score > best_overall_score:
                        best_model = model
                        best_overall_score = final_score

                except Exception as e:
                    print(f"Error training HMM (components={n_components}): {e}")
        return best_model

    def _relabel_states_by_volatility(self, model, X, original_states):
        state_vols = []
        for state in range(model.n_components):
            state_obs = X[original_states == state]
            vol = np.std(state_obs)
            state_vols.append((state, vol))

        sorted_states = sorted(state_vols, key=lambda x: x[1])
        state_map = {old: new for new, (old, _) in enumerate(sorted_states)}
        return np.vectorize(state_map.get)(original_states)

    def train_all(self):
        for ticker, df in self.data_dict.items():
            print(f"Training HMM for {ticker} (1 to {self.max_components} states)...")
            try:    
                X = self._compute_log_returns(df)
                best_model = self._fit_hmm(X)
                self.models[ticker] = best_model

                if best_model:
                    raw_states = best_model.predict(X)
                    relabeled = self._relabel_states_by_volatility(best_model, X, raw_states)
                    self.states[ticker] = relabeled
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    def _get_states(self):
        state_dict = {}
        for ticker, states in self.states.items():
            if states is not None:
                aligned_index = self.data_dict[ticker].index[-len(states):]
                state_dict[ticker] = pd.Series(states, index=aligned_index)
        return pd.DataFrame(state_dict)

    def generate_state_labeled_data(self):
        """
        Combines each ticker's original data with its HMM state labels into a single DataFrame.
        Stores results in self.state_labeled_data.
        """
        states_df = self._get_states()
        
        for ticker, df in self.data_dict.items():
            if ticker in states_df:
                state_series = states_df[ticker]
                state_series = pd.Series(states_df[ticker], index=df.index[-len(df):])
                aligned_states = state_series.rename("regime_state").to_frame()
                print(f"Aligning states for {ticker}...")
                print(aligned_states)
                print(df)
                merged = df.merge(aligned_states, left_index=True, right_index=True, how="left")
                self.state_labeled_data[ticker] = merged

    def get_transition_matrix(self, ticker):
        model = self.models.get(ticker)
        if model:
            return pd.DataFrame(model.transmat_,
                                index=[f"VS{i}" for i in range(model.n_components)],
                                columns=[f"VS{i}" for i in range(model.n_components)])
        else:
            print(f"No model found for {ticker}.")
            return None

    def expected_steps_before_change(self, ticker):
        model = self.models.get(ticker)
        if not model:
            print(f"No model for {ticker}.")
            return None

        transmat = model.transmat_
        diag = np.diag(transmat)
        expected_steps = 1 / (1 - diag + 1e-10)
        return pd.Series(expected_steps, index=[f"VS{i}" for i in range(model.n_components)],
                         name="ExpectedStepsInState")