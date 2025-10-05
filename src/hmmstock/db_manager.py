import pandas as pd
import logging
from pathlib import Path

from .database import get_engine, get_session_local, create_db_tables
from .database.models import RegimeState, TransitionMatrix, ModelResult
from .models.markov_model import MarkovModel

logger = logging.getLogger(__name__)


def sanitize_ticker(ticker: str) -> str:
    """Sanitize ticker symbol by removing/replacing unwanted characters."""
    return ticker.replace("^", "").replace("/", "_")


class DatabaseManager:
    def __init__(self):
        self._db_path = Path("results/hmm_data.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = get_engine(str(self._db_path))
        self._SessionLocal = get_session_local(self._engine)
        self._create_tables()

    def _create_tables(self):
        create_db_tables(self._engine)

    def save_regime_states(self, ticker: str, df: pd.DataFrame):
        session = self._SessionLocal()
        try:
            session.query(RegimeState).filter(RegimeState.ticker == ticker).delete()
            for index, row in df.iterrows():
                regime_state = RegimeState(
                    ticker=ticker,
                    date=str(index),
                    normalized_returns=row["normalized_returns"],
                    vol_2=row["vol_2"],
                    vol_3=row["vol_3"],
                    market_vola=row["market_vola"],
                    regime_layer0=row["regime_layer0"],
                )
                session.add(regime_state)
            session.commit()
            logger.info(f"Saved regime states for {ticker} to SQLite.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving regime states for {ticker}: {e}")
        finally:
            session.close()

    def load_regime_states(self, ticker: str) -> pd.DataFrame | None:
        session = self._SessionLocal()
        try:
            regime_states = (
                session.query(RegimeState)
                .filter(RegimeState.ticker == ticker)
                .order_by(RegimeState.date)
                .all()
            )
            if regime_states:
                data = [
                    {
                        "date": pd.to_datetime(rs.date),
                        "normalized_returns": rs.normalized_returns,
                        "vol_2": rs.vol_2,
                        "vol_3": rs.vol_3,
                        "market_vola": rs.market_vola,
                        "regime_layer0": rs.regime_layer0,
                    }
                    for rs in regime_states
                ]
                df = pd.DataFrame(data).set_index("date")
                return df
        except Exception as e:
            logger.error(f"Error loading regime states for {ticker}: {e}")
        finally:
            session.close()
        return None

    def save_transition_matrix(
        self, ticker: str, layer_idx: int, trans_df: pd.DataFrame
    ):
        session = self._SessionLocal()
        try:
            session.query(TransitionMatrix).filter(
                TransitionMatrix.ticker == ticker,
                TransitionMatrix.layer_idx == layer_idx,
            ).delete()
            for from_state, row in trans_df.iterrows():
                for to_state, probability in row.items():
                    transition_matrix_entry = TransitionMatrix(
                        ticker=ticker,
                        layer_idx=layer_idx,
                        from_state=from_state,
                        to_state=to_state,
                        probability=probability,
                    )
                    session.add(transition_matrix_entry)
            session.commit()
            logger.info(
                f"Saved transition matrix for {ticker} (Layer {layer_idx}) to SQLite."
            )
        except Exception as e:
            session.rollback()
            logger.error(
                f"Error saving transition matrix for {ticker} (Layer {layer_idx}): {e}"
            )
        finally:
            session.close()

    def load_transition_matrix(
        self, ticker: str, layer_idx: int
    ) -> pd.DataFrame | None:
        session = self._SessionLocal()
        try:
            transition_entries = (
                session.query(TransitionMatrix)
                .filter(
                    TransitionMatrix.ticker == ticker,
                    TransitionMatrix.layer_idx == layer_idx,
                )
                .all()
            )
            if transition_entries:
                data = [
                    {
                        "from_state": te.from_state,
                        "to_state": te.to_state,
                        "probability": te.probability,
                    }
                    for te in transition_entries
                ]
                df = pd.DataFrame(data)
                trans_df = df.pivot(
                    index="from_state", columns="to_state", values="probability"
                )
                return trans_df
        finally:
            session.close()
        return None

    def save_model_result(self, ticker: str, model: MarkovModel, model_name: str):
        session = self._SessionLocal()
        try:
            session.query(ModelResult).filter(
                ModelResult.ticker == ticker, ModelResult.model_name == model_name
            ).delete()
            model_result = ModelResult(
                ticker=ticker,
                model_name=model_name,
                n_components=model.model.n_components if model.model else 0,
                best_score=model.best_score,
                normalized_ll=(
                    model.normalized_ll if hasattr(model, "normalized_ll") else None
                ),
                entropy=model.entropy if hasattr(model, "entropy") else None,
                random_seed=(
                    model.random_state if hasattr(model, "random_state") else None
                ),
            )
            session.add(model_result)
            session.commit()
            logger.info(f"Saved model result for {ticker} to SQLite.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving model result for {ticker}: {e}")
        finally:
            session.close()

    def load_data(self, ticker: str) -> dict[str, pd.DataFrame] | None:
        ticker = sanitize_ticker(ticker)
        regime_states_df = self.load_regime_states(ticker)
        transition_matrix_df = self.load_transition_matrix(ticker, layer_idx=0)
        if regime_states_df is not None and transition_matrix_df is not None:
            return {
                "regime_states": regime_states_df,
                "transition_matrix": transition_matrix_df,
            }
        return None
