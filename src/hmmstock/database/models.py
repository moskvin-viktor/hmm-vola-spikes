from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class RegimeState(Base):
    __tablename__ = "regime_states"

    ticker = Column(String, primary_key=True)
    date = Column(String, primary_key=True) # Store as string for simplicity, convert to datetime in app
    normalized_returns = Column(Float)
    vol_2 = Column(Float)
    vol_3 = Column(Float)
    market_vola = Column(Float)
    regime_layer0 = Column(Integer)

    def __repr__(self):
        return f"<RegimeState(ticker='{self.ticker}', date='{self.date}')>"

class TransitionMatrix(Base):
    __tablename__ = "transition_matrices"

    ticker = Column(String, primary_key=True)
    layer_idx = Column(Integer, primary_key=True)
    from_state = Column(String, primary_key=True)
    to_state = Column(String, primary_key=True)
    probability = Column(Float)

    def __repr__(self):
        return f"<TransitionMatrix(ticker='{self.ticker}', layer_idx={self.layer_idx}, from='{self.from_state}', to='{self.to_state}')>"

class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    n_components = Column(Integer, nullable=False)
    best_score = Column(Float, nullable=False)
    normalized_ll = Column(Float)
    entropy = Column(Float)
    random_seed = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelResult(ticker='{self.ticker}', model_name='{self.model_name}', n_components={self.n_components})>"