def main():
    from omegaconf import OmegaConf
    from src.hmmstock import DataManager, RegimeModelManager, HMMModel, LayeredHMMModel, HierarchicalHMMModel

    # Load configuration
    config = OmegaConf.load("config/data.yaml")
    dm = DataManager(config)
    data = dm.get_data()

    # Initialize and train the model
    model = RegimeModelManager(data_dict=data, config_path="config/model.yaml", model_class=LayeredHMMModel)
    model.train_all()
    # model = RegimeModelManager(data_dict=data, config_path="config/model.yaml", model_class=HMMModel)
    # model.train_all()
    model = RegimeModelManager(data_dict=data, config_path="config/model.yaml", model_class=HierarchicalHMMModel)
    model.train_all()

    # Generate transition matrices and state-labeled data
    # tickers = ['AAPL', 'MSFT', '^GSPC', 'AMZN']
    # for ticker in tickers:
    #     print(f"Transition matrix for {ticker}:")
    #     model.get_transition_matrix(ticker)
    #     print("\n")

if __name__ == "__main__":
    main()