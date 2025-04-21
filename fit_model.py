def main():
 
    from omegaconf import OmegaConf
    from src.hmmstock import DataManager

    config = OmegaConf.load("config/data.yaml")
    dm = DataManager(config)
    data = dm.get_data()

    from src.hmmstock import HMMStockModel


    # data_short = {'AAPL' : data['AAPL'].query('Date > "2023-01-01"')}

    model = HMMStockModel(data_dict=data, config_path="config/model.yaml")
    model.train_all()

    for ticker in ['AAPL', 'MSFT', '^GSPC', 'AMZN']:
        print(f"Transition matrix for {ticker}:")
        model.get_transition_matrix(ticker)
        model.generate_state_labeled_data()
        print("\n")

if __name__ == "__main__":
    main()