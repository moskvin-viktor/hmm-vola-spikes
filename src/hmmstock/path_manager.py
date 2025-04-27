from pathlib import Path

class PathManager:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_base_path(self, model_name: str) -> Path:
        """Return the base path for a specific model."""
        model_path = self.models_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    def get_csvs_path(self, model_name: str) -> Path:
        """Return the csvs directory path for a model."""
        csvs_path = self.get_model_base_path(model_name) / "csvs"
        csvs_path.mkdir(parents=True, exist_ok=True)
        return csvs_path

    def get_ticket_csv_path(self, model_name: str, ticket: str) -> Path:
        """Return the path for a specific ticket inside csvs."""
        ticket_path = self.get_csvs_path(model_name) / ticket
        ticket_path.mkdir(parents=True, exist_ok=True)
        return ticket_path

    def get_logs_path(self, model_name: str) -> Path:
        """Return the logs directory path for a model."""
        logs_path = self.get_model_base_path(model_name) / "logs"
        logs_path.mkdir(parents=True, exist_ok=True)
        return logs_path

    def get_saved_models_path(self, model_name: str) -> Path:
        """Return the saved_models directory path for a model."""
        saved_models_path = self.get_model_base_path(model_name) / "saved_models"
        saved_models_path.mkdir(parents=True, exist_ok=True)
        return saved_models_path

    def get_ticket_csv_file(self, model_name: str, ticket: str, filename: str) -> Path:
        """Return the full path to a CSV file inside a ticket folder."""
        ticket_folder = self.get_ticket_csv_path(model_name, ticket)
        return ticket_folder / filename

    def get_log_file(self, model_name: str, log_filename: str) -> Path:
        """Return the full path to a log file inside logs."""
        logs_folder = self.get_logs_path(model_name)
        return logs_folder / log_filename

    def get_saved_model_file(self, model_name: str, model_filename: str) -> Path:
        """Return the full path to a saved model file."""
        saved_folder = self.get_saved_models_path(model_name)
        return saved_folder / model_filename