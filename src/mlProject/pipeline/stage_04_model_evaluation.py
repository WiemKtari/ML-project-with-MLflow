from mlProject.components.model_evaluation import ModelEvaluation
from mlProject.config.configuration import Configuration

def main():
    config = Configuration().get_model_evaluation_config()
    evaluator = ModelEvaluation(config)
    evaluator.log_into_mlflow()  # charge test set, prédit et log sur MLflow

if __name__ == "__main__":
    main()
