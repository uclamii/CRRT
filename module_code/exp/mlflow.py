import mlflow
from exp.cv import run_cv


def generate_mlFlowReport(
    run_name,
    model_name,
    dataset_name,
    cv_kwargs,
):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("Dataset used", dataset_name)
        results_dict = run_cv(**cv_kwargs)
        mlflow.log_metrics(results_dict['mean_scores'])
        mlflow.log_metrics(results_dict['std_scores'])

