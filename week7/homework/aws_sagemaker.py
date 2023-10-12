from sagemaker.pytorch import PyTorch, TrainingCompilerConfig

from config import Config


hyperparameters={
    "n_gpus": 1,
    "batch_size": Config["batch_size"],
    "learning_rate": Config["learning_rate"],
}

pytorch_estimator=PyTorch(
    entry_point='main.py',
    role="arn:aws:iam::636571777167:role/sagemaker",
    # source_dir='path-to-requirements-file', # Optional. Add this if need to install additional packages.
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='1.13.1',
    py_version='py39',
    hyperparameters=hyperparameters,
    compiler_config=TrainingCompilerConfig(),
    disable_profiler=True,
    debugger_hook_config=False
)

pytorch_estimator.fit()