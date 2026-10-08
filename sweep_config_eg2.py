
# Continuous Bayesian Optimization Configuration

sweep_config = {
    "method": "bayes",
    "metric": {"name": "Linf_error", "goal": "minimize"},
    "parameters": {
        "f_alpha1": {"distribution": "uniform", "min": -0.026, "max":0.026 },
        "f_alpha2": {"distribution": "uniform", "min": -0.026, "max":0.026 },
        "f_alpha3": {"distribution": "uniform", "min": -0.026, "max":0.026 },
    },
}