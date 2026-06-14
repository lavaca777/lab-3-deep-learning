"""Exercise placeholder for the classical PCA baseline."""


class ClassicalBaseline:
    """TODO(alumno): implement E1 with PCA and classical estimators.

    This is the only strategy that is not expected to use PyTorch for the
    estimator itself. It should still use exactly the same split manifest as
    the neural experiments and report gender and age metrics separately.

    Suggested ablations:
    - Number of PCA components.
    - Gender classifier: GaussianNB or LogisticRegression.
    - Age regressor: Ridge, LinearRegression or RandomForestRegressor.
    """

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "E1 baseline clasico no ha sido implementado. "
            "Complete src/baselines/classical_todo.py."
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError("TODO(alumno): implementar predict del baseline clasico.")
