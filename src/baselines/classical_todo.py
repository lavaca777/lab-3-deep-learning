
from collections.abc import Sequence
from typing import Literal

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.data.parser import UTKFaceRecord

GenderModelName = Literal["gaussian_nb", "logistic"]
AgeModelName = Literal["linear", "ridge", "random_forest"]


class ClassicalBaseline:

    def __init__(
        self,
        image_size: int = 64,
        n_components: int = 100,
        gender_model: GenderModelName = "logistic",
        age_model: AgeModelName = "ridge",
        random_state: int = 42,
        ridge_alpha: float = 1.0,
        rf_n_estimators: int = 200,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size debe ser mayor que cero.")
        if n_components <= 0:
            raise ValueError("n_components debe ser mayor que cero.")

        self.image_size = image_size
        self.n_components = n_components
        self.gender_model = gender_model
        self.age_model = age_model
        self.random_state = random_state
        self.ridge_alpha = ridge_alpha
        self.rf_n_estimators = rf_n_estimators

        self.gender_pipeline: Pipeline | None = None
        self.age_pipeline: Pipeline | None = None
        self._is_fitted = False

    
    def fit(self, records: Sequence[UTKFaceRecord]) -> "ClassicalBaseline":

        features, genders, ages = self._build_matrices(records)

        effective_components = min(self.n_components, features.shape[0], features.shape[1])

        self.gender_pipeline = self._build_gender_pipeline(effective_components)
        self.age_pipeline = self._build_age_pipeline(effective_components)

        self.gender_pipeline.fit(features, genders)
        self.age_pipeline.fit(features, ages)
        self._is_fitted = True
        return self

    def predict(self, records: Sequence[UTKFaceRecord]) -> tuple[np.ndarray, np.ndarray]:
        
        if not self._is_fitted or self.gender_pipeline is None or self.age_pipeline is None:
            raise RuntimeError("Llame a fit() antes de predict().")

        features, _, _ = self._build_matrices(records)
        gender_predictions = self.gender_pipeline.predict(features)
        age_predictions = self.age_pipeline.predict(features)
        return gender_predictions, age_predictions

    def count_trainable_parameters(self) -> int:

        if self.gender_pipeline is None or self.age_pipeline is None:
            raise RuntimeError("Llame a fit() antes de contar parametros.")

        pca: PCA = self.gender_pipeline.named_steps["pca"]
        pca_parameters = pca.components_.size + pca.mean_.size

        gender_parameters = self._estimator_parameter_count(
            self.gender_pipeline.named_steps["classifier"]
        )
        age_parameters = self._estimator_parameter_count(
            self.age_pipeline.named_steps["regressor"]
        )
        return pca_parameters + gender_parameters + age_parameters

   
    def _build_gender_pipeline(self, n_components: int) -> Pipeline:
        if self.gender_model == "gaussian_nb":
            classifier = GaussianNB()
        elif self.gender_model == "logistic":
            classifier = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:
            raise ValueError(f"gender_model desconocido: {self.gender_model}")

        return Pipeline(
            steps=[
                (
                    "pca",
                    PCA(n_components=n_components, whiten=True, random_state=self.random_state),
                ),
                ("classifier", classifier),
            ]
        )

    def _build_age_pipeline(self, n_components: int) -> Pipeline:
        if self.age_model == "linear":
            regressor = LinearRegression()
        elif self.age_model == "ridge":
            regressor = Ridge(alpha=self.ridge_alpha)
        elif self.age_model == "random_forest":
            regressor = RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"age_model desconocido: {self.age_model}")

        return Pipeline(
            steps=[
                (
                    "pca",
                    PCA(n_components=n_components, whiten=True, random_state=self.random_state),
                ),
                ("regressor", regressor),
            ]
        )


    def _build_matrices(
        self, records: Sequence[UTKFaceRecord]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not records:
            raise ValueError("No se puede construir el baseline clasico sin registros.")

        vectors = np.stack([self._vectorize(record) for record in records])
        genders = np.array([record.gender for record in records], dtype=int)
        ages = np.array([record.age for record in records], dtype=float)
        return vectors, genders, ages

    def _vectorize(self, record: UTKFaceRecord) -> np.ndarray:
        with Image.open(record.path) as image_file:
            image = image_file.convert("L")  # escala de grises
            image = image.resize((self.image_size, self.image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0
        return array.flatten()

    @staticmethod
    def _estimator_parameter_count(estimator: object) -> int:
        if hasattr(estimator, "coef_"):
            count = np.asarray(estimator.coef_).size
            if hasattr(estimator, "intercept_"):
                count += np.asarray(estimator.intercept_).size
            return int(count)
        if hasattr(estimator, "theta_"):  # GaussianNB
            return int(np.asarray(estimator.theta_).size + np.asarray(estimator.var_).size)
        if hasattr(estimator, "estimators_"):  # RandomForestRegressor
            return sum(tree.tree_.node_count for tree in estimator.estimators_)
        return 0
