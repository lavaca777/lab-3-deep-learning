
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import torch
from torch import nn, optim

from src.baselines.classical_todo import ClassicalBaseline
from src.config import AppConfig
from src.data.datamodule import UTKFaceDataModule
from src.evaluation.metrics import MultiTaskEvaluator, MultiTaskMetrics
from src.evaluation.plots import ResultPlotter
from src.evaluation.reporter import ExperimentResult, ExperimentStatus
from src.models.cnn import MultiTaskCNN
from src.models.resnet_todo import MultiTaskResNet
from src.training.losses import MultiTaskLoss
from src.training.trainer import MultiTaskTrainer
from src.utils import set_seed


@dataclass(frozen=True)
class ExperimentSpec:
    

    strategy_id: str
    strategy_name: str
    name: str
    variant: str
    changed_component: str
    implemented: bool
    model_kind: str
    use_augmentation: bool = True
    dropout: float = 0.4
    lambda_age: float = 0.01
    learning_rate: float = 1e-3
    
    classical_image_size: int = 64
    n_components: int = 100
    gender_model: str = "logistic"
    age_model: str = "ridge"


def build_experiment_catalog(config: AppConfig) -> dict[str, ExperimentSpec]:
    

    low_lambda = config.lambda_age / 10
    high_lambda = config.lambda_age * 10

    specs = [
        
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_base", "base", "ninguno", True, "classical",
            n_components=100, gender_model="logistic", age_model="ridge",
        ),
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_pca_50", "ablacion", "PCA=50 componentes", True, "classical",
            n_components=50, gender_model="logistic", age_model="ridge",
        ),
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_pca_200", "ablacion", "PCA=200 componentes", True, "classical",
            n_components=200, gender_model="logistic", age_model="ridge",
        ),
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_gaussian_nb", "ablacion", "clasificador=GaussianNB", True, "classical",
            n_components=100, gender_model="gaussian_nb", age_model="ridge",
        ),
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_age_random_forest", "ablacion", "regresor_edad=RandomForest", True, "classical",
            n_components=100, gender_model="logistic", age_model="random_forest",
        ),
        
        ExperimentSpec("E2", "MLP multitarea", "mlp_base", "base", "ninguno", False, "mlp"),
        ExperimentSpec("E2", "MLP multitarea", "mlp_no_dropout", "ablacion", "dropout=0.0", False, "mlp"),
        ExperimentSpec("E2", "MLP multitarea", "mlp_lambda_low", "ablacion", f"lambda_age={low_lambda:g}", False, "mlp"),
        ExperimentSpec("E2", "MLP multitarea", "mlp_lambda_high", "ablacion", f"lambda_age={high_lambda:g}", False, "mlp"),
        
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_base",
            "base",
            "ninguno",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_no_augmentation",
            "ablacion",
            "sin aumentacion",
            True,
            "cnn",
            use_augmentation=False,
            dropout=0.4,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_no_dropout",
            "ablacion",
            "dropout=0.0",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.0,
            lambda_age=config.lambda_age,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_lambda_low",
            "ablacion",
            f"lambda_age={low_lambda:g}",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=low_lambda,
            learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3",
            "CNN simple multitarea",
            "cnn_lambda_high",
            "ablacion",
            f"lambda_age={high_lambda:g}",
            True,
            "cnn",
            use_augmentation=True,
            dropout=0.4,
            lambda_age=high_lambda,
            learning_rate=config.learning_rate,
        ),
        # E4: frozen ResNet transfer learning exercises.
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_base", "base", "ninguno", True, "resnet_frozen"),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_no_augmentation", "ablacion", "sin aumentacion", True, "resnet_frozen"),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_lambda_low", "ablacion", f"lambda_age={low_lambda:g}", True, "resnet_frozen"),
        ExperimentSpec("E4", "ResNet18 congelada", "resnet_frozen_lambda_high", "ablacion", f"lambda_age={high_lambda:g}", True, "resnet_frozen"),
        # E5: fine-tuning exercises.
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_base", "base", "ninguno", True, "resnet_finetuning"),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_unfreeze_more", "ablacion", "mas bloques descongelados", True, "resnet_finetuning"),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_lr_low", "ablacion", "learning rate menor", True, "resnet_finetuning"),
        ExperimentSpec("E5", "ResNet18 fine-tuning", "resnet_finetuning_lambda_high", "ablacion", f"lambda_age={high_lambda:g}", True, "resnet_finetuning"),
    ]
    return {spec.name: spec for spec in specs}


class ExperimentRunner:
    

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        catalog: dict[str, ExperimentSpec],
    ) -> None:
        self.config = config
        self.device = device
        self.catalog = catalog
        self.plotter = ResultPlotter(config.plots_dir)

    def run(self, selected_names: set[str]) -> list[ExperimentResult]:
        unknown = selected_names.difference(self.catalog)
        if unknown:
            raise ValueError(f"Experimentos desconocidos: {', '.join(sorted(unknown))}")

        results: list[ExperimentResult] = []
        for spec in self.catalog.values():
            if not spec.implemented:
                results.append(self._not_implemented_result(spec))
            elif spec.name not in selected_names:
                results.append(self._not_executed_result(spec))
            else:
                results.append(self._run_spec(spec))

        for strategy_id in ("E1", "E2", "E3", "E4", "E5"):
            self.plotter.plot_ablation_comparison(results, strategy_id)
        return results

    def _run_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        if spec.model_kind == "classical":
            return self._run_classical_spec(spec)
        return self._run_neural_spec(spec)

    def _run_classical_spec(self, spec: ExperimentSpec) -> ExperimentResult:
       

        print(f"\nEjecutando {spec.name}: {spec.changed_component}")
        try:
            set_seed(self.config.seed)
            
            data_module = UTKFaceDataModule(self.config, use_augmentation=False)
            data_module.setup()

            baseline = ClassicalBaseline(
                image_size=spec.classical_image_size,
                n_components=spec.n_components,
                gender_model=spec.gender_model,
                age_model=spec.age_model,
                random_state=self.config.seed,
            )

            start_time = time.perf_counter()
            baseline.fit(self._require_records(data_module, "train"))
            training_seconds = time.perf_counter() - start_time

            test_records = self._require_records(data_module, "test")
            gender_predictions, age_predictions = baseline.predict(test_records)
            gender_targets = [record.gender for record in test_records]
            age_targets = [record.age for record in test_records]

            evaluation = MultiTaskMetrics.calculate(
                gender_targets=torch.tensor(gender_targets, dtype=torch.long),
                gender_predictions=torch.tensor(gender_predictions, dtype=torch.long),
                age_targets=torch.tensor(age_targets, dtype=torch.float32),
                age_predictions=torch.tensor(age_predictions, dtype=torch.float32),
            )

            self.plotter.plot_confusion_matrix(evaluation, spec.name)
            self.plotter.plot_age_predictions(evaluation, spec.name)

            checkpoint_path = self.config.checkpoints_dir / spec.name / "baseline.joblib"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "experiment_name": spec.name,
                    "image_size": spec.classical_image_size,
                    "gender_pipeline": baseline.gender_pipeline,
                    "age_pipeline": baseline.age_pipeline,
                },
                checkpoint_path,
            )

            sizes = data_module.split_sizes()
            metrics = dict(evaluation.metrics)
            metrics.update(
                {
                    "train_samples": sizes["train"],
                    "validation_samples": sizes["validation"],
                    "test_samples": sizes["test"],
                }
            )

            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.COMPLETED,
                metrics=metrics,
                trainable_parameters=baseline.count_trainable_parameters(),
                training_seconds=training_seconds,
                checkpoint=str(checkpoint_path),
                message="",
            )
        except Exception as error:
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    @staticmethod
    def _require_records(data_module: UTKFaceDataModule, split: str):
        dataset = {
            "train": data_module.train_dataset,
            "validation": data_module.val_dataset,
            "test": data_module.test_dataset,
        }[split]
        if dataset is None:
            raise RuntimeError(f"El dataset de {split} no esta listo. Ejecute setup() primero.")
        return dataset.records

    def _run_neural_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        print(f"\nEjecutando {spec.name}: {spec.changed_component}")
        try:
            set_seed(self.config.seed)
            data_module = UTKFaceDataModule(
                self.config,
                use_augmentation=spec.use_augmentation,
            )
            data_module.setup()

            model, model_kwargs = self._build_model(spec)
            model = model.to(self.device)
            optimizer = optim.Adam(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=spec.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            loss_function = MultiTaskLoss(lambda_age=spec.lambda_age)
            checkpoint_path = (
                self.config.checkpoints_dir / spec.name / "best_model.pt"
            )
            trainer = MultiTaskTrainer(
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
                device=self.device,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata={
                    "experiment_name": spec.name,
                    "strategy_id": spec.strategy_id,
                    "model_name": spec.model_kind,
                    "model_kwargs": model_kwargs,
                    "image_size": self.config.image_size,
                    "lambda_age": spec.lambda_age,
                },
            )
            history, training_seconds = trainer.fit(
                data_module.train_dataloader(),
                data_module.val_dataloader(),
                epochs=self.config.epochs,
            )
            trainer.load_best_checkpoint()

            evaluator = MultiTaskEvaluator(self.device)
            evaluation = evaluator.evaluate(model, data_module.test_dataloader())
            self.plotter.plot_training_history(history, spec.name)
            self.plotter.plot_confusion_matrix(evaluation, spec.name)
            self.plotter.plot_age_predictions(evaluation, spec.name)

            sizes = data_module.split_sizes()
            metrics = dict(evaluation.metrics)
            metrics.update(
                {
                    "train_samples": sizes["train"],
                    "validation_samples": sizes["validation"],
                    "test_samples": sizes["test"],
                }
            )
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.COMPLETED,
                metrics=metrics,
                trainable_parameters=self._count_trainable_parameters(model),
                training_seconds=training_seconds,
                checkpoint=str(checkpoint_path),
                message="",
            )
        except Exception as error:
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    @staticmethod
    @staticmethod
    def _build_model(spec: ExperimentSpec) -> tuple[nn.Module, dict[str, float]]:
        if spec.model_kind == "cnn":
            model_kwargs = {"dropout": spec.dropout}
            return MultiTaskCNN(**model_kwargs), model_kwargs

        if spec.model_kind == "resnet_frozen":
            # ResNet congelada: 0 bloques descongelados
            model_kwargs = {"num_unfrozen_blocks": 0, "dropout": spec.dropout}
            return MultiTaskResNet(**model_kwargs), model_kwargs
            
        if spec.model_kind == "resnet_finetuning":
            # Para fine-tuning descongelamos 1 bloque por defecto.
            # Si es el experimento 'unfreeze_more', descongelamos 2 bloques.
            num_unfrozen = 2 if spec.name == "resnet_finetuning_unfreeze_more" else 1
            model_kwargs = {"num_unfrozen_blocks": num_unfrozen, "dropout": spec.dropout}
            return MultiTaskResNet(**model_kwargs), model_kwargs

        raise NotImplementedError(f"No existe una fabrica para model_kind={spec.model_kind}.")

    @staticmethod
    def _count_trainable_parameters(model: nn.Module) -> int:
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _not_implemented_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_IMPLEMENTED,
            message="El experimento debe ser completado por los alumnos.",
        )

    @staticmethod
    def _not_executed_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_EXECUTED,
            message="Implementado, pero no fue seleccionado en esta ejecucion.",
        )
