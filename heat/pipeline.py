"""Pipeline utilities for chaining Heat estimators."""

from __future__ import annotations

import heat as ht

__all__ = ["Pipeline", "make_pipeline"]


class Pipeline(ht.BaseEstimator):
    """Chain Heat estimators in sequence.

    Parameters
    ----------
    steps : list of tuple[str, ht.BaseEstimator]
        Named estimators in execution order. Intermediate steps must be
        Heat transformers. The final step may be a transformer,
        classifier, or regressor.

    Notes
    -----
    This implementation follows the scikit-learn pipeline pattern but
    respects Heat's distributed DNDarray conventions and the existing
    estimator mixins.
    """

    def __init__(self, steps: list[tuple[str, ht.BaseEstimator]]) -> None:
        self.steps = list(steps)
        self._validate_steps()
        self.named_steps = dict(self.steps)
        self._fitted = False

    def __repr__(self) -> str:
        """Return a concise representation of the pipeline."""
        step_names = [name for name, _ in self.steps]
        return f"{self.__class__.__name__}(steps={step_names})"

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("A pipeline needs at least one step.")

        names: list[str] = []
        for index, step in enumerate(self.steps):
            if not isinstance(step, (tuple, list)) or len(step) != 2:
                raise TypeError("Each step must be a (name, estimator) pair.")

            name, estimator = step

            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("Step names must be nonempty strings without '__'.")
            if name in names:
                raise ValueError(f"Duplicate step name: {name!r}.")
            names.append(name)

            if not isinstance(estimator, ht.BaseEstimator):
                raise TypeError(f"Step {name!r} is not a Heat estimator.")

            if index < len(self.steps) - 1:
                if not ht.is_transformer(estimator):
                    raise TypeError(f"Intermediate step {name!r} must be a Heat transformer.")
            elif not (
                ht.is_transformer(estimator)
                or ht.is_classifier(estimator)
                or ht.is_regressor(estimator)
            ):
                raise TypeError(
                    "The final step must be a Heat transformer, classifier, or regressor."
                )

    @staticmethod
    def _check_X(X: ht.DNDarray) -> None:
        if not isinstance(X, ht.DNDarray):
            raise TypeError("X must be a Heat DNDarray.")
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")

    @staticmethod
    def _check_y(X: ht.DNDarray, y: ht.DNDarray | None) -> None:
        if y is None:
            raise ValueError("Classification and regression require labels y.")
        if not isinstance(y, ht.DNDarray):
            raise TypeError("y must be a Heat DNDarray.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must have shape (n_samples,) matching X.")

    def _require(self, method: str) -> None:
        if not callable(getattr(self.steps[-1][1], method, None)):
            raise TypeError(f"The final step does not support {method}().")
        if not self._fitted:
            raise RuntimeError("Call fit() before using the pipeline.")

    def _transform_intermediate(self, X: ht.DNDarray) -> ht.DNDarray:
        self._check_X(X)
        for _, transformer in self.steps[:-1]:
            X = transformer.transform(X)
        return X

    def fit(self, X: ht.DNDarray, y: ht.DNDarray | None = None) -> "Pipeline":
        """Fit the pipeline.

        Parameters
        ----------
        X : ht.DNDarray
            Training features of shape (n_samples, n_features).
        y : ht.DNDarray or None, optional
            Labels for a supervised final estimator.

        Returns
        -------
        Pipeline
            The fitted pipeline.
        """
        self._fitted = False
        self._check_X(X)
        final = self.steps[-1][1]
        if ht.is_classifier(final) or ht.is_regressor(final):
            self._check_y(X, y)
        for _, transformer in self.steps[:-1]:
            X = transformer.fit_transform(X)

        if ht.is_classifier(final) or ht.is_regressor(final):
            final.fit(X, y)
        else:
            final.fit(X)
        self._fitted = True
        return self

    def predict(self, X: ht.DNDarray) -> ht.DNDarray:
        """Predict labels or targets for the provided samples."""
        self._require("predict")
        final = self.steps[-1][1]
        transformed = self._transform_intermediate(X)
        return final.predict(transformed)

    def score(self, X: ht.DNDarray, y: ht.DNDarray) -> float:
        """Return the score of the final estimator on transformed data."""
        self._check_X(X)
        self._check_y(X, y)
        final = self.steps[-1][1]
        if not callable(getattr(final, "score", None)):
            raise TypeError("The final step does not support score().")
        if not self._fitted:
            raise RuntimeError("Call fit() before using the pipeline.")
        transformed = self._transform_intermediate(X)
        return final.score(transformed, y)

    def transform(self, X: ht.DNDarray) -> ht.DNDarray:
        """Apply the pipeline to transform features."""
        self._require("transform")
        final = self.steps[-1][1]
        transformed = self._transform_intermediate(X)
        return final.transform(transformed)

    def fit_transform(self, X: ht.DNDarray, y: ht.DNDarray | None = None) -> ht.DNDarray:
        """Fit a transformer-only pipeline and return transformed data."""
        final = self.steps[-1][1]
        if not ht.is_transformer(final):
            raise TypeError("fit_transform() requires a final Heat transformer.")

        self._fitted = False
        self._check_X(X)
        for _, transformer in self.steps[:-1]:
            X = transformer.fit_transform(X)
        result = final.fit_transform(X)
        self._fitted = True
        return result


def make_pipeline(*estimators: ht.BaseEstimator) -> Pipeline:
    """Construct a pipeline with automatic step names.

    Parameters
    ----------
    *estimators : ht.BaseEstimator
        Transformers followed by a final transformer, classifier, or regressor.

    Returns
    -------
    Pipeline
        A pipeline with generated lowercase step names.
    """
    if not estimators:
        raise ValueError("At least one estimator is required to build a pipeline.")

    names = [type(estimator).__name__.lower() for estimator in estimators]
    occurrences: dict[str, int] = {}
    steps: list[tuple[str, ht.BaseEstimator]] = []
    for name, estimator in zip(names, estimators):
        occurrences[name] = occurrences.get(name, 0) + 1
        step_name = f"{name}-{occurrences[name]}" if names.count(name) > 1 else name
        steps.append((step_name, estimator))
    return Pipeline(steps)
