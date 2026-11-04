"""Heat pipeline helpers for the PCA tutorial."""

import heat as ht


class Pipeline:
    """Chain Heat transformers and an optional final classifier.

    Parameters
    ----------
    steps : list of (str, ht.BaseEstimator)
        Named estimators. Every intermediate step must be a transformer.
        Estimators are fitted in place; no cloning is performed.

    Attributes
    ----------
    steps : list of (str, ht.BaseEstimator)
        Estimators in execution order.
    named_steps : dict of str to ht.BaseEstimator
        Access the same estimators by name.

    Notes
    -----
    Estimator distribution constraints still apply. With the current Heat
    KNN, keep training labels replicated (split=None), even if X is split.
    """

    def __init__(self, steps: list[tuple[str, ht.BaseEstimator]]) -> None:
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("A pipeline needs at least one step.")
        names = []
        for index, step in enumerate(self.steps):
            if not isinstance(step, (tuple, list)) or len(step) != 2:
                raise TypeError("Each step must be a (name, estimator) pair.")
            name, estimator = step
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("Step names must be nonempty strings without '__'.")
            if name in names:
                raise ValueError(f"Duplicate step name: {name!r}.")
            names.append(name)
            if index < len(self.steps) - 1:
                if not ht.is_transformer(estimator):
                    raise TypeError(f"Intermediate step {name!r} must be a Heat transformer.")
            elif not (ht.is_transformer(estimator) or ht.is_classifier(estimator)):
                raise TypeError("The final step must be a Heat transformer or classifier.")
        self.named_steps = dict(self.steps)
        self._fitted = False

    def _repr_html_(self) -> str:
        """Return a collapsible HTML diagram of the pipeline's configuration."""
        from html import escape
        from inspect import Parameter, signature

        style = """<style>
        .heat-pipeline {border:1px dashed currentColor;border-radius:10px;
            padding:12px 16px;margin:0;min-width:0;width:fit-content;max-width:100%;color:inherit}
        .heat-pipeline legend {font-weight:600;padding:0 6px}
        .heat-pipeline .heat-steps {display:flex;flex-wrap:wrap;align-items:flex-start;gap:12px}
        .heat-pipeline details {border:1px solid #6890bd;border-radius:8px;
            min-width:150px;max-width:100%}
        .heat-pipeline summary {background:rgba(72,131,207,.16);border-radius:7px;
            padding:10px 14px;cursor:pointer;text-align:center;list-style-position:inside}
        .heat-pipeline summary small {opacity:.75}
        .heat-pipeline details[open] > summary {border-bottom:1px solid #6890bd;
            border-radius:7px 7px 0 0}
        .heat-pipeline table {border-collapse:collapse;font-size:.9em;width:100%;margin:0}
        .heat-pipeline th,.heat-pipeline td {padding:5px 10px;text-align:left;
            overflow-wrap:anywhere}
        .heat-pipeline th {font-weight:normal}
        .heat-pipeline .heat-arrow {align-self:center;font-size:1.4em;opacity:.7}
        .heat-pipeline .heat-empty {display:block;padding:10px}
        </style>"""
        boxes = []
        for name, estimator in self.steps:
            rows = []
            for parameter in signature(type(estimator)).parameters.values():
                if parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
                    continue
                if not hasattr(estimator, parameter.name):
                    continue
                value = getattr(estimator, parameter.name)
                display = (
                    getattr(value, "__name__", type(value).__name__)
                    if callable(value)
                    else repr(value)
                )
                rows.append(
                    f'<tr><th scope="row"><code>{escape(parameter.name)}</code></th>'
                    f"<td><code>{escape(display)}</code></td></tr>"
                )
            parameters = (
                "<table>" + "".join(rows) + "</table>"
                if rows
                else '<small class="heat-empty">No parameters</small>'
            )
            boxes.append(
                f"<details><summary><strong>{escape(type(estimator).__name__)}</strong><br>"
                f"<small>{escape(name)}</small></summary>{parameters}</details>"
            )
        arrow = '<span class="heat-arrow" aria-hidden="true">&rarr;</span>'
        return (
            style + '<fieldset class="heat-pipeline"><legend>Pipeline</legend>'
            '<div class="heat-steps">' + arrow.join(boxes) + "</div></fieldset>"
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
            raise ValueError("Classification requires labels y.")
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
        """Fit transformers without labels, then fit the final estimator.

        Parameters
        ----------
        X : ht.DNDarray
            Training features of shape (n_samples, n_features).
        y : ht.DNDarray or None, optional
            One-dimensional class labels; required for a final classifier.

        Returns
        -------
        Pipeline
            This pipeline with its steps fitted in place.
        """
        self._fitted = False
        self._check_X(X)
        final = self.steps[-1][1]
        if ht.is_classifier(final):
            self._check_y(X, y)
        for _, transformer in self.steps[:-1]:
            X = transformer.fit_transform(X)
        if ht.is_classifier(final):
            final.fit(X, y)
        else:
            final.fit(X)
        self._fitted = True
        return self

    def predict(self, X: ht.DNDarray) -> ht.DNDarray:
        """Predict labels after applying the fitted intermediate transformers.

        Parameters
        ----------
        X : ht.DNDarray
            Features of shape (n_samples, n_features).
        """
        self._require("predict")
        return self.steps[-1][1].predict(self._transform_intermediate(X))

    def score(self, X: ht.DNDarray, y: ht.DNDarray) -> float:
        """Return classification accuracy using a global Heat reduction.

        Parameters
        ----------
        X : ht.DNDarray
            Features of shape (n_samples, n_features).
        y : ht.DNDarray
            True class labels of shape (n_samples,).
        """
        self._check_X(X)
        self._check_y(X, y)
        if X.shape[0] == 0:
            raise ValueError("Accuracy requires at least one sample.")
        predictions = self.predict(X)
        return float(ht.mean((predictions == y).astype(ht.float64)).item())

    def transform(self, X: ht.DNDarray) -> ht.DNDarray:
        """Apply all fitted steps, including the final transformer.

        Parameters
        ----------
        X : ht.DNDarray
            Features of shape (n_samples, n_features).
        """
        self._require("transform")
        return self.steps[-1][1].transform(self._transform_intermediate(X))

    def fit_transform(self, X: ht.DNDarray, y: ht.DNDarray | None = None) -> ht.DNDarray:
        """Fit a transformer-only pipeline and return transformed features.

        Parameters
        ----------
        X : ht.DNDarray
            Training features of shape (n_samples, n_features).
        y : ht.DNDarray or None, optional
            Ignored by transformer-only pipelines.
        """
        if not ht.is_transformer(self.steps[-1][1]):
            raise TypeError("fit_transform() requires a final Heat transformer.")
        self._fitted = False
        self._check_X(X)
        for _, transformer in self.steps[:-1]:
            X = transformer.fit_transform(X)
        result = self.steps[-1][1].fit_transform(X)
        self._fitted = True
        return result


def make_pipeline(*estimators: ht.BaseEstimator) -> Pipeline:
    """Create a Heat pipeline with automatic lowercase step names.

    Parameters
    ----------
    *estimators : ht.BaseEstimator
        Transformers followed by a final transformer or classifier.
        Repeated class names receive suffixes '-1', '-2', and so on.

    Returns
    -------
    Pipeline
        A pipeline containing the supplied estimator instances.
    """
    names = [type(estimator).__name__.lower() for estimator in estimators]
    occurrences: dict[str, int] = {}
    steps = []
    for name, estimator in zip(names, estimators):
        occurrences[name] = occurrences.get(name, 0) + 1
        step_name = f"{name}-{occurrences[name]}" if names.count(name) > 1 else name
        steps.append((step_name, estimator))
    return Pipeline(steps)
