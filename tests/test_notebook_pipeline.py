"""Exercise the PCA notebook's pipeline and single-process tutorial."""

import ast
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch

import heat as ht
import pytest


def _imports_pipeline(cell):
    return cell["cell_type"] == "code" and any(
        isinstance(node, ast.ImportFrom)
        and node.module == "pipeline"
        and any(alias.name == "Pipeline" for alias in node.names)
        for node in ast.parse("".join(cell["source"])).body
    )


@pytest.fixture(scope="module")
def helpers():
    """Import the helper module beside the notebook without running the tutorial."""
    notebook_path = Path(__file__).resolve().parents[1] / "doc/source/tutorials/notebooks/PCA.ipynb"
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            assert not (isinstance(node, ast.ClassDef) and node.name == "Pipeline")
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
                assert node.module == "sklearn.datasets"
                assert [alias.name for alias in node.names] == ["load_iris"]
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("sklearn") for alias in node.names)
    assert any(_imports_pipeline(cell) for cell in notebook["cells"])
    with patch.object(sys, "path", [str(notebook_path.parent), *sys.path]):
        from pipeline import Pipeline, make_pipeline

    return Pipeline, make_pipeline


@pytest.fixture(params=[None, 0], ids=["replicated", "split-0"])
def data(request):
    """Create deterministic features and labels with uneven MPI chunks."""
    # Uneven chunks on 2, 3, and 4 ranks exercise global accuracy reduction.
    X = ht.array(
        [[i / 5, (i % 4) / 3, (i * i % 7) / 7, i / 10 + (i % 3)] for i in range(17)],
        dtype=ht.float64,
        split=request.param,
    )
    # Heat KNN currently needs replicated labels: distributed label lookup
    # reaches a missing DNDarray.numel() call in core/_indexing_utils.py.
    y = ht.array([i % 3 for i in range(17)], dtype=ht.int64)
    query = X + 0.03
    return X, y, query


@pytest.mark.parametrize("n_components", [None, 2, 3])
def test_classifier_matches_manual_chain(helpers, data, n_components):
    """Match manual Heat fitting, prediction, and global accuracy."""
    Pipeline, make_pipeline = helpers
    X, y, query = data
    steps = [ht.preprocessing.StandardScaler()]
    if n_components is not None:
        steps.append(ht.decomposition.PCA(n_components=n_components, svd_solver="full"))
    classifier = ht.classification.KNeighborsClassifier(n_neighbors=3)
    steps.append(classifier)
    model = make_pipeline(*steps)
    assert isinstance(model, Pipeline)
    assert model.named_steps["standardscaler"] is steps[0]
    assert model.named_steps["kneighborsclassifier"] is classifier
    assert model.fit(X, y) is model

    scaler = ht.preprocessing.StandardScaler()
    manual_X = scaler.fit_transform(X)
    manual_query = scaler.transform(query)
    if n_components is not None:
        pca = ht.decomposition.PCA(n_components=n_components, svd_solver="full")
        manual_X = pca.fit_transform(manual_X)
        manual_query = pca.transform(manual_query)
        assert ht.allclose(
            model.named_steps["pca"].explained_variance_ratio_, pca.explained_variance_ratio_
        )
    manual_classifier = ht.classification.KNeighborsClassifier(n_neighbors=3)
    assert manual_classifier.fit(manual_X, y) is None
    expected = manual_classifier.predict(manual_query)
    predictions = model.predict(query)
    assert isinstance(predictions, ht.DNDarray)
    assert predictions.shape == y.shape
    assert predictions.split == expected.split
    assert ht.equal(predictions, expected)
    accuracy = model.score(query, y)
    assert isinstance(accuracy, float)
    assert accuracy == pytest.approx((expected == y).sum().item() / y.shape[0])

    # Prediction/scoring must use fitted state; numpy() must never be needed.
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(ht.DNDarray, "numpy", side_effect=AssertionError("No gathering"))
        )
        for step in steps:
            stack.enter_context(
                patch.object(step, "fit", side_effect=AssertionError("No refitting"))
            )
        assert ht.equal(model.predict(query), expected)
        assert model.score(query, y) == accuracy


def test_transformer_pipeline(helpers, data):
    """Keep labels out of unsupervised fitting and support final transformation."""
    Pipeline, make_pipeline = helpers
    X, y, query = data
    scaler = ht.preprocessing.StandardScaler()
    pca = ht.decomposition.PCA(n_components=2, svd_solver="full")
    model = Pipeline([("scale", scaler), ("reduce", pca)])
    # Labels must not be forwarded to the scaler or PCA.
    with patch.object(scaler, "transform", wraps=scaler.transform) as scale_transform:
        with patch.object(pca, "transform", wraps=pca.transform) as pca_transform:
            transformed = model.fit_transform(X, y)
            assert scale_transform.call_count == 1
            assert pca_transform.call_count == 1
    assert isinstance(transformed, ht.DNDarray)
    assert transformed.shape == (X.shape[0], 2)
    expected = pca.transform(scaler.transform(X))
    assert transformed.split == expected.split
    assert ht.allclose(transformed, expected)
    assert ht.allclose(model.transform(query), pca.transform(scaler.transform(query)))
    with pytest.raises(TypeError, match="predict"):
        model.predict(query)
    with pytest.raises(TypeError, match="predict"):
        model.score(query, y)
    assert make_pipeline(ht.preprocessing.StandardScaler()).fit(X) is not None


def test_score_with_distributed_labels(helpers, data):
    """Reduce correct counts globally, including uneven rank-local accuracies."""

    class ZeroClassifier(ht.BaseEstimator, ht.ClassificationMixin):
        def fit(self, X: ht.DNDarray, y: ht.DNDarray) -> None:
            pass

        def predict(self, X: ht.DNDarray) -> ht.DNDarray:
            return ht.zeros(X.shape[0], dtype=ht.int64, split=X.split, device=X.device, comm=X.comm)

    _, make_pipeline = helpers
    X, _, _ = data
    y = ht.array([0] * 9 + [1] * 8, dtype=ht.int64, split=X.split)
    model = make_pipeline(ZeroClassifier()).fit(X, y)
    assert model.score(X, y) == pytest.approx(9 / 17)


def test_html_display(helpers, data):
    """Render steps in order without fitting or changing estimator state."""
    _, make_pipeline = helpers
    X, y, _ = data
    scaler = ht.preprocessing.StandardScaler()
    pca = ht.decomposition.PCA(n_components=2, svd_solver="full")
    model = make_pipeline(scaler, pca, ht.classification.KNeighborsClassifier())
    html = model._repr_html_()
    assert html.index("StandardScaler") < html.index("PCA") < html.index("KNeighborsClassifier")
    assert html.count("&rarr;") == 2
    assert html.count("<details>") == 3
    assert html.count("<summary>") == 3
    assert "<legend>Pipeline</legend>" in html
    assert "<details open" not in html
    assert "<script" not in html
    assert "standardscaler" in html
    assert "flex-wrap:wrap" in html
    assert "currentColor" in html
    assert not model._fitted
    assert not hasattr(scaler, "mean_")
    assert pca.mean_ is None
    model.fit(X, y)
    mean = scaler.mean_.copy()
    components = pca.components_.copy()
    assert model._repr_html_() == html
    assert model._fitted
    assert ht.equal(scaler.mean_, mean)
    assert ht.equal(pca.components_, components)
    assert "<code>mean_</code>" not in html
    assert "<code>components_</code>" not in html
    assert "<code>x</code>" not in html
    assert "<code>y</code>" not in html


def test_html_constructor_parameters(helpers):
    """Include keyword-only settings, current values, and readable callable names."""
    _, make_pipeline = helpers
    scaler = ht.preprocessing.StandardScaler(with_std=False)
    classifier = ht.classification.KNeighborsClassifier(n_neighbors=3)
    model = make_pipeline(scaler, classifier)
    html = model._repr_html_()
    assert "<code>copy</code></th><td><code>True</code>" in html
    assert "<code>with_mean</code></th><td><code>True</code>" in html
    assert "<code>with_std</code></th><td><code>False</code>" in html
    assert "<code>n_neighbors</code></th><td><code>3</code>" in html
    assert "<code>effective_metric_</code></th><td><code>cdist</code>" in html
    assert "0x" not in html
    scaler.with_std = True
    assert "<code>with_std</code></th><td><code>True</code>" in model._repr_html_()


def test_html_accessible_parameters_and_escaping(helpers):
    """Skip unavailable constructor attributes and escape parameter values."""

    class ExampleTransformer(ht.TransformMixin, ht.BaseEstimator):
        def __init__(self, label='<b title="unsafe">&</b>', missing=None):
            self.label = label
            self.learned_attribute_ = "not a constructor setting"

    _, make_pipeline = helpers
    html = make_pipeline(ExampleTransformer())._repr_html_()
    assert "&lt;b title=&quot;unsafe&quot;&gt;&amp;&lt;/b&gt;" in html
    assert '<b title="unsafe">' not in html
    assert "<code>missing</code>" not in html
    assert "learned_attribute_" not in html
    assert "not a constructor setting" not in html
    assert "No parameters" in make_pipeline(ht.TransformMixin())._repr_html_()


def test_html_escapes_labels(helpers):
    """Treat both step names and estimator class names as text."""
    Pipeline, _ = helpers
    estimator = type("Scaler<unsafe>", (ht.preprocessing.StandardScaler,), {})()
    model = Pipeline([('<scale & "input">', estimator)])
    html = model._repr_html_()
    assert "&lt;scale &amp; &quot;input&quot;&gt;" in html
    assert "Scaler&lt;unsafe&gt;" in html
    assert '<scale & "input">' not in html
    assert "Scaler<unsafe>" not in html
    assert "&rarr;" not in html


def test_names_and_single_classifier(helpers, data):
    """Name duplicates consistently and allow a classifier without transformers."""
    Pipeline, make_pipeline = helpers
    first, second = ht.preprocessing.StandardScaler(), ht.preprocessing.StandardScaler()
    model = make_pipeline(first, second)
    assert list(model.named_steps) == ["standardscaler-1", "standardscaler-2"]
    assert model.named_steps["standardscaler-1"] is first
    assert model.named_steps["standardscaler-2"] is second
    X, y, query = data
    classifier = ht.classification.KNeighborsClassifier(n_neighbors=1)
    single = Pipeline([("knn", classifier)])
    assert single.fit(X, y) is single
    assert ht.equal(single.predict(query), classifier.predict(query))


@pytest.mark.parametrize("name", ["", "bad__name", None, 7])
def test_invalid_names(helpers, name):
    """Reject empty, ambiguous, and non-string names."""
    Pipeline, _ = helpers
    with pytest.raises(ValueError, match="Step names"):
        Pipeline([(name, ht.preprocessing.StandardScaler())])


def test_invalid_steps(helpers):
    """Reject malformed pipelines before fitting starts."""
    Pipeline, make_pipeline = helpers
    with pytest.raises(ValueError, match="at least one"):
        make_pipeline()
    with pytest.raises(ValueError, match="Duplicate"):
        Pipeline([("scale", ht.preprocessing.StandardScaler())] * 2)
    with pytest.raises(TypeError, match="pair"):
        Pipeline([ht.preprocessing.StandardScaler()])
    with pytest.raises(TypeError, match="Intermediate"):
        make_pipeline(ht.classification.KNeighborsClassifier(), ht.preprocessing.StandardScaler())
    with pytest.raises(TypeError, match="final step"):
        make_pipeline(object())


def test_invalid_operations_and_inputs(helpers, data):
    """Report unsupported operations, unfitted state, and invalid array shapes."""
    _, make_pipeline = helpers
    X, y, query = data
    scaler = ht.preprocessing.StandardScaler()
    model = make_pipeline(scaler, ht.classification.KNeighborsClassifier(n_neighbors=1))
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(query)

    with pytest.raises(RuntimeError, match="fit"):
        model.score(query, y)
    with pytest.raises(RuntimeError, match="fit"):
        make_pipeline(scaler).transform(query)
    with pytest.raises(ValueError, match="labels"):
        model.fit(X)
    with pytest.raises(TypeError, match="y must"):
        model.fit(X, [0] * X.shape[0])
    with pytest.raises(ValueError, match="matching X"):
        model.fit(X, ht.zeros((X.shape[0], 1), dtype=ht.int64, split=X.split))
    with pytest.raises(TypeError, match="X must"):
        model.fit([[1.0]], y)
    with pytest.raises(ValueError, match="n_features"):
        model.fit(y, y)
    with pytest.raises(TypeError, match="final Heat transformer"):
        model.fit_transform(X, y)
    model.fit(X, y)
    with pytest.raises(TypeError, match="transform"):
        model.transform(query)
    with pytest.raises(ValueError, match="matching X"):
        model.score(query, ht.zeros(X.shape[0] - 1, dtype=ht.int64, split=X.split))
    with pytest.raises(ValueError, match="at least one sample"):
        model.score(ht.empty((0, 4)), ht.empty((0,), dtype=ht.int64))
    # A failed refit must not leave the pipeline marked as successfully fitted.
    with pytest.raises(ValueError, match="labels"):
        model.fit(X)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(query)


@pytest.fixture(scope="module")
def tutorial_run():
    """Run every tutorial cell and record pipeline fits and evaluations."""
    if ht.MPI_WORLD.size != 1:
        pytest.skip("The complete notebook is intended for one process.")
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn.datasets")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(__file__).resolve().parents[1] / "doc/source/tutorials/notebooks/PCA.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    fits, evaluations = [], []
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "path", [str(path.parent), *sys.path]))
        stack.enter_context(patch.object(plt, "show", side_effect=lambda: plt.close("all")))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), f"{path}:cell-{index}", "exec"), namespace)
            if _imports_pipeline(cell):
                pipeline = namespace["Pipeline"]
                original_fit, original_score = pipeline.fit, pipeline.score

                def record_fit(self, X, y=None):
                    fits.append((self, X, y))
                    return original_fit(self, X, y)

                def record_score(self, X, y):
                    evaluations.append((self, X, y))
                    return original_score(self, X, y)

                stack.enter_context(patch.object(pipeline, "fit", record_fit))
                stack.enter_context(patch.object(pipeline, "score", record_score))
    return namespace, fits, evaluations


def test_tutorial_split_and_model_arrays(tutorial_run):
    """Keep aligned Iris rows and unsplit arrays throughout model fitting."""
    namespace, fits, _ = tutorial_run
    X, y = namespace["X"], namespace["y"]
    X_train, X_test = namespace["X_train"], namespace["X_test"]
    y_train, y_test = namespace["y_train"], namespace["y_test"]
    assert X.dtype == ht.float64
    assert y.dtype == ht.int64
    assert X_train.shape == (105, 4)
    assert X_test.shape == (45, 4)
    assert y_train.shape == (105,)
    assert y_test.shape == (45,)
    assert all(
        isinstance(a, ht.DNDarray) and a.split is None
        for a in (X, y, X_train, X_test, y_train, y_test)
    )

    def rows(features, labels):
        return Counter((tuple(row), label) for row, label in zip(features.numpy(), labels.numpy()))

    assert rows(X_train, y_train) + rows(X_test, y_test) == rows(X, y)
    repeated = ht.utils.data.create_train_val_split(X, y, p=0.70, seed=42)
    assert all(ht.equal(a, b) for a, b in zip(repeated, (X_train, X_test, y_train, y_test)))
    for _, features, labels in fits:
        assert isinstance(features, ht.DNDarray) and features.split is None
        assert isinstance(labels, ht.DNDarray) and labels.split is None
    assert namespace["points_2d"].shape == (105, 2)
    assert namespace["explained_variance"].sum() == pytest.approx(1.0)


def test_tutorial_component_weights(tutorial_run):
    """Label the PCA recipes correctly and reconstruct the plotted coordinates."""
    namespace, _, _ = tutorial_run
    weights = namespace["component_weights"]
    pca = namespace["pca_step"]
    assert list(weights.index) == ["PCA feature 1", "PCA feature 2"]
    assert list(weights.columns) == namespace["iris"].feature_names
    coefficients = ht.array(weights.to_numpy(copy=True))
    assert ht.equal(coefficients, pca.components_)
    centered = namespace["scaled_train"] - pca.mean_
    reconstructed = centered @ coefficients.T
    assert ht.allclose(reconstructed, ht.array(namespace["points_2d"]))


def test_tutorial_cross_validation(tutorial_run):
    """Use each training sample once for validation and fit fresh steps each time."""
    namespace, fits, evaluations = tutorial_run
    folds = [fold.numpy().tolist() for fold in namespace["folds"]]
    assert sorted(index for fold in folds for index in fold) == list(range(105))
    y_train = namespace["y_train"].numpy()
    for species in range(3):
        counts = [sum(y_train[index] == species for index in fold) for fold in folds]
        assert max(counts) - min(counts) <= 1

    # Three main fits/evaluations precede the 3 x 5 cross-validation runs.
    assert len(fits) == 18
    assert len(evaluations) == 21
    cv_fits, cv_evaluations = fits[3:], evaluations[6:]
    assert len({id(model) for model, _, _ in fits}) == 18
    estimators = [step for model, _, _ in fits for _, step in model.steps]
    assert len({id(step) for step in estimators}) == len(estimators)
    for run, ((model, fit_X, fit_y), (evaluated_model, val_X, val_y)) in enumerate(
        zip(cv_fits, cv_evaluations)
    ):
        fold = run % 5
        assert evaluated_model is model
        train_indices = ht.concatenate(
            [indices for i, indices in enumerate(namespace["folds"]) if i != fold]
        )
        val_indices = namespace["folds"][fold]
        assert not (set(train_indices.numpy().tolist()) & set(folds[fold]))
        assert ht.equal(fit_X, namespace["X_train"][train_indices])
        assert ht.equal(fit_y, namespace["y_train"][train_indices])
        assert ht.equal(val_X, namespace["X_train"][val_indices])
        assert ht.equal(val_y, namespace["y_train"][val_indices])
