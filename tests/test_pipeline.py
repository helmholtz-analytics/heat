import heat as ht

from heat.testing.basic_test import TestCase


class TestPipeline(TestCase):
    def test_construction_and_validation(self):
        with self.assertRaises(ValueError):
            ht.Pipeline([])

        with self.assertRaises(TypeError):
            ht.Pipeline([("transformer", 7)])

        with self.assertRaises(TypeError):
            ht.Pipeline([("scaler", ht.preprocessing.StandardScaler()), ("bad", 3)])

        with self.assertRaises(ValueError):
            ht.Pipeline([("", ht.preprocessing.StandardScaler())])

        with self.assertRaises(ValueError):
            ht.Pipeline([("a", ht.preprocessing.StandardScaler()), ("a", ht.preprocessing.StandardScaler())])

        with self.assertRaises(TypeError):
            ht.Pipeline([("scaler", ht.preprocessing.StandardScaler()), ("bad", ht.DNDarray)])

        # Regression estimators are valid final pipeline steps; they should be accepted.
        reg_pipe = ht.Pipeline([("scaler", ht.preprocessing.StandardScaler()), ("reg", ht.regression.Lasso())])
        self.assertIsInstance(reg_pipe, ht.Pipeline)

    def test_make_pipeline(self):
        pipe = ht.make_pipeline(ht.preprocessing.StandardScaler(), ht.preprocessing.StandardScaler())
        self.assertIsInstance(pipe, ht.Pipeline)
        self.assertEqual(list(pipe.named_steps), ["standardscaler-1", "standardscaler-2"])

        clf_pipe = ht.make_pipeline(
            ht.preprocessing.StandardScaler(), ht.classification.KNeighborsClassifier(n_neighbors=1)
        )
        self.assertEqual(list(clf_pipe.named_steps), ["standardscaler", "kneighborsclassifier"])

    def test_transformer_only_pipeline(self):
        X = ht.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], split=0)
        pipe = ht.Pipeline([("scaler", ht.preprocessing.StandardScaler()), ("scaler_2", ht.preprocessing.StandardScaler())])

        transformed = pipe.fit_transform(X)
        pipe_pred = pipe.transform(X)

        self.assertIsInstance(transformed, ht.DNDarray)
        self.assertIsInstance(pipe_pred, ht.DNDarray)
        self.assertEqual(transformed.shape, X.shape)
        self.assertEqual(pipe_pred.shape, X.shape)

    def test_classifier_pipeline(self):
        X = ht.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], split=0)
        y = ht.array([0, 1, 0, 1], split=0)

        scaler = ht.preprocessing.StandardScaler()
        clf = ht.classification.KNeighborsClassifier(n_neighbors=1)
        pipe = ht.Pipeline([("scaler", scaler), ("clf", clf)])
        fit_pipe = pipe.fit(X, y)
        self.assertIs(fit_pipe, pipe)

        preds = pipe.predict(X)
        manual = scaler.fit_transform(X)
        clf_manual = ht.classification.KNeighborsClassifier(n_neighbors=1)
        clf_manual.fit(manual, y)
        expected = clf_manual.predict(manual)

        self.assertEqual(preds.shape, y.shape)
        self.assertTrue(ht.all(preds == expected))
        self.assertGreaterEqual(pipe.score(X, y), 0.9)

    def test_regressor_pipeline(self):
        X = ht.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], split=0)
        y = ht.array([1.0, 2.0, 3.0, 4.0], split=0)

        scaler = ht.preprocessing.StandardScaler()
        reg = ht.regression.Lasso(max_iter=200, tol=None)
        pipe = ht.Pipeline([("scaler", scaler), ("reg", reg)])
        pipe.fit(X, y)

        preds = pipe.predict(X)
        manual = scaler.fit_transform(X)
        reg_manual = ht.regression.Lasso(max_iter=200, tol=None)
        reg_manual.fit(manual, y)
        expected = reg_manual.predict(manual)

        self.assertEqual(preds.shape, expected.shape)
        self.assertTrue(ht.allclose(preds, expected))
        self.assertAlmostEqual(float(pipe.score(X, y)), float(reg_manual.score(manual, y)))

    def test_invalid_input_and_methods(self):
        X = ht.array([[1.0, 2.0], [3.0, 4.0]], split=0)
        y = ht.array([0, 1], split=0)
        pipe = ht.Pipeline([("scaler", ht.preprocessing.StandardScaler()), ("clf", ht.classification.KNeighborsClassifier(n_neighbors=1))])

        with self.assertRaises(ValueError):
            pipe.fit(ht.array([[1.0, 2.0, 3.0]]), y)

        with self.assertRaises(TypeError):
            pipe.transform("not an array")

        with self.assertRaises(TypeError):
            ht.Pipeline([("scaler", ht.preprocessing.StandardScaler())]).fit_transform("bad")

        with self.assertRaises(TypeError):
            ht.Pipeline([("scaler", ht.preprocessing.StandardScaler())]).predict(X)

        with self.assertRaises(ValueError):
            pipe.score(X, ht.array([0.0, 1.0, 2.0], split=0))
