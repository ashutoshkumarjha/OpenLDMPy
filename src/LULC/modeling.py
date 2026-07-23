"""Per-class suitability model fitting and prediction.

Ports R's ``fitModelSeparately`` (Rasterise_dev_68akj.r lines 999-1118) and
``constructSM``/``ParallelconstructSuitablity`` (lines 422-486), plus
``getModelSummary`` (lines 2007-2025).

Model-type equivalences (chosen for algorithmic fidelity to the R packages):

* ``logistic``     R ``glm(family=binomial)``            -> ``LogisticRegression(penalty=None)``
  (unregularized, matching R's un-penalized IRLS fit).
* ``nnet``         R ``nnet::multinom`` on a 0/1 response — mathematically a
  plain (multinomial) logistic regression with no hidden layer — is likewise
  mapped to ``LogisticRegression(penalty=None)``, NOT an MLP: R's ``multinom``
  has no hidden units, so a hidden-layer network would be *less* faithful.
* ``regression``   R ``lm``                              -> ``LinearRegression``.
* ``randomForest`` R ``randomForest`` (factor response)  -> ``RandomForestClassifier``
  with R's defaults: ntree=500, mtry=floor(sqrt(p)), sampling with
  replacement, nodesize=1.
* ``svm``          R ``e1071::svm(probability=TRUE)``    -> ``SVC(probability=True,
  kernel='rbf', gamma='auto')`` — e1071's default gamma is 1/n_features,
  which is sklearn's ``'auto'`` (not ``'scale'``).

Training-data semantics preserved from R: rows are cells with a valid T1
class; all remaining NAs (e.g. driver gaps) become 0
(``dta[is.na(dta)] <- 0``); the response for class *i* is its one-hot 0/1
column. Only ``method='NotIncludeCurrentClass'`` is supported — R aborts on
anything else (line 1109).

Prediction semantics: suitability is predicted from the *T2* drivers for all
model types. (The R source's lm/glm/nnet branches pass ``data=`` instead of
``newdata=`` to ``predict()``, which silently returns training-set fitted
values instead of T2 predictions; the randomForest/svm branches use
``newdata=`` correctly. The intended behavior — predict on T2 — is
implemented uniformly here.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from .config import PARALLEL_BACKEND, PARALLEL_JOBS, logger

RANDOM_STATE = 0  # fixed for determinism


@dataclass
class FittedClassModel:
    class_name: str
    model_type: str
    driver_names: List[str]
    estimator: object
    n_positive: int
    # Wald-test p-values, [intercept] + driver_names order. Only populated
    # for logistic/nnet (statsmodels.Logit fit alongside the sklearn
    # estimator purely for display — see get_model_summary); None otherwise,
    # or if the statsmodels fit failed to converge (e.g. perfect separation).
    pvalues: Optional[np.ndarray] = None


def _build_estimator(model_type: str):
    if model_type in ("logistic", "nnet"):
        # solver='newton-cg' (not the default 'lbfgs'): with unscaled driver
        # features (distances in the thousands next to a 0-1 elevation
        # index), lbfgs's gradient-only updates converge to a materially
        # different point than R's glm(family=binomial) IRLS fit, even
        # though both are unpenalized MLE solutions in principle. newton-cg
        # (and newton-cholesky) use curvature information like IRLS and
        # reproduce R's coefficients to ~1e-2, verified against the
        # tests/r_oracle glm_class1_coefs.csv fixture.
        return LogisticRegression(penalty=None, solver="newton-cg", max_iter=2000)
    if model_type == "regression":
        return LinearRegression()
    if model_type == "randomForest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if model_type == "svm":
        return SVC(probability=True, kernel="rbf", gamma="auto", random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model type: {model_type!r}")


def _significance_star(p: float) -> str:
    """R glm summary's significance codes, from a Wald-test p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _fit_logistic_pvalues(x: np.ndarray, y: np.ndarray, cname: str) -> Optional[np.ndarray]:
    """Wald-test p-values for a logistic fit, [intercept] + drivers order,
    via a parallel statsmodels.Logit fit (display-only — sklearn's
    LogisticRegression above remains the model actually used for
    prediction). Returns None if the fit doesn't converge (e.g. perfect
    separation on a small/degenerate class) rather than failing the whole
    model-fitting pass over one diagnostic value.

    Imports statsmodels lazily, here rather than at module level: this is
    the *only* place in the whole package that needs it (everything else
    is sklearn), and statsmodels being unimportable — confirmed for real
    inside a QGIS plugin environment, where a compiled submodule can be
    blocked by macOS code-signing enforcement even after a successful
    pip install — shouldn't take the entire LULC package down with it via
    LULC/__init__.py's eager `from . import ... modeling ...`. A missing/
    broken statsmodels now only costs this one cosmetic diagnostic
    (significance stars), the same graceful degradation already applied
    below for a statsmodels fit that runs but doesn't converge."""
    try:
        import statsmodels.api as sm
        sm_model = sm.Logit(y, sm.add_constant(x, has_constant="add")).fit(disp=0)
        return np.asarray(sm_model.pvalues)
    except Exception as exc:
        logger.warning(f"  {cname}: statsmodels p-value fit failed ({exc}); significance stars omitted.")
        return None


def parse_formula_drivers(formula: str, all_driver_names: Sequence[str]) -> List[str]:
    """Extract driver names from an R-style formula string.

    Accepts the R backend's naming scheme (``"T1.BuildUp ~ TD1.Elevation +
    TD1.DistanceToRoad"``), stripping the ``TD1.``/``TD2.`` prefixes so the
    result matches the driver-dictionary keys.
    """
    rhs = formula.split("~", 1)[1]
    tokens = [t.strip() for t in rhs.split("+")]
    drivers = []
    for token in tokens:
        name = token
        for prefix in ("TD1.", "TD2."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name not in all_driver_names:
            raise ValueError(
                f"Formula driver {token!r} (-> {name!r}) not among available drivers {list(all_driver_names)}"
            )
        drivers.append(name)
    return drivers


def fit_models_separately(
    t1_codes: np.ndarray,
    driver_stack_t1: np.ndarray,
    driver_names: Sequence[str],
    class_ids: Sequence[int],
    class_names: Sequence[str],
    model_types: Sequence[str],
    model_formulas: Optional[Sequence[Optional[str]]] = None,
    method: str = "NotIncludeCurrentClass",
) -> List[FittedClassModel]:
    """Fit one binary suitability model per class (R: fitModelSeparately).

    ``t1_codes``: (rows, cols) class-code array; ``driver_stack_t1``:
    (n_drivers, rows, cols).
    """
    if method != "NotIncludeCurrentClass":
        # R aborts identically (line 1109: "No Model Exists yet").
        raise NotImplementedError(f"method={method!r} is not implemented (nor was it in R)")

    train_mask = np.isin(t1_codes, list(class_ids))
    n_train = int(train_mask.sum())
    logger.info(f"Fitting {len(class_ids)} class models on {n_train} training cells...")

    x_full = np.stack([band[train_mask] for band in driver_stack_t1], axis=1)
    x_full = np.nan_to_num(x_full, nan=0.0)  # R: dta[is.na(dta)] <- 0

    models: List[FittedClassModel] = []
    for idx, (cid, cname) in enumerate(zip(class_ids, class_names)):
        model_type = model_types[idx]
        formula = model_formulas[idx] if model_formulas is not None else None
        if formula:
            selected = parse_formula_drivers(formula, driver_names)
        else:
            selected = list(driver_names)
        cols = [list(driver_names).index(d) for d in selected]

        y = (t1_codes[train_mask] == cid).astype(int)
        estimator = _build_estimator(model_type)
        logger.info(f"  {cname} ({model_type}) ~ {'+'.join(selected)}")
        estimator.fit(x_full[:, cols], y)
        pvalues = _fit_logistic_pvalues(x_full[:, cols], y, cname) if model_type in ("logistic", "nnet") else None
        models.append(
            FittedClassModel(
                class_name=cname,
                model_type=model_type,
                driver_names=selected,
                estimator=estimator,
                n_positive=int(y.sum()),
                pvalues=pvalues,
            )
        )
    return models


def _predict_single(model: FittedClassModel, x: np.ndarray) -> np.ndarray:
    est = model.estimator
    if isinstance(est, LinearRegression):
        return est.predict(x)
    if isinstance(est, LogisticRegression):
        return est.predict_proba(x)[:, 1]
    # RandomForestClassifier / SVC: probability of the positive class,
    # matching R's predict(type='prob')[,2] / attr("probabilities")[,2].
    classes = list(est.classes_)
    positive_col = classes.index(1) if 1 in classes else len(classes) - 1
    return est.predict_proba(x)[:, positive_col]


def construct_suitability(
    models: List[FittedClassModel],
    driver_stack: np.ndarray,
    driver_names: Sequence[str],
    n_jobs: int = PARALLEL_JOBS,
) -> Dict[str, np.ndarray]:
    """Predict per-class suitability on the full grid.

    Port of R's ``ParallelconstructSuitablity``: prediction is restricted to
    cells whose driver values are not all NA/0 (R filters
    ``rowSums(driver, na.rm=TRUE) != 0``); everything else is NaN. Returns
    ``{class_name: (rows, cols) float array}``.
    """
    grid_shape = driver_stack.shape[1:]
    driver_sums = np.nansum(driver_stack, axis=0)
    pred_mask = driver_sums != 0

    x = np.stack([band[pred_mask] for band in driver_stack], axis=1)
    name_index = {n: i for i, n in enumerate(driver_names)}

    def predict_one(model: FittedClassModel) -> np.ndarray:
        cols = [name_index[d] for d in model.driver_names]
        out = np.full(grid_shape, np.nan)
        out[pred_mask] = _predict_single(model, x[:, cols])
        return out

    logger.info(f"Predicting suitability for {len(models)} classes on {int(pred_mask.sum())} cells...")
    n_jobs_eff = min(len(models), n_jobs) if n_jobs > 0 else n_jobs
    results = Parallel(n_jobs=n_jobs_eff, backend=PARALLEL_BACKEND)(delayed(predict_one)(m) for m in models)
    return {m.class_name: r for m, r in zip(models, results)}


def get_model_summary(models: List[FittedClassModel]) -> Dict[str, str]:
    """Human-readable per-class model diagnostics (R: getModelSummary)."""
    summaries: Dict[str, str] = {}
    for model in models:
        est = model.estimator
        lines = [f"Class: {model.class_name}  Model: {model.model_type}"]
        lines.append(f"Positive samples: {model.n_positive}")
        if isinstance(est, (LogisticRegression, LinearRegression)):
            coefs = np.ravel(est.coef_)
            intercept = np.ravel(est.intercept_)[0] if np.ndim(est.intercept_) else est.intercept_
            names = ["(Intercept)"] + list(model.driver_names)
            values = [intercept] + list(coefs)
            if model.pvalues is not None and len(model.pvalues) == len(values):
                for name, value, pvalue in zip(names, values, model.pvalues):
                    lines.append(f"{name}\t{value:.6g}\t{pvalue:.6g}\t{_significance_star(pvalue)}")
            else:
                for name, value in zip(names, values):
                    lines.append(f"{name}\t{value:.6g}")
        elif isinstance(est, RandomForestClassifier):
            lines.append("Variable\tImportance")
            for name, imp in zip(model.driver_names, est.feature_importances_):
                lines.append(f"{name}\t{imp:.6g}")
        elif isinstance(est, SVC):
            lines.append(f"Support vectors: {est.n_support_.tolist()}")
        summaries[model.class_name] = "\n".join(lines)
    return summaries
