import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 15
TEST_SIZE = 0.20


def load_and_prepare_data():
    df = sns.load_dataset("tips").copy()

    # Binary encoding
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["smoker"] = df["smoker"].map({"Yes": 1, "No": 0})
    df["time"] = df["time"].map({"Dinner": 1, "Lunch": 0})

    # One-hot encoding for day
    df = pd.get_dummies(df, columns=["day"], drop_first=True, dtype=int)

    return df


def plot_correlation_heatmap(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }

    return results, y_pred


def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Tip")
    plt.ylabel("Predicted Tip")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compare_basic_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "LassoCV": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, max_iter=10000))
        ]),
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5))
        ]),
        "ElasticNetCV": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(
                cv=5,
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                max_iter=10000
            ))
        ]),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        predictions[name] = y_pred

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)

    return models, results_df, predictions


def compare_polynomial_models(X_train, X_test, y_train, y_test, degrees):
    results = {}
    predictions = {}

    for degree in degrees:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        metrics, y_pred = evaluate_model(
            model, X_train, X_test, y_train, y_test,
            f"Polynomial Regression (degree={degree})"
        )

        results[degree] = metrics
        predictions[degree] = y_pred

    results_df = pd.DataFrame(results.values()).sort_values(by="R2", ascending=False).reset_index(drop=True)

    return results_df, predictions


def main():
    df = load_and_prepare_data()

    print("First 5 rows:")
    print(df.head())
    print("\nData shape:", df.shape)

    plot_correlation_heatmap(df)

    X = df.drop("tip", axis=1)
    y = df["tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Basic models
    basic_models, basic_results, basic_predictions = compare_basic_models(
        X_train, X_test, y_train, y_test
    )

    print("\nBasic Model Results:")
    print(basic_results)

    # Best basic model plot
    best_basic_model_name = basic_results.loc[0, "Model"]
    plot_actual_vs_predicted(
        y_test,
        basic_predictions[best_basic_model_name],
        f"{best_basic_model_name} - Actual vs Predicted"
    )

    # Polynomial models
    poly_results, poly_predictions = compare_polynomial_models(
        X_train, X_test, y_train, y_test, degrees=[1, 2, 3, 4]
    )

    print("\nPolynomial Model Results:")
    print(poly_results)

    best_poly_name = poly_results.loc[0, "Model"]
    best_poly_degree = int(best_poly_name.split("=")[-1].replace(")", ""))

    plot_actual_vs_predicted(
        y_test,
        poly_predictions[best_poly_degree],
        f"Polynomial Regression (degree={best_poly_degree}) - Actual vs Predicted"
    )

    # Best alpha values from CV models
    print("\nBest hyperparameters:")
    print("LassoCV alpha:", basic_models["LassoCV"].named_steps["model"].alpha_)
    print("RidgeCV alpha:", basic_models["RidgeCV"].named_steps["model"].alpha_)
    print("ElasticNetCV alpha:", basic_models["ElasticNetCV"].named_steps["model"].alpha_)
    print("ElasticNetCV l1_ratio:", basic_models["ElasticNetCV"].named_steps["model"].l1_ratio_)


if __name__ == "__main__":
    main()