from __future__ import annotations

import warnings
from functools import partial

import lightgbm
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from flaml import AutoML

# Money Patch LightGBM to mute its output during openfe
lightgbm.LGBMRegressor = partial(lightgbm.LGBMRegressor, verbosity=-1)
lightgbm.LGBMClassifier = partial(lightgbm.LGBMClassifier, verbosity=-1)

from openfe import openfe, transform

# -- Get Data
X_train = pd.read_csv("/train.csv")
X_test = pd.read_csv("./test.csv")
original_dataset = pd.read_csv("./data.csv", sep=";")
label = "Target"

# -- Add original data
original_dataset["id"] = np.arange(1, len(original_dataset) + 1) + 10000000
original_dataset.columns = [x.strip() for x in original_dataset.columns]
X_train = pd.concat([X_train, original_dataset[list(X_train.columns)]], axis=0).reset_index(drop=True)


# -- Run OpenFE
with warnings.catch_warnings():
    # from openfe import get_candidate_features # use this to reduce time OpenFE needs (see below)

    from sklearn.exceptions import DataConversionWarning

    warnings.simplefilter("ignore", category=DataConversionWarning)
    ofe = openfe()
    features = ofe.fit(
        data=X_train.drop(columns=[label]),
        label=X_train[label].to_frame(),
        task="classification",
        # candidate_features_list=get_candidate_features(numerical_features=list(X_train)[:-1])[:5000],
    )[:32]  # restrict to 32 best features
    y = X_train[label].copy()
    X_train, X_test = transform(X_train, X_test, features, n_jobs=4)
    X_train[label] = y

# -- Set Categorical Features
nominal_features = [
    "Marital status",
    "Application mode",
    "Course",
    "Nacionality",
    "Mother's occupation",
    "Father's occupation",
    "Educational special needs",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
X_train[nominal_features] = X_train[nominal_features].astype("category")
X_test[nominal_features] = X_test[nominal_features].astype("category")

# -- Only use Neural Networks at Layer 2
hyperparameters_raw = get_hyperparameter_config("zeroshot")
hyperparameters_only_nn = {"NN_TORCH": hyperparameters_raw["NN_TORCH"], "FASTAI": hyperparameters_raw["FASTAI"]}

hyperparameters = {
    1: hyperparameters_raw,
    "default": hyperparameters_only_nn,
}

# -- Run AutoGluon
predictor = TabularPredictor(
    label=label,
    eval_metric="accuracy",
    problem_type="multiclass",
    verbosity=2,
)

predictor.fit(
    time_limit=60 * 60 * 3,
    train_data=X_train,
    presets="best_quality",
    dynamic_stacking=False,
    hyperparameters=hyperparameters,
    # Early Stopping
    ag_args_fit={
        "stopping_metric": "log_loss",
    },
    # Validation Protocol
    num_bag_sets=2,
    num_stack_levels=1,
)
predictor.fit_summary(verbosity=1)

predict_proba_ag = predictor.predict_proba(X_test)

# -- Run FLAML's Lottery Ticket Model (https://www.kaggle.com/code/gauravduttakiit/pss4e6-flaml-roc-auc-ovo, Version from June 1, 2024 at 12:13 PM)
X_train = pd.read_csv("./train.csv")
X_test = pd.read_csv("./test.csv")

# Reproduce data from the notebook
for df in [X_train, X_test]:
    df["Marital status"] = df["Marital status"].replace(
        {1: "single", 2: "married", 3: "widower", 4: "divorced", 5: "facto union", 6: "legally separated"},
    )

    df["Application mode"] = df["Application mode"].replace(
        {
            1: "1st phase - general contingent",
            2: "Ordinance No. 612/93",
            5: "1st phase - special contingent (Azores Island)",
            7: "Holders of other higher courses",
            10: "Ordinance No. 854-B/99",
            15: "International student (bachelor)",
            16: "1st phase - special contingent (Madeira Island)",
            17: "2nd phase - general contingent",
            18: "3rd phase - general contingent",
            26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
            27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
            39: "Over 23 years old",
            42: "Transfer",
            43: "Change of course",
            44: "Technological specialization diploma holders",
            51: "Change of institution/course",
            53: "Short cycle diploma holders",
            57: "Change of institution/course (International)",
        },
    )

    df["Course"] = df["Course"].replace(
        {
            33: "Biofuel Production Technologies",
            171: "Animation and Multimedia Design",
            8014: "Social Service (evening attendance)",
            9003: "Agronomy",
            9070: "Communication Design",
            9085: "Veterinary Nursing",
            9119: "Informatics Engineering",
            9130: "Equinculture",
            9147: "Management",
            9238: "Social Service",
            9254: "Tourism",
            9500: "Nursing",
            9556: "Oral Hygiene",
            9670: "Advertising and Marketing Management",
            9773: "Journalism and Communication",
            9853: "Basic Education",
            9991: "Management (evening attendance)",
        },
    )
    df["Daytime/evening attendance"] = df["Daytime/evening attendance"].replace({1: "daytime", 0: "evening"})

    df["Previous qualification"] = df["Previous qualification"].replace(
        {
            1: "Secondary education",
            2: "Higher education - bachelor's degree",
            3: "Higher education - degree",
            4: "Higher education - master's",
            5: "Higher education - doctorate",
            6: "Frequency of higher education",
            9: "12th year of schooling - not completed",
            10: "11th year of schooling - not completed",
            12: "Other - 11th year of schooling",
            14: "10th year of schooling",
            15: "10th year of schooling - not completed",
            19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
            38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
            39: "Technological specialization course",
            40: "Higher education - degree (1st cycle)",
            42: "Professional higher technical course",
            43: "Higher education - master (2nd cycle)",
        },
    )

    df["Nacionality"] = df["Nacionality"].replace(
        {
            1: "Portuguese",
            2: "German",
            6: "Spanish",
            11: "Italian",
            13: "Dutch",
            14: "English",
            17: "Lithuanian",
            21: "Angolan",
            22: "Cape Verdean",
            24: "Guinean",
            25: "Mozambican",
            26: "Santomean",
            32: "Turkish",
            41: "Brazilian",
            62: "Romanian",
            100: "Moldova (Republic of)",
            101: "Mexican",
            103: "Ukrainian",
            105: "Russian",
            108: "Cuban",
            109: "Colombian",
        },
    )
    df = df.rename({"Nacionality": "Nationality"}, axis="columns")
    df["Mother's qualification"] = df["Mother's qualification"].replace(
        {
            1: "Secondary Education - 12th Year of Schooling or Eq.",
            2: "Higher Education - Bachelor's Degree",
            3: "Higher Education - Degree",
            4: "Higher Education - Master's",
            5: "Higher Education - Doctorate",
            6: "Frequency of Higher Education",
            9: "12th Year of Schooling - Not Completed",
            10: "11th Year of Schooling - Not Completed",
            11: "7th Year (Old)",
            12: "Other - 11th Year of Schooling",
            14: "10th Year of Schooling",
            18: "General commerce course",
            19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
            22: "Technical-professional course",
            26: "7th year of schooling",
            27: "2nd cycle of the general high school course",
            29: "9th Year of Schooling - Not Completed",
            30: "8th year of schooling",
            34: "Unknown",
            35: "Can't read or write",
            36: "Can read without having a 4th year of schooling",
            37: "Basic education 1st cycle (4th/5th year) or equiv.",
            38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
            39: "Technological specialization course",
            40: "Higher education - degree (1st cycle)",
            41: "Specialized higher studies course",
            42: "Professional higher technical course",
            43: "Higher Education - Master (2nd cycle)",
            44: "Higher Education - Doctorate (3rd cycle)",
        },
    )
    df["Father's qualification"] = df["Father's qualification"].replace(
        {
            1: "Secondary Education - 12th Year of Schooling or Eq.",
            2: "Higher Education - Bachelor's Degree",
            3: "Higher Education - Degree",
            4: "Higher Education - Master's",
            5: "Higher Education - Doctorate",
            6: "Frequency of Higher Education",
            9: "12th Year of Schooling - Not Completed",
            10: "11th Year of Schooling - Not Completed",
            11: "7th Year (Old)",
            12: "Other - 11th Year of Schooling",
            13: "2nd year complementary high school course",
            14: "10th Year of Schooling",
            18: "General commerce course",
            19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
            20: "Complementary High School Course",
            22: "Technical-professional course",
            25: "Complementary High School Course - not concluded",
            26: "7th year of schooling",
            27: "2nd cycle of the general high school course",
            29: "9th Year of Schooling - Not Completed",
            30: "8th year of schooling",
            31: "General Course of Administration and Commerce",
            33: "Supplementary Accounting and Administration",
            34: "Unknown",
            35: "Can't read or write",
            36: "Can read without having a 4th year of schooling",
            37: "Basic education 1st cycle (4th/5th year) or equiv.",
            38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
            39: "Technological specialization course",
            40: "Higher education - degree (1st cycle)",
            41: "Specialized higher studies course",
            42: "Professional higher technical course",
            43: "Higher Education - Master (2nd cycle)",
            44: "Higher Education - Doctorate (3rd cycle)",
        },
    )
    df["Admission grade"] = df["Admission grade"].astype("int")
    df["Displaced"] = df["Displaced"].replace({1: "yes", 0: "no"})
    df["Educational special needs"] = df["Educational special needs"].replace({1: "yes", 0: "no"})
    df["Debtor"] = df["Debtor"].replace({1: "yes", 0: "no"})
    df["Tuition fees up to date"] = df["Tuition fees up to date"].replace({1: "yes", 0: "no"})
    df["Gender"] = df["Gender"].replace({1: "male", 0: "female"})
    df["Scholarship holder"] = df["Scholarship holder"].replace({1: "yes", 0: "no"})
    df["Age at enrollment"] = df["Age at enrollment"].astype("int")
    df["International"] = df["International"].replace({1: "yes", 0: "no"})

X_train = X_train.drop(columns=["id"])

X_train["Application mode"] = X_train["Application mode"].replace(
    {12: np.NaN, 4: np.NaN, 35: np.NaN, 9: np.NaN, 3: np.NaN},
)
X_test["Application mode"] = X_test["Application mode"].replace(
    {14: np.NaN, 35: np.NaN, 19: np.NaN, 3: np.NaN},
)
X_train = X_train[X_train["Application mode"] != "Ordinance No. 533-A/99, item b2) (Different Plan)"]
X_train["Application mode"] = X_train["Application mode"].replace(
    {"Ordinance No. 533-A/99, item b3 (Other Institution)": "1st phase - general contingent"},
)
X_test["Application mode"] = X_test["Application mode"].replace(
    {"Ordinance No. 533-A/99, item b3 (Other Institution)": np.NAN},
)
X_train["Course"] = X_train["Course"].replace({979: np.NaN, 39: np.NaN})
X_test["Course"] = X_test["Course"].replace({7500: np.NaN, 9257: np.NaN, 2105: np.NaN, 4147: np.NaN})
X_train["Previous qualification"] = X_train["Previous qualification"].replace(
    {37: np.NaN, 36: np.NaN, 17: np.NaN, 11: np.NaN},
)
X_test["Previous qualification"] = X_test["Previous qualification"].replace({17: np.NaN, 11: np.NaN, 16: np.NaN})
X_test["Nationality"] = X_test["Nationality"].replace({"English": np.NaN})
X_train["Nationality"] = X_train["Nationality"].replace({"Lithuanian": np.NaN})
X_train["Nationality"] = X_train["Nationality"].replace({"Turkish": np.NaN})
X_test["Nationality"] = X_test["Nationality"].replace({"Turkish": np.NaN})
X_train["Mother's qualification"] = X_train["Mother's qualification"].replace(
    {15: np.NaN, 7: np.NaN, 33: np.NaN, 8: np.NaN, 28: np.NaN, 31: np.NaN},
)
X_test["Mother's qualification"] = X_test["Mother's qualification"].replace(
    {25: np.NaN, 13: np.NaN, 31: np.NaN, 33: np.NaN},
)
X_train["Mother's qualification"] = X_train["Mother's qualification"].replace(
    {
        "2nd cycle of the general high school course": np.NaN,
    },
)
X_train["Father's qualification"] = X_train["Father's qualification"].replace(
    {21: np.NaN, 7: np.NaN, 23: np.NaN, 15: np.NaN, 24: np.NaN},
)
X_test["Father's qualification"] = X_test["Father's qualification"].replace(
    {16: np.NaN, 28: np.NaN, 7: np.NaN, 21: np.NaN},
)
X_train["Father's qualification"] = X_train["Father's qualification"].replace(
    {"Higher Education - Doctorate (3rd cycle)": np.NaN, "Complementary High School Course": np.NaN},
)
X_train["Father's qualification"] = X_train["Father's qualification"].replace(
    {
        "Supplementary Accounting and Administration": np.NaN,
        "General Course of Administration and Commerce": np.NaN,
        "Complementary High School Course - not concluded": np.NaN,
    },
)
X_test["Father's qualification"] = X_test["Father's qualification"].replace(
    {
        "Supplementary Accounting and Administration": np.NaN,
        "General Course of Administration and Commerce": np.NaN,
        "Complementary High School Course - not concluded": np.NaN,
    },
)

# Reproduce model from the notebook and get OOF predictions
gbm_hyperparameters = {
    "n_estimators": 211,
    "num_leaves": 115,
    "min_child_samples": 4,
    "learning_rate": 0.0526991568211921,
    "log_max_bin": 8,
    "colsample_bytree": 0.4804131128760313,
    "reg_alpha": 0.0009765625,
    "reg_lambda": 0.029242177822527835,
}
starting_points = {"lgbm": gbm_hyperparameters}
automl = AutoML(eval_method="cv", max_iter=1)
automl.fit(
    X_train.drop(columns=[label]),
    X_train[label],
    task="classification",
    metric="roc_auc_ovo",
    time_budget=500,
    starting_points=starting_points,
)
predict_proba_flaml = automl.predict_proba(X_test.drop(columns=["id"]))

# -- Merge Predictions
final_proba = 0.14 * predict_proba_ag + 0.86 * predict_proba_flaml
submission = X_test["id"].to_frame()
submission["Target"] = pd.Series(np.argmax(final_proba, axis=1)).replace(
    {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate",
    },
)
submission.to_csv("submission.csv", index=False)
