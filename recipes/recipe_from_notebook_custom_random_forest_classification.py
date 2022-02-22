# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd, numpy as np

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# load 'train' dataset as a Pandas dataframe
df = dataiku.Dataset("train").get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#-----------------------------------------------------------------
# Dataset Settings
#-----------------------------------------------------------------

# Select a subset of features to use for training
SCHEMA = {    
    'target': 'high_value',    
    'features_num': ['age', 'price_first_item_purchased', 'pages_visited'],    
    'features_cat': ['gender', 'campaign']    
}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#-----------------------------------------------------------------
# Preprocessing on Training Set
#-----------------------------------------------------------------

# Numerical variables
df_num = df[SCHEMA['features_num']]

trf_num = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sts', StandardScaler()),
])

# Categorical variables
df_cat = df[SCHEMA['features_cat']]

trf_cat = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", trf_num, SCHEMA['features_num']),
        ("cat", trf_cat, SCHEMA['features_cat'])
    ]
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#-------------------------------------------------------------------------
# TRAINING
#-------------------------------------------------------------------------

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("clf", RandomForestClassifier())]
)

param_grid = {
    "clf__max_depth"        : [3, None],
    "clf__max_features"     : [1, 3, 5],
    "clf__min_samples_split": [2, 3, 10],
    "clf__min_samples_leaf" : [1, 3, 10],
    "clf__bootstrap"        : [True, False],
    "clf__criterion"        : ["gini", "entropy"],
    "clf__n_estimators"     : [10]
}

gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, scoring='roc_auc', cv=3)
X = df[SCHEMA['features_num'] + SCHEMA['features_cat']]
Y = df[SCHEMA['target']].values
gs.fit(X, Y)
clf = gs.best_estimator_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# #-----------------------------------------------------------------
# # Score Test Set
# #-----------------------------------------------------------------

# # load 'test' dataset as a Pandas dataframe
# df_test = dataiku.Dataset("to_assess_prepared").get_dataframe()

# # Actually score the new records
# scores = clf.predict_proba(df_test)

# # Reshape
# preds = pd.DataFrame(scores, index=df_test.index).rename(columns={0: 'proba_False', 1: 'proba_True'})
# all_preds = df_test.join(preds)

# # Sample of the test dataset with predicted probabilities
# all_preds.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # Compute AUC results
# auc = roc_auc_score(all_preds['high_value'].astype(bool).values, all_preds['proba_True'].values)
# auc

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
custom_random_forest = dataiku.Folder("CwRNawNH").get_path()

for file in os.listdir(custom_random_forest):
    try: os.remove(file)
    except: pass

from sklearn.externals import joblib
fp = os.path.join(custom_random_forest, "clf.pkl")
joblib.dump(clf, fp)
