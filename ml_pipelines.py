import sklearn.base
import sklearn.impute
import sklearn.compose
import sklearn.tree
import sklearn.pipeline
import sklearn.ensemble
import lightgbm


class CategoricalEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self,  columns):
        self.columns = columns
        self.encoder = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        )
        self.feature_names_in_ = []

    def fit(self, x, y):
        self.feature_names_in_ = x.columns
        xc = x.loc[:, self.columns].values
        self.encoder.fit(xc, y)
        return self

    def transform(self, x, y=None):
        xc = x.loc[:, self.columns].values
        x2 = self.encoder.transform(xc)
        result = x.copy()
        result.loc[:, self.columns] = x2
        return result

    def get_feature_names_out(self, input_feature=None):
        return self.feature_names_in_


class ProbPos(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):

    def __init__(self, base_classifer):
        self.base_classifier = base_classifer

    def __repr__(self, N_CHAR_MAX=700):
        return f"ProbPos({self.base_classifier})"

    def fit(self, x, y):
        self.base_classifier.fit(x, y)
        return self

    def predict(self, x):
        return self.base_classifier.predict(x)

    def predict_proba(self, x):
        prob_matrix = self.base_classifier.predict_proba(x)
        return prob_matrix[:, 1]


categorical_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]
preps = {
    'TreePreprocessing': [
        ('encode', CategoricalEncoder(categorical_columns)),
        ('impute', sklearn.impute.SimpleImputer(fill_value=0.0)),
    ],
}
classifiers = {
    'DecisionTree': sklearn.tree.DecisionTreeClassifier(),
    'RandomForest': sklearn.ensemble.RandomForestClassifier(),
    'LGBMClassifier': lightgbm.LGBMClassifier(n_estimators=50),
}
