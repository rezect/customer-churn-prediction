from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

RND_SEED = 42

lr = LogisticRegression(random_state=RND_SEED)

lin_svc = LinearSVC(C=1, random_state=RND_SEED)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

rnd_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
