from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

print 'Loading RF...'
rf = joblib.load('/RF/randomforest.pkl')

print rf.feature_importances_
print rf.get_params(deep=True)