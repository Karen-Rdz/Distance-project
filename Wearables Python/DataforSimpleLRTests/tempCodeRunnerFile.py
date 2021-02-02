scores = cross_val_score(clf_svc, X, y, cv=5)
# print("Accruracy: ", scores)