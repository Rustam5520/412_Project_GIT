from sklearn.ensemble import RandomForestClassifier

clf_forest = RandomForestClassifier(max_depth=20, n_estimators=50)
clf_forest.fit(train, y_train)


print(clf_forest.score(test, y_test))

pred_for = np.rint(clf_forest.predict(test))

confusion_matrix(pred_for, y_test)
