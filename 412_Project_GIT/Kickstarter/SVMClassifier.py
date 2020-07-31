from sklearn.svm import LinearSVC

clf_svc = LinearSVC()
clf_svc.fit(train, y_train)
print(clf_svc.score(test, y_test))

predsc = np.rint(clf_svc.predict(test))
confusion_matrix(predsc, y_test)
