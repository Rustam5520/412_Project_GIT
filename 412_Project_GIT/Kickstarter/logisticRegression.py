from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


LR = LogisticRegression()
LR.fit(train, y_train)

pred_log = np.rint(LR.predict(test))

confusion_matrix(pred_log, y_test)
