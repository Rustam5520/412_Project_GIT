import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# baseline model
def create_baseline():
	# create model
    model = Sequential()
    model.add(Dense(191, input_dim=191, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
	  # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=2, verbose=1)
#kfold = StratifiedKFold(n_splits=10, shuffle=True)
#results = cross_val_score(estimator, train, y_train, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

m = create_baseline()
m.fit(train, y_train, batch_size=1024, epochs=50, validation_split=0.1)



pred = np.rint(m.predict(test))

print(1 - (np.abs(pred.flatten() - y_test.to_numpy())).mean())

print(confusion_matrix(pred, y_test))
