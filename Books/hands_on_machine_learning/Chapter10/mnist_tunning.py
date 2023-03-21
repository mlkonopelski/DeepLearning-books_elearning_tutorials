import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def import_data(from_library=True):
    if from_library:
        mnist = tf.keras.datasets.mnist

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        return X_train, y_train, X_val, y_val, X_test, y_val


def build_model(n_hidden=1, n_neurons=1, lr=0.001, input_shape=(28, 28)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    for hidden in range(n_hidden):
        model.add(tf.keras.layers.Dense(units=n_neurons, activation='relu'))

    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=['accuracy'])

    return model


def params_distribution():
    params = {
        'n_hidden': [1, 2, 3, 4, 5],
        'n_neurons': [5, 10, 20, 40, 60, 70, 100],
        'lr': [0.00001, 0.0001, 0.001]
    }

    return params


if __name__ == '__main__':
    N_ITER = 2
    CV = 2

    X_train, y_train, X_val, y_val, X_test, y_val = import_data()

    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)

    rnd_search_cv = RandomizedSearchCV(estimator=model,
                                       param_distributions=params_distribution(),
                                       n_iter=N_ITER,
                                       cv=CV,
                                       verbose=10,
                                       n_jobs=10)
    rnd_search_cv.fit(X=X_train,
                      y=y_train,
                      epochs=1000,
                      validation_data=(X_val, y_val),
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

    print('Best params:\n{}\nBest score:{}'.format(rnd_search_cv.best_params_,
                                                   rnd_search_cv.best_score_))

    best_model = rnd_search_cv.best_estimator_.model

    # PRINT OUT:
    # Best params:
    # {'n_neurons': 60, 'n_hidden': 2, 'lr': 0.001}
    # Best score:0.964555561542511

    # best_model.save('optimized_best_model.h5')
