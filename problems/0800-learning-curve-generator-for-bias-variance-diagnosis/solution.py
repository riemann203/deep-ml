import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    train_errors = []
    val_errors = []

    X_train = np.asarray(X_train).ravel()
    X_val = np.asarray(X_val).ravel()

    Phi_train = np.array(
        [
            [x ** i for i in range(degree+1)]
            for x in X_train
        ]
    )
    Phi_val = np.array(
        [
            [x ** i for i in range(degree+1)]
            for x in X_val
        ]
    )


    for size in train_sizes:
        selected_Phi_train = Phi_train[:size, :]
        selected_y_train = y_train[:size]
        best_weights = np.linalg.pinv(selected_Phi_train) @ selected_y_train

        train_error = np.mean((selected_y_train - selected_Phi_train @ best_weights) ** 2)
        val_error = np.mean((y_val - Phi_val @ best_weights) ** 2)
        train_errors.append(train_error)
        val_errors.append(val_error)

    final_train_error = train_errors[-1]
    final_val_error = val_errors[-1]
    if final_train_error > bias_threshold:
        diagnosis = 'high_bias'
    elif (final_val_error - final_train_error) > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'

    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'diagnosis': diagnosis
    }