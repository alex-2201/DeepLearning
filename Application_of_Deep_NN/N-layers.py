import numpy as np
import matplotlib.pyplot as plt
from Application_of_Deep_NN.dnn_app_utils_v3 import load_data, print_mislabeled_images
from Deep_NN.deep_NN_model import L_model_backward, L_model_forward, compute_cost, update_parameters, \
    initialize_parameters_deep, predict

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model


# GRADED FUNCTION: L_layer_model
def L_layer_model(X, Y, layers_dims, learning_rate=0.75, num_iterations=3000, print_cost=False):  # lr was 0.009
    np.random.seed(1)
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
print_mislabeled_images(classes, test_x, test_y, pred_test)
