import numpy as np 
from lr_utils import load_dataset
import matplotlib.pyplot as plt 
import scipy 
from scipy import ndimage



class Cat_vs_Noncat(object):
    """
    Class Cat_vs_Noncat provides all the necessary implemention of menthods and attributes for a Neural Network to learn and differentiate
    between image of a Cat and Noncat.

    """


    def __init__(self, epochs, learning_rate):
        """
        Initiates the class Cat_vs_Noncat and initializes the values of epochs and learning_rate.

        Arguments:
        epoch -- Number of iterations (Hyperparameter)
        lerning_rate -- Learning rate for algorithm (Rate at which algorithm learns) (Hyperparameter)

        """

        self.epochs = epochs
        self.learning_rate = learning_rate



    def sigmoid(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size

        Returns:
        s -- sigmoid(z)

        """

        s = 1 / (1 + np.exp(-z))

        return s

    

    def initialize(self, dim):
        """
        This function creates vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Arguments:
        dim -- Size of w vector we want (or number of parameter in this case)

        Returns:

        """

        self.w = np.zeros((dim, 1))
        self.b = 0

        assert(self.w.shape == (dim, 1))
        assert(isinstance(self.b, float) or isinstance(self.b, int))


    
    def propagate(self, X, Y):
        """
        Implement cost function and its gradient.

        Arguments:
        X -- Input data of shape (num_px * num_px * 3, m)
        Y -- Label vector of shape (1, m) containing 1 for cat and 0 for non-cat

        Returns:
        cost -- total logistic loss for given weight(w) and bias(b)
        dw -- gradient of the loss wrt w, thus same shape as w
        db -- gradient of the loss wrt b, thus same shape as b

        """

        m = X.shape[1]

        A = self.sigmoid(self.w.T @ X + self.b)
        cost = -np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m

        dZ = A - Y
        dw = (X @ dZ.T)  / m
        db = np.sum(dZ) / m

        assert(dw.shape == self.w.shape)
        assert(db.dtype == float)

        cost = np.squeeze(cost)

        assert(cost.shape == ())

        grads = {"dw": dw,
        "db": db}

        return grads, cost



    def optimize(self, X, Y):
        """
        This method optimizes w and b by running gradient descent algorithm.

        Arguments:
        X -- Input data of shape (num_px * num_px * 3, m)
        Y -- Label vector of shape (1, m) containing 1 for cat and 0 for non-cat

        Returns:
        costs -- list of cost at every 100th iteration

        """

        costs = []

        for i in range(self.epochs):

            grads, cost = self.propagate(X, Y)
            dw = grads["dw"]
            db = grads["db"]

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if i % 100 == 0:

                costs.append(cost)

        return costs



    def predict(self, X):
        """
        This method predicts the output based on the learned parameters w and b

        Arguments:
        X -- Input data of shape (num_px * num_px * 3, m)

        Returns:
        y_predicted -- Retruns array of predicted outcomes of shape (1, m)

        """

        m = X.shape[1]

        A = self.sigmoid(self.w.T @ X + self.b)
        y_predicted = A > 0.5
        y_predicted = y_predicted.astype(int)

        assert(y_predicted.shape == (1, m))

        return y_predicted



# Loads the data set
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Shows an image of given index from train data
index = 9
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[0, index]) + ". It's a '" + classes[np.squeeze(train_set_y[0, index])].decode("utf-8") + "' picture")
plt.savefig("C:\\Users\\imraj\\Documents\\Cat-vs-Non-Cat\\outputs\\ex_fig.jpg")
plt.close()

# Calculate no. of training example, test example and size of a image
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Flatten given images into array of vector
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T 
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T 

# Standardizes the data by dividing pixel values by 255
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# Initiates the model
epochs = 2000
learning_rate = 0.004
catvsnotcat = Cat_vs_Noncat(epochs, learning_rate)

# Initialises the weight and bias values
dim = train_set_x.shape[0]
catvsnotcat.initialize(dim)

# Trains Model
costs = catvsnotcat.optimize(train_set_x, train_set_y)

# Predicts the class of given pictures
y_predicted_train = catvsnotcat.predict(train_set_x)
y_predicted_test = catvsnotcat.predict(test_set_x)

# Calculates the train and test accuracy
print(f"Train accuracy: {(100 - np.mean(np.abs(y_predicted_train - train_set_y))*100)}")
print(f"Test accuracy: {(100 - np.mean(np.abs(y_predicted_test - test_set_y))*100)}")

# Plots cost vs No. of iterations
plt.plot(costs)
plt.xlabel("Iterations(In multiple of 100")
plt.ylabel("Cost")
plt.title("Learning rate" + str(learning_rate))
plt.savefig("C:\\Users\\imraj\\Documents\\Cat-vs-Non-Cat\\outputs\\cost_vs_iteration.jpg")
plt.close()

# # Choice of different learning rates
# epochs = 2000
# learning_rates = [0.01, 0.001, 0.0001]
# costs_on_dif_lr = []

# for learning_rate in learning_rates:

#     catvsnotcat = Cat_vs_Noncat(epochs, learning_rate)
#     catvsnotcat.initialize(dim)
#     costs_on_dif_lr.append(catvsnotcat.optimize(train_set_x, train_set_y))

# # Plots Costs vs No. of iterations on different learning rate
# for i in range(len(learning_rates)):
#     plt.plot(costs_on_dif_lr[i], label = str(learning_rates[i]))

# plt.ylabel("Cost")
# plt.xlabel("Iteration(Hundereds")
# legend = plt.legend(loc = "best", shadow = True)
# frame = legend.get_frame()
# frame.set_facecolor("0.90")
# plt.savefig("C:\\Users\\imraj\\Documents\\Cat-vs-Non-Cat\\outputs\\cost_vs_iteration_dif_lr.jpg")
# plt.close()

# Predicting for own single image
fname = 'C:\\Users\\imraj\\Documents\\Cat-vs-Non-Cat\\images\\my_image2.jpg'
image = ndimage.imread(fname, flatten=False)
my_image = scipy.misc.imresize(image, size=(num_px, num_px, 3)).reshape((1, num_px * num_px * 3)).T
predicted_image = catvsnotcat.predict(my_image)
print("y = " + str(np.squeeze(predicted_image)) + ". This is a '" + classes[np.squeeze(predicted_image)].decode("utf-8") + "' image")
plt.imshow(image)
plt.savefig("C:\\Users\\imraj\\Documents\\Cat-vs-Non-Cat\\outputs\\test_img.jpg")
plt.close()