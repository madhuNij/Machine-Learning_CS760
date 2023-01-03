import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import seaborn as sns
from logistic_regression import Linear_LR
from support_vector_machine import SVM
from sklearn.datasets import make_circles, load_breast_cancer
from neural_network import NeuralNetwork
from sklearn.preprocessing import StandardScaler
#from kernel_logistic_regression import KLR
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC as LSVC



mu = [2.5]
#mu = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
knn_neighbours = 15
h = 0.02
cmap_light = ListedColormap(['orange', 'blue'])
cmap_bold = ['darkred','darkblue']

def plot_data_with_labels(x, y, ax):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = cmap_bold[li], zorder = 1, s = 8)

def plot_separator(ax, w, b):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + (slope * x_vals)
    ax.plot(x_vals, y_vals, 'k-')

def plot_margin(X1_train, X2_train, clf, ax):
    x = np.vstack((X1_train, X2_train))
    y = np.hstack((np.ones(X1_train.shape[0]), np.zeros(X2_train.shape[0]) ))
    plot_data_with_labels(x, y, ax)
    w, bias = clf.w, clf.b
    plot_separator(ax, w, bias)

def plot_contour1(x, y, model):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(x=x[:, 0], y=x[:, 1])
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y,
                    palette=cmap_bold, alpha=1.0, edgecolor="black")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_contour(X1_train, X2_train, clf, ax):
    x = np.vstack((X1_train, X2_train))
    y = np.hstack((np.ones(X1_train.shape[0]), np.zeros(X2_train.shape[0])))
    plot_data_with_labels(x, y, ax)
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, zorder = -1, cmap = 'gray')



def generate_2d_dataset(mean = 2.5):
    m1 = np.array([-mean, 0])
    m2 = np.array([mean, 0])
    cov = np.identity(2)
    x1 = np.random.multivariate_normal(m1, cov, 750)
    y1 = np.ones(len(x1))

    x2 = np.random.multivariate_normal(m2, cov, 750)
    y2 = np.ones(len(x2)) * -1
    return x1, y1, x2, y2




def synthetic_dataset_1():
    linear_svm_accuracies = []
    logistic_regression_accuracies = []
    knn_accuracies = []
    nb_accuracies = []
    for i in mu:
        x1, y1, x2, y2 = generate_2d_dataset(i)
        X = np.vstack((x1, x2))
        y = np.hstack((y1, y2))
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(1250), random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=int(1000), random_state=42)

        #Running Linear SVM on dataset
        svm_model = SVM(kernel='linear')
        svm_model.fit(X_train, y_train)

        svm_predict = svm_model.predict(X_test)
        svm_correct = np.sum(svm_predict == y_test)
        svm_accuracy = (svm_correct/len(svm_predict)) * 100
        linear_svm_accuracies.append(svm_accuracy)
        print("Linear SVM accuracy:", svm_accuracy)
        fig, ax = plt.subplots()
        plot_margin(X_train[y_train == 1], X_train[y_train == -1], svm_model, ax)
        ax.set_title("Linear SVM")
        plt.show()

        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        #Running Logistic regression on the data
        logistic_regression_model = Linear_LR(batch_size=20, epochs=1500, learning_rate=0.01)
        logistic_regression_model.fit(X_train, y_train)
        lr_predict = logistic_regression_model.predict(X_test)
        lr_correct = np.sum(lr_predict == y_test)
        lr_accuracy = (lr_correct/len(lr_predict)) * 100
        logistic_regression_accuracies.append(lr_accuracy)
        print("Logistic Regression accuracy :", lr_accuracy )
        fig, ax = plt.subplots()
        plot_margin(X_train[y_train == 1], X_train[y_train == 0], logistic_regression_model, ax)
        ax.set_title("Logistic Regression")
        plt.show()

        #Running kNN on the data
        knn_model = KNeighborsClassifier(knn_neighbours)
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)
        knn_correct_predict = np.sum(knn_pred == y_test)
        knn_accuracy = (knn_correct_predict/len(knn_pred)) * 100
        knn_accuracies.append(knn_accuracy)
        print("Knn accuracy:", knn_accuracy)
        #plot_contour1(X, y, knn_model)
        fig, ax = plt.subplots()
        plot_contour(X_train[y_train==1], X_train[y_train==0], knn_model, ax)
        ax.set_title("K-Nearest Neighbours")
        plt.show()

        #Running Naive Bayes on the data
        nB_model = GaussianNB()
        nB_model.fit(X_train, y_train)
        nB_predict = nB_model.predict(X_test)
        nB_correct_predict = np.sum(nB_predict == y_test)
        nB_accuracy = (nB_correct_predict / len(nB_predict)) * 100
        nb_accuracies.append(nB_accuracy)
        print("Naive Bayes accuracy:", nB_accuracy)
        fig, ax = plt.subplots()
        plot_contour(X_train[y_train==1], X_train[y_train==0], nB_model, ax)
        ax.set_title("Naive Bayes")
        plt.show()

    models = [linear_svm_accuracies, logistic_regression_accuracies, knn_accuracies, nb_accuracies]
    d_sets = ["Linear SVM", "Logistic Regression", "KNN", "Naive Bayes"]
    j = 0
    for i in models:
        plt.plot(mu, i, label=d_sets[j])
        j +=1
    plt.xlabel("Mean")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



def synthetic_dataset_2():
    X, y = make_circles(n_samples=1500, random_state=42)
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(1250), random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=int(1000), random_state=42)

    # Running Linear SVM on dataset
    svm_model = SVM(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_predict = svm_model.predict(X_test)
    svm_correct = np.sum(svm_predict == y_test)
    svm_accuracy = (svm_correct / len(svm_predict)) * 100
    print("Linear SVM accuracy:", svm_accuracy)
    fig, ax = plt.subplots()
    plot_margin(X_train[y_train == 1], X_train[y_train == -1], svm_model, ax)
    ax.set_title("Linear SVM")
    plt.show()

    # Running Polynomial Kernel SVM
    svm_kernel_polynomial = SVM(kernel='polynomial')
    svm_kernel_polynomial.fit(X_train, y_train)
    svm_kernel_polynomial_predict = svm_kernel_polynomial.predict(X_test)
    svm_kernel_polynomial_correct = np.sum(svm_kernel_polynomial_predict == y_test)
    svm_kernel_polynomial_accuracy = (svm_kernel_polynomial_correct / len(svm_kernel_polynomial_predict)) * 100
    print("Polynomial Kernel SVM accuracy:", svm_kernel_polynomial_accuracy)
    fig, ax = plt.subplots()
    plot_contour(X_train[y_train == 1], X_train[y_train == -1], svm_kernel_polynomial, ax)
    ax.set_title("Polynomial Kernel SVM")
    plt.show()

    # Running RBF Kernel SVM
    svm_kernel_rbf = SVM(kernel='rbf')
    svm_kernel_rbf.fit(X_train, y_train)
    svm_kernel_rbf_predict = svm_kernel_rbf.predict(X_test)
    svm_kernel_rbf_correct = np.sum(svm_kernel_rbf_predict == y_test)
    svm_kernel_rbf_accuracy = (svm_kernel_rbf_correct / len(svm_kernel_rbf_predict)) * 100
    print("RBF Kernel accuracy:", svm_kernel_rbf_accuracy)
    fig, ax = plt.subplots()
    plot_contour(X_train[y_train == 1], X_train[y_train == -1], svm_kernel_rbf, ax)
    ax.set_title("RBF Kernel SVM")
    plt.show()

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    # Running Logistic regression on the data
    logistic_regression_model = Linear_LR(batch_size=30, epochs=1500, learning_rate=0.01)
    #logistic_regression_model = LR(batch_size=30, epochs=1500, learning_rate=0.04)
    logistic_regression_model.fit(X_train, y_train)
    lr_predict = logistic_regression_model.predict(X_test)
    lr_correct = np.sum(lr_predict == y_test)
    lr_accuracy = (lr_correct / len(lr_predict)) * 100
    print("Logistic Regression accuracy :", lr_accuracy)
    fig, ax = plt.subplots()
    plot_margin(X_train[y_train == 1], X_train[y_train == 0], logistic_regression_model, ax)
    ax.set_title("Logistic Regression")
    plt.show()
    '''
    #Kernel Logistic Regression
    kernel_logistic_regression = KLR(epochs=50, learning_rate=0.01)
    kernel_logistic_regression.fit(X_train, y_train)
    kernel_logistic_regression_predict = kernel_logistic_regression.predict(X_test)
    kernel_logistic_regression_predict = np.array([1 if i > 0.5 else 0 for i in kernel_logistic_regression_predict])
    kernel_logistic_regression_correct = np.sum(kernel_logistic_regression_predict == y_test)
    kernel_logistic_regression_accuracy = (kernel_logistic_regression_correct/len(kernel_logistic_regression_predict)) * 100
    print("Kernel Logistic Regression accuracy:", kernel_logistic_regression_accuracy)
    fig, ax = plt.subplots()
    plot_contour(X_train[y_train == 1], X_train[y_train == -1], kernel_logistic_regression, ax)
    ax.set_title("Kernel Logistic Regression")
    plt.show()
    '''
    # Running kNN on the data
    knn_model = KNeighborsClassifier(knn_neighbours)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_correct_predict = np.sum(knn_pred == y_test)
    knn_accuracy = (knn_correct_predict / len(knn_pred)) * 100
    print("Knn accuracy:", knn_accuracy)
    fig, ax = plt.subplots()
    plot_contour(X_train[y_train==1], X_train[y_train==0], knn_model, ax)
    ax.set_title("K-Nearest Neighbours")
    plt.show()


    #Running NN on the data
    nn_model = NeuralNetwork(epochs=100, lr=0.3)
    nn_model.train_model(X_train, y_train)
    nn_predict = nn_model.predict(X_test)
    nn_correct = np.sum(nn_predict == y_test)
    nn_accuracy = (nn_correct/len(nn_predict)) * 100
    print("Neural Network accuracy:", nn_accuracy)
    fig, ax = plt.subplots()
    plot_contour(X_train[y_train==1], X_train[y_train==0], nn_model, ax)
    ax.set_title("Neural Network")
    plt.show()
    print("All decision boundaries")

def get_cancer_data():
  data = load_breast_cancer()
  x = data.data
  y = (data.target*2)-1
  x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.33, stratify=y)
  x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)
  return x_train, y_train, x_test, y_test, x_val, y_val


def real_dataset():
    dataload = load_breast_cancer(return_X_y=False)
    X = dataload.data
    y = dataload.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #L1 regularization
    canc_params = get_cancer_data()
    labels = load_breast_cancer().feature_names
    lsvc = LSVC(penalty='l1', dual=False, C=1e10, max_iter=100000)
    lsvc.fit(canc_params[0], canc_params[1])
    tar = np.abs(lsvc.coef_) > 1
    print(len(labels[tar.flatten()]))
    print("The relevant labels are:", labels[tar.flatten()])

    #Running Logistic Regression on data
    logistic_regression_model = Linear_LR(batch_size=30, epochs=1500, learning_rate=0.01)
    logistic_regression_model.fit(X_train, y_train)
    lr_predict = logistic_regression_model.predict(X_test)
    lr_correct = np.sum(lr_predict == y_test)
    lr_accuracy = (lr_correct / len(lr_predict)) * 100
    print("Logistic Regression accuracy :", lr_accuracy)

    # Running kNN on the data
    knn_model = KNeighborsClassifier(knn_neighbours)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_correct_predict = np.sum(knn_pred == y_test)
    knn_accuracy = (knn_correct_predict / len(knn_pred)) * 100
    print("Knn accuracy:", knn_accuracy)

    # Running NN on the data
    nn_model = NeuralNetwork(input=X_train.shape[1], epochs=100, lr=0.3)
    nn_model.train_model(X_train, y_train)
    nn_predict = nn_model.predict(X_test)
    nn_correct = np.sum(nn_predict == y_test)
    nn_accuracy = (nn_correct / len(nn_predict)) * 100
    print("Neural Network accuracy:", nn_accuracy)

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    # Running Linear SVM on dataset
    svm_model = SVM(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_predict = svm_model.predict(X_test)
    svm_correct = np.sum(svm_predict == y_test)
    svm_accuracy = (svm_correct / len(svm_predict)) * 100
    print("Linear SVM accuracy:", svm_accuracy)

    # Running Polynomial Kernel SVM
    svm_kernel_polynomial = SVM(kernel='polynomial')
    svm_kernel_polynomial.fit(X_train, y_train)
    svm_kernel_polynomial_predict = svm_kernel_polynomial.predict(X_test)
    svm_kernel_polynomial_correct = np.sum(svm_kernel_polynomial_predict == y_test)
    svm_kernel_polynomial_accuracy = (svm_kernel_polynomial_correct / len(svm_kernel_polynomial_predict)) * 100
    print("Polynomial Kernel SVM accuracy:", svm_kernel_polynomial_accuracy)

    # Running RBF Kernel SVM
    svm_kernel_rbf = SVM(kernel='rbf')
    svm_kernel_rbf.fit(X_train, y_train)
    svm_kernel_rbf_predict = svm_kernel_rbf.predict(X_test)
    svm_kernel_rbf_correct = np.sum(svm_kernel_rbf_predict == y_test)
    svm_kernel_rbf_accuracy = (svm_kernel_rbf_correct / len(svm_kernel_rbf_predict)) * 100
    print("RBF Kernel accuracy:", svm_kernel_rbf_accuracy)



if __name__ == "__main__":
    synthetic_dataset_1()
    synthetic_dataset_2()
    real_dataset()

