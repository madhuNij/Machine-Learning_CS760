import numpy as np
from matplotlib import pyplot as plt

def read_file_visualise():
    data_three = np.loadtxt("three.txt", dtype=int)
    plt.imshow(np.transpose(np.reshape(data_three[0], (16, 16))),cmap='gray', vmin=0, vmax=255)
    plt.show()

    data_eight = np.loadtxt("eight.txt", dtype=int)
    plt.imshow(np.transpose(np.reshape(data_eight[0], (16, 16))), cmap='gray', vmin=0, vmax=255)
    plt.show()

    return data_three, data_eight

def compute_mean(data_three, data_eight):
    X = np.vstack((data_three, data_eight))
    y = X.mean(axis = 0)
    plt.imshow(np.transpose(np.reshape(y, (16, 16))), cmap='gray', vmin=0, vmax=255)
    plt.show()
    return X, y

def center_dataset_covariance(X, y):
    centered_dataset = X - y
    x = np.array(centered_dataset)
    dataset_transpose = np.transpose(x)
    covariance_dataset = np.dot(dataset_transpose, x)
    S = covariance_dataset * (1 / (len(x) - 1))
    print("Covariance sub matrix:", S[0:5, 0:5])
    return S, centered_dataset

def get_eigenvalue_vector(covariance_matrix):
    eig_vals, eig_vectors = np.linalg.eigh(covariance_matrix)
    print("λ1 =", eig_vals[-1])
    print("λ2 =",eig_vals[-2])
    vec1 = np.reshape(eig_vectors[:,-1], (256,1))
    vec2 = np.reshape(eig_vectors[:,-2], (256,1))
    vec1 = ((vec1-np.min(vec1))/(np.max(vec1)-np.min(vec1))) * 255
    vec2 = ((vec2-np.min(vec2))/(np.max(vec2)-np.min(vec2))) * 255
    fig, ax1 = plt.subplots()
    pos1 = ax1.imshow(np.transpose(np.reshape(vec1, (16, 16))), cmap='gray', vmin=0, vmax=255)
    #fig.colorbar(pos1, ax=ax1)
    plt.show()
    fig, ax2 = plt.subplots()
    pos2 = ax2.imshow(np.transpose(np.reshape(vec2, (16, 16))), cmap='gray', vmin=0, vmax=255)
    #fig.colorbar(pos2, ax=ax2)
    plt.show()
    return eig_vectors

def project_data(X, eigh_vectors):
    vec1 = np.reshape(eigh_vectors[:, -1], (256, 1))
    vec2 = np.reshape(eigh_vectors[:, -2], (256, 1))
    V = np.hstack((vec1, vec2))
    proj = np.dot(X, V)
    three = proj[0]
    eight = proj[200]
    print("Coordinate from three: ",three)
    print("Coordinate from eight: ",eight)
    return proj

def plot_pca(proj):
    plt.scatter(proj[:200, 0], proj[:200, 1], label = 'three', c='r')
    plt.scatter(proj[200:, 0], proj[200:, 1], label = 'eight', c='b')
    plt.legend()
    plt.show()

def pca():
    data_three, data_eight = read_file_visualise()
    X, y = compute_mean(data_three, data_eight)
    covariance_matrix, centered_dataset = center_dataset_covariance(X, y)
    eigh_vectors = get_eigenvalue_vector(covariance_matrix)
    proj = project_data(centered_dataset, eigh_vectors)
    plot_pca(proj)


if __name__ == "__main__":
    pca()