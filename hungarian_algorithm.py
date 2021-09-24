import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def plot_image(img, label, label2, no_clusters, name):
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,3,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2),plt.imshow(label)
    plt.title('Segmented Image when number of clusters = %i' % \
              no_clusters), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(label2)
    plt.title('Segmented Image when number of clusters = %i' % \
              no_clusters), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(name+'.png')
    plt.close()
    
def findIntersect(I2, clusterIDsB, tupelA_I, tupel):
    nA_I = len(tupelA_I)  # number of elements in cluster I
    tupelB_I = tupel[clusterIDsB == I2]
    nB_I = len(tupelB_I)
    nTupelIntersect = len(np.intersect1d(tupelA_I, tupelB_I))

    return ((nA_I - nTupelIntersect) + (nB_I - nTupelIntersect))

def minWeightBipartiteMatching(clusteringA, clusteringB):
    idsA = np.unique(clusteringA)
    idsB = np.unique(clusteringB)
    
    # number of instances
    nA = len(clusteringA)
    nB = len(clusteringB)
    if (len(idsA) != len(idsB)) | (nA != nB):
        print("number of cluster or number of instances do not match")
        return None
        
    nC = len(idsA)
    tupel = np.arange(nA)
    assignmentMatrix = -1*np.ones((nC, nC))
    for I in range(nC):
        
        tupelClusterI = tupel[clusteringA==I]
        solRowI = [findIntersect(I2, clusteringB, tupelClusterI, tupel) for I2 in range(nC)]
        assignmentMatrix[I, ] = solRowI
        
    row_ind, col_ind = linear_sum_assignment(assignmentMatrix)
    return row_ind, col_ind, assignmentMatrix


def main(image_name, factor, no_clusters, attempts):
    
    cv2img = cv2.imread(image_name)
    resized_image = cv2.resize(cv2img, (round(cv2img.shape[1]/factor), round(cv2img.shape[0]/factor))) 
    img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    vectorized = np.float32(resized_image.reshape((-1,3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret,label,center=cv2.kmeans(vectorized,no_clusters,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    label_image = label.reshape((round(img.shape[0]/factor), round(img.shape[1]/factor)))

    ret2,label2,center2=cv2.kmeans(
        vectorized,no_clusters,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS
    )
    label_image2 = label2.reshape((round(img.shape[0]/factor), round(img.shape[1]/factor)))

    plot_image(img, label_image, label_image2, no_clusters, name = 'before_hungarian')
    
    row_ind, col_ind, cost_matrix = minWeightBipartiteMatching(np.array(label_image.reshape(-1)), \
                           np.array(label_image2.reshape(-1)))
    
    new_image = label_image2.copy()

    for r, c in zip(row_ind, col_ind):
        new_image[label_image2==c] = r
        
    plot_image(img, label_image, new_image, no_clusters, name = 'after_hungarian')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--factor", help="factor to resize the image",
                        type=int, default=5, required=False)
    parser.add_argument("-c","--no_clusters", help="number of clusters in k means clustering",
                        type=int, default=4, required=False)
    parser.add_argument("-a","--attempts", help="number of attempts in k means clustering",
                        type=int, default=10, required=False)
    args = parser.parse_args()
    factor = args.factor
    no_clusters = args.no_clusters
    attempts=args.attempts
    main("poggies.JPG", factor, no_clusters, attempts)

