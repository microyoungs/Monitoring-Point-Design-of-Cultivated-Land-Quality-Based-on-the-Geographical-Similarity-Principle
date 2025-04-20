import numpy as np
import matplotlib.pyplot as plt
import logging
import csv
from FileUtil import FileUtil

from osgeo import gdal

# log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Initialize the input data
def init_data(path):
    logger.info("init data start")
    # Read data from the file
    inputData = []
    with open(path, encoding="utf-8") as fin:
        reader = csv.reader(fin)
        # Skip the first line
        next(reader)
        for line in reader:
            inputData.append(line)

    # Convert a python array to an np matrix
    datas = np.array(inputData).astype(np.float32)
    logger.info("init data end")
    return datas

# Initialization matrix
def setup_datas(data_A, data_B):
    logger.info("setup datas start")
    # Read the number of rows and columns of the matrix
    N, D_A = np.shape(dataA)
    N, D_B = np.shape(dataB)

    distance = np.zeros((N, N))
    file_util = FileUtil("result_distance")
    for i in range(N):
        row_result = []
        logger.info("setup iter datas %d", i)
        for j in range(N):
            # Discrete data
            first_A_row = data_A[i]
            second_A_row = data_A[j]
            # Simple matching distance

            result_A = first_A_row == second_A_row
            max_A = 1-np.min(result_A)

            # sameNum = np.sum(first_A_row == second_A_row)
            # value_A = 1-float(sameNum) / float(D_A)

            # Continuous data
            first_B_row = data_B[i]
            second_B_row = data_B[j]
            # Gower distance
            differ = first_B_row - second_B_row
            abs_differ = np.absolute(differ)

            max_B = np.max(abs_differ)/100

            # differ_sum = np.sum(abs_differ)
            # value_B =float(differ_sum) / (100.0 * float(D_B))
            value = max(max_A, max_B)
            # value = (value_A + value_B) * 0.5
            row_result.append(value)

        distance[i] = np.array(row_result).astype(np.float32)
        # Write the distance matrix to the file
        file_util.write(row_result)
    # Handle the diagonals
    # m = np.median(S)
    # np.fill_diagonal(S, m)
    # logger.info("setup datas end")
    return distance


# Find the threshold dc for density calculation
# The average number of points with a distance less than dc around each point should account for 1% to 2% of the total number of points
def select_dc(distance):
    '''Algorithm 1'''
    N = np.shape(distance)[0]
    tt = np.reshape(distance, N * N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]

    ''' Algorithm 2 '''
    # N = np.shape(distance)[0]
    # max_dis = np.max(distance)
    # min_dis = np.min(distance)
    # dc = (max_dis + min_dis) / 2
    #
    # while True:
    #     n_neighs = np.where(distance<dc)[0].shape[0]-N
    #     rate = n_neighs/(N*(N-1))
    #
    #     if rate>=0.01 and rate<=0.02:
    #         break
    #     if rate<0.01:
    #         min_dis = dc
    #     else:
    #         max_dis = dc
    #
    #     dc = (max_dis + min_dis) / 2
    #     if max_dis - min_dis < 0.0001:
    #         break

    return dc

# Calculate the local density pi of each point
def get_density(distance, dc, method=None):
    N = np.shape(distance)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(distance[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(distance[i, :] / dc) ** 2)) - 1
    return rho


# Calculate the density distance of each data point
# That is, for each point, find all the points with a density greater than it
# Then find the distance of the nearest point among these points
def get_deltas(distance, rho):
    N = np.shape(distance)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # Sort the density from high to low
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # For the point with the highest density
        if i == 0:
            continue

        # For other points
        #  Find the serial number of the point with a density greater than it
        index_higher_rho = index_rho[:i]
        # Obtain the distances from these points to the current point and find the minimum value
        deltas[index] = np.min(distance[index, index_higher_rho])

        # Save the number of the nearest neighbor point
        index_nn = np.argmin(distance[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber

# Determine the category points, calculate the gamma value and draw the graph
def show_decision_chart(rho, deltas, centers):
    # Normalization
    normal_rho = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))
    normal_deltas = (deltas - np.min(deltas)) / (np.max(deltas) - np.min(deltas))
    gamma = normal_rho * normal_deltas
    file_util = FileUtil("result_gamma")
    plt.cla()
    print(np.shape(gamma)[0])

    plt.figure(num=2, figsize=(15, 10))
    x_data = range(len(deltas))
    y_data = -np.sort(-gamma)
    plt.scatter(x=x_data, y=y_data, c='k', marker='o', s=-np.sort(-gamma) * 100)
    plt.xlabel('data_num')
    plt.ylabel('gamma')
    plt.title('gamma')
    plt.savefig("./result/gamma.jpg",dpi=1200)
    # Write gamma to the file
    for row_gamma in gamma:
        file_util.write([row_gamma], show_log=False)
    return gamma

# Select the points where both rho and delta are large through the threshold
#  As a clustering center
def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) / 2
    delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
    N = np.shape(rho)[0]

    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return np.array(centers)


# Select the point where the product of rho and delta is larger as Clustering Center
def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)

    # Label several clusters
    for i, center in enumerate(centers):
        labs[center] = i

    # Sort the density from high to low
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # Label from the points with high density
        if labs[index] == -1:
            # If it has not been marked
            # Then the cluster label is the closest to it and has a higher density than it
            # The labels of the dots in are the same
            labs[index] = labs[int(nearest_neiber[index])]
    return labs


def draw_decision(rho, deltas, name="0_decision.jpg"):
    plt.cla()
    for i in range(np.shape(dataA)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))  # 标注文字
        plt.xlabel("rho")
        plt.ylabel("deltas")
    plt.savefig(name)


def draw_cluster(datas, labs, centers, dic_colors, name="0_cluster.jpg"):
    plt.cla()
    K = np.shape(centers)[0]

    for k in range(K):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        # Draw the data points
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k % 7])
        # Draw the cluster center
        plt.scatter(datas[centers[k], 0], datas[centers[k], 1], color="k", marker="+", s=200.)
    plt.savefig(name)


if __name__ == "__main__":
    logger.info("main start")

    # Initialize the data
    inputA_path = "./data/HDLS.csv"
    dataA = init_data(inputA_path)

    inputB_path = "./data/HDLX.csv"
    dataB = init_data(inputB_path)

    logger.info("computer start")
    # Calculate the distance matrix
    distance = setup_datas(dataA, dataB)
    # Calculate thedc
    dc = select_dc(distance)
    print("dc", dc)
    # Calculate the local density
    rho = get_density(distance, dc, method="Gaussion")
    # Calculate thegamma
    # gamma =
    # Calculate the density distance
    deltas, nearest_neiber = get_deltas(distance, rho)

    # Obtain the cluster center points
    # centers = find_centers_K(rho, deltas,3)
    centers = find_centers_auto(rho, deltas)
    print("centers", centers)
    centers_writer = FileUtil("result_centers")
    centers_writer.write(centers)

    # Non-clustering center classification
    cluster = cluster_PD(rho, centers, nearest_neiber) #Perform clustering
    print('Perform clustering', cluster)
    cluster_writer = FileUtil("result_cluster")
    cluster_writer.write(cluster)

    logger.info("computer end")

    logger.info("draw start")

    # 展示gamma图
    logger.info("draw gamma")
    gamma = show_decision_chart(rho, deltas, centers)

    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0)}
    file_name = "result/dpc"
    # Draw the density/distance distribution map
    # logger.debug("draw decision")
    # draw_decision(rho, deltas, name=file_name + "_decision.jpg")
    # labs = cluster_PD(rho, centers, nearest_neiber)
    # logger.debug("draw cluster_A")
    # draw_cluster(dataA, labs, centers, dic_colors, name=file_name + "_cluster_A.jpg")
    # logger.debug("draw cluster_B")
    # draw_cluster(dataB, labs, centers, dic_colors, name=file_name + "_cluster_B.jpg")

    # labs = cluster_PD(rho, centers, nearest_neiber)
    # draw_cluster(dataA, labs, centers, dic_colors, name=file_name + "_cluster_A.jpg")
    # draw_cluster(dataB, labs, centers, dic_colors, name=file_name + "_cluster_B.jpg")

    # datatype = gdal.GDT_Float64
    # driver = gdal.GetDriverByName("GTiff")
    # # tif image storage location
    # dataset = driver.Create(r'result\dict.tif', distance.shape[1], distance.shape[0], 1, datatype)
    # dataset.GetRasterBand(1).WriteArray(distance)
    print(dc)
    logger.info("draw end")






























