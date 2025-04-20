import numpy as np
import matplotlib.pyplot as plt
import logging
import csv
from FileUtil import FileUtil

from osgeo import gdal

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 初始化输入数据
def init_data(path):
    logger.info("init data start")
    # 从文件读取数据
    inputData = []
    with open(path, encoding="utf-8") as fin:
        reader = csv.reader(fin)
        # 跳过首行
        next(reader)
        for line in reader:
            inputData.append(line)

    # 将 python 数组转换为 np 矩阵
    datas = np.array(inputData).astype(np.float32)
    logger.info("init data end")
    return datas

# 初始化矩阵
def setup_datas(data_A, data_B):
    logger.info("setup datas start")
    # 读取矩阵行数和列数
    N, D_A = np.shape(dataA)
    N, D_B = np.shape(dataB)

    distance = np.zeros((N, N))
    file_util = FileUtil("result_distance")
    for i in range(N):
        row_result = []
        logger.info("setup iter datas %d", i)
        for j in range(N):
            # 离散数据
            first_A_row = data_A[i]
            second_A_row = data_A[j]
            # 简单匹配距离

            result_A = first_A_row == second_A_row
            max_A = 1-np.min(result_A)

            # sameNum = np.sum(first_A_row == second_A_row)
            # value_A = 1-float(sameNum) / float(D_A)

            # 连续数据
            first_B_row = data_B[i]
            second_B_row = data_B[j]
            # 高尔距离
            differ = first_B_row - second_B_row
            abs_differ = np.absolute(differ)

            max_B = np.max(abs_differ)/100

            # differ_sum = np.sum(abs_differ)
            # value_B =float(differ_sum) / (100.0 * float(D_B))
            value = max(max_A, max_B)
            # value = (value_A + value_B) * 0.5
            row_result.append(value)

        distance[i] = np.array(row_result).astype(np.float32)
        # 将距离矩阵写入文件
        file_util.write(row_result)
    # 处理对角线
    # m = np.median(S)
    # np.fill_diagonal(S, m)
    # logger.info("setup datas end")
    return distance


# 找到密度计算的阈值dc
# 要求平均每个点周围距离小于dc的点的数目占总点数的1%-2%
def select_dc(distance):
    '''算法1'''
    N = np.shape(distance)[0]
    tt = np.reshape(distance, N * N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]

    ''' 算法 2 '''
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

# 计算每个点的局部密度pi
def get_density(distance, dc, method=None):
    N = np.shape(distance)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(distance[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(distance[i, :] / dc) ** 2)) - 1
    return rho


# 计算每个数据点的密度距离
# 即对每个点，找到密度比它大的所有点
# 再在这些点中找到距离其最近的点的距离
def get_deltas(distance, rho):
    N = np.shape(distance)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(distance[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(distance[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber

# 确定类别点，计算gamma值并画图
def show_decision_chart(rho, deltas, centers):
    # 注意进行归一化
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
    # 将gamma写入文件
    for row_gamma in gamma:
        file_util.write([row_gamma], show_log=False)
    return gamma

# 通过阈值选取 rho与delta都大的点
# 作为聚类中心
def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) / 2
    delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
    N = np.shape(rho)[0]

    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return np.array(centers)


# 选取 rho与delta乘积较大的点作为
# 聚类中心
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

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        labs[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labs[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
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
        # 画数据点
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k % 7])
        # 画聚类中心
        plt.scatter(datas[centers[k], 0], datas[centers[k], 1], color="k", marker="+", s=200.)
    plt.savefig(name)


if __name__ == "__main__":
    logger.info("main start")

    # 初始化数据
    inputA_path = "./data/HDLS.csv"
    dataA = init_data(inputA_path)

    inputB_path = "./data/HDLX.csv"
    dataB = init_data(inputB_path)

    logger.info("computer start")
    # 计算距离矩阵
    distance = setup_datas(dataA, dataB)
    # 计算dc
    dc = select_dc(distance)
    print("dc", dc)
    # 计算局部密度
    rho = get_density(distance, dc, method="Gaussion")
    # # 计算gamma
    # gamma =
    # 计算密度距离
    deltas, nearest_neiber = get_deltas(distance, rho)

    # 获取聚类中心点
    # centers = find_centers_K(rho, deltas,3)
    centers = find_centers_auto(rho, deltas)
    print("centers", centers)
    centers_writer = FileUtil("result_centers")
    centers_writer.write(centers)

    # 非聚类中心分类
    cluster = cluster_PD(rho, centers, nearest_neiber) #进行聚类
    print('进行聚类', cluster)
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
    # 绘制密度/距离分布图
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
    # # tif影像存放位置
    # dataset = driver.Create(r'result\dict.tif', distance.shape[1], distance.shape[0], 1, datatype)
    # dataset.GetRasterBand(1).WriteArray(distance)
    print(dc)
    logger.info("draw end")






























