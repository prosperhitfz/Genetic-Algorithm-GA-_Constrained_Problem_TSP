import numpy as np
import matplotlib.pyplot as plt
import random

city_numbers = 22


def encoding_random_initialize(city_number, random_times):  # 编码生成染色体（即将city_numbers个数的城市打乱顺序）
    order = []
    for x in range(city_number):
        order.append(x)  # 写入起始顺序
    final_order = []
    for loop in range(random_times):
        np.random.shuffle(order)
        final_order.append(order.copy())  # 随机打乱后得到新顺序（合法个体或说染色体）
    return final_order


def fitness_function_city_distance(city_number, location):  # 计算不同城市之间的距离，为确定适应度函数做准备
    distance = np.zeros((city_number, city_number))  # 初始化二维距离矩阵distance
    for loop in range(city_number):
        for loop1 in range(city_number):
            if loop == loop1:
                distance[loop][loop1] = np.inf
                # 距离矩阵对角线值设置为正无穷（不可以自己走到自己，没意义）
            line = location[loop]  # distance矩阵横行
            row = location[loop1]  # distance矩阵纵列
            mid = []
            for loop2 in zip(line, row):
                mid.append((loop2[0] - loop2[1]) ** 2)
            distance[loop][loop1] = np.sqrt(sum(mid))
            # 距离计算函数（欧式距离）
    return distance


def total_path_length(path, distance):  # 计算每个个体（染色体）所选择的路径总长度
    a = path[0]  # 出发城市
    b = path[-1]  # 到达城市
    length = distance[a][b]
    for loop in range(len(path) - 1):
        a = path[loop]
        b = path[loop + 1]
        length += distance[a][b]
    return length


def choose(individual_fitness_score, individual_choose_from_candidate_parents):  # 选择操作（轮盘赌选择法）
    global para1, para2  # 定义全局变量
    sum_score = sum(individual_fitness_score)
    score_percent = []
    for minus in individual_fitness_score:
        score_percent.append(minus * 1.0 / sum_score)  # 初始化得分占比数组
    randn1 = np.random.rand()
    randn2 = np.random.rand()
    # 产生两个随机数
    for number, minus in enumerate(score_percent):
        if randn1 >= 0:
            randn1 -= minus
            if randn1 < 0:
                para1 = number
        if randn2 >= 0:
            randn2 -= minus
            if randn2 < 0:
                para2 = number
        if randn1 < 0 and randn2 < 0:
            break
    # 轮盘赌过程
    return list(individual_choose_from_candidate_parents[para1]), \
           list(individual_choose_from_candidate_parents[para2])
    # 选出两个体用于交叉变异后将更高适应度个体添加进群体，
    # 其中适应度越高的个体（染色体）被选中的概率越大


def cross(individual1, individual2):  # 交叉操作
    length = len(individual1)
    path_list = []
    for loop in range(length):
        path_list.append(loop)
    order = list(random.sample(path_list, 2))
    order.sort()
    start, end = order
    # 随机选取两个位置，然后小的作为交叉操作起点；大的作为终点

    tmp = individual1[start:end]
    gene1_conflict_index = []
    for sub in tmp:
        sub_index = individual2.index(sub)
        if not (start <= sub_index < end):  # python可以简化，C C++不可以！
            gene1_conflict_index.append(sub_index)
    gene2_conflict_index = []
    tmp = individual2[start:end]
    for sub in tmp:
        sub_index = individual1.index(sub)
        if not (start <= sub_index < end):  # python可以简化，C C++不可以！
            gene2_conflict_index.append(sub_index)
    # 找到基因（存储路径）冲突点并存下他们的下标,
    # gene1_conflict_index中存储的是gene2中的下标,
    # gene2_conflict_index中存储gene1与它冲突的下标

    tmp = individual1[start:end].copy()
    individual1[start:end] = individual2[start:end]
    individual2[start:end] = tmp
    # 交叉操作
    for sub_index in range(len(gene1_conflict_index)):
        i = gene1_conflict_index[sub_index]
        j = gene2_conflict_index[sub_index]
        individual2[i], individual1[j] = individual1[j], individual2[i]
    # 解决冲突
    return list(individual1), list(individual2)


def mutate(individual):  # 变异操作
    gene_list = []
    for loop in range(len(individual)):
        gene_list.append(loop)
        # 获取所有染色体编码序列的序号
    order = list(random.sample(gene_list, 2))  # 随机取两个点（序号）
    start, end = min(order), max(order)  # 保存起点、终点
    convert = individual[start:end]
    convert = convert[::-1]  # 倒序函数
    individual[start: end] = convert  # 完成逆转变异操作（将随机选取的子段倒序）
    return list(individual)


def show_path_result(res, order):  # 展示结果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 解决中文显示问题
    plt.suptitle('针对ulysses22.tsp问题的遗传算法计算结果展示')
    if order == 1:
        plt.title('原始数据展示（各城市坐标位置）')
        plt.scatter(res[:, 0], res[:, 1])  # 按城市坐标参数数据（x，y）展示原始数
        for loop in range(city_numbers):
            plt.text(res[loop, 0], res[loop, 1], loop + 1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    if order == 2:
        plt.scatter(res[:, 0], res[:, 1])  # 标记出城市
        for loop in range(city_numbers):
            plt.text(res[loop, 0], res[loop, 1], loop + 1)
        res = np.vstack([res, res[0]])  # 在结果数组末尾加上起点信息以回到起点
        plt.plot(res[:, 0], res[:, 1], color='red')
        plt.title('随机路线图')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    if order == 3:
        plt.scatter(res[:, 0], res[:, 1])  # 标记出城市
        for loop in range(city_numbers):
            plt.text(res[loop, 0], res[loop, 1], loop + 1)
        res = np.vstack([res, res[0]])  # 在结果数组末尾加上起点信息以回到起点
        plt.plot(res[:, 0], res[:, 1], color='red')
        plt.title('最优解路线图')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


class Genetic_Algorithm:
    def __init__(self, city_number, random_times, iteration, position):
        self.city_number = city_number  # 城市总数
        self.random_times = random_times  # 采取随机操作的次数（由我自己设定）
        self.iteration = iteration  # 迭代次数
        self.location = position  # 城市位置（坐标）
        self.city_distance = fitness_function_city_distance(city_number, position)
        # distance是存有各个城市之间的距离的二维numpy矩阵
        self.gene = encoding_random_initialize(city_number, random_times)
        # gene是存有随机后得到的所有个体（染色体）的list，即随机选择经过城市顺序得到的的所有可能个体
        fitness_score = self.fitness_function(self.gene)  # 根据适应度函数得到各个个体（染色体）适应度得分
        sort_gene_number = np.argsort(-fitness_score)
        # 按照个体适应度降序（从大到小）得到个体（染色体）适应度从优到差排序
        fitness_best = self.gene[sort_gene_number[0]]
        fitness_best = self.location[fitness_best]
        # 找到适应度最优个体（染色体）即最优路径并按路径中经过的城市顺序获取坐标
        show_path_result(fitness_best, 2)  # 显示随机初始化后得到的选择路径

        self.choose_percent = 0.3  # 选择操作中选择的适应度高的个体占全部个体的比率
        self.mutate_possible_percent = 0.1  # 发生变异操作的概率
        self.iteration_x = [0]
        self.iteration_y = [1. / fitness_score[sort_gene_number[0]]]
        # 初始化定义第0次迭代的结果（随机化一下的结果），用于存储之后经每次遗传算法选择后迭代的结果以及画出收敛图

    def fitness_function(self, genes):  # 确定适应度函数
        fit = []
        for gene in genes:
            length = total_path_length(gene, self.city_distance)
            # 路径长度，用total_path_length函数求取
            # 参数一为个体list，参数二为用fitness_function_city_distance函数求取的各个城市间距离的二维矩阵
            fit.append(1.0 / length)
            # 适应度计算公式为恰好走遍city_numbers个城市再回到出发城市的倒数
        fitness_score = np.array(fit)
        return fitness_score

    def candidate_parent(self, scores, choose_percent):  # 定义父代群体
        fitness_scores = np.argsort(-scores)  # 获取所有个体适应度降序排序序列
        fitness_scores = fitness_scores[0: int(choose_percent * len(fitness_scores))]
        # 全部个体按适应度降序排序后选择其中前choose_percent%的高适应度个体做父代群体
        candidate_parents = []
        candidate_parents_fitness_scores = []
        for fitness_score in fitness_scores:
            candidate_parents.append(self.gene[fitness_score])
            candidate_parents_fitness_scores.append(scores[fitness_score])
            # 获取父代群体中个体（染色体）的编码信息和适应度值
        return candidate_parents, candidate_parents_fitness_scores

    def ga(self):  # 定义遗传算法
        print('遗传算法处理中...')
        best_gene = 0
        best_gene_fitness_score = 0
        # 初始化最优个体及其适应度
        for i in range(1, self.iteration + 1):
            fitness_scores = self.fitness_function(self.gene)  # 获取全部个体适应度
            candidate_parents, candidate_parents_fitness_scores = self.candidate_parent(fitness_scores,
                                                                                        self.choose_percent)
            # 按适应度从高到低选择前choose_percent%个个体作为父代群体
            mid_best_gene = candidate_parents[0]
            mid_best_gene_fitness_score = candidate_parents_fitness_scores[0]
            # 初始父代群体中的最优个体
            colony = candidate_parents  # 最初代群体
            while len(colony) < self.random_times:  # 生成新的群体
                gene_x, gene_y = choose(candidate_parents_fitness_scores, candidate_parents)
                # 对父代个体（染色体）进行选择操作（轮盘赌选择）

                gene_x_new, gene_y_new = cross(gene_x, gene_y)
                # 交叉操作

                if np.random.rand() < self.mutate_possible_percent:
                    gene_x_new = mutate(gene_x_new)
                if np.random.rand() < self.mutate_possible_percent:
                    gene_y_new = mutate(gene_y_new)
                x_fitness_score = 1. / total_path_length(gene_x_new, self.city_distance)
                y_fitness_score = 1. / total_path_length(gene_y_new, self.city_distance)
                # 变异操作

                if x_fitness_score > y_fitness_score and (gene_x_new not in colony):
                    colony.append(gene_x_new)
                elif x_fitness_score <= y_fitness_score and (gene_y_new not in colony):
                    colony.append(gene_y_new)
                # 将适应度高的个体放入新群体中

            self.gene = colony  # 更新经遗传算法后得到的新群体
            self.iteration_x.append(i)
            self.iteration_y.append(1. / mid_best_gene_fitness_score)
            if mid_best_gene_fitness_score > best_gene_fitness_score:
                best_gene_fitness_score = mid_best_gene_fitness_score
                best_gene = mid_best_gene
            print('第', i, '次迭代选出的最优个体，其适应度为：', 1. / best_gene_fitness_score)
        print('\n最优个体适应度：', 1. / best_gene_fitness_score)
        result = np.array(self.location[best_gene])
        plt.suptitle('针对ulysses22.tsp问题的遗传算法计算结果展示')
        plt.title('遗传算法优化过程')
        plt.xlabel('迭代次数')
        plt.ylabel('最优值(即适应度)')
        plt.plot(self.iteration_x, self.iteration_y)
        plt.show()
        return result, 1. / best_gene_fitness_score


# .tsp文件数据读取预处理
lines = open('data/ulysses22.tsp', 'r').readlines()  # 按行打开读取数据
index = lines.index('NODE_COORD_SECTION\n')  # 检测字符串'NODE_COORD_SECTION'作为开始
pre_data = lines[index + 1:-1]  # 得到预处理数据
# 从上句'NODE_COORD_SECTION'位置的下一位开始读取city的序号和坐标location，一直读取到字符串'EOF'之前为止
final_data = []  # 设置临时存放data的数组
for line in pre_data:
    line = line.strip().split(' ')
    # 去除空开头结尾的0（空白），将每一行的字符串按' '的位置转换成float型数据
    line1 = []
    for info in line:
        line1.append(float(info))  # 将每行中单独的数据存储进行
    final_data.append(line1)  # 将每行的数据存储进临时data数组
final_data = np.array(final_data)  # 转换成numpy可处理数组类型
final_data = final_data[:, 1:]
show_path_result(final_data, 1)

best_path, path_len = Genetic_Algorithm(city_number=final_data.shape[0],
                                        random_times=50,
                                        iteration=30,
                                        position=final_data.copy()).ga()
show_path_result(best_path, 3)
