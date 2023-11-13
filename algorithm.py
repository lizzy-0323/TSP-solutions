# coding: utf-8
"""
@author: laziyu
@date: 2023/11/11
using genetic algorithm and ant colony algorithm to solve TSP problem
"""
import numpy as np


class GeneticAlgorithm(object):
    """
    genetic algorithm
    """

    def __init__(
        self,
        population_size=500,
        evolution_size=300,
        cross_size=150,
        mutation_size=50,
        max_generation=1000,
        city_count=100,
        dist_mtx=None,
    ) -> None:
        """
        Initialize the genetic algorithm.
        :param population_size: Population size
        :param evolution_size: Number of individuals evolving in each generation
        :param cross_size: Number of crossovers
        :param mutation_size: Number of mutations
        :param max_generation: Maximum number of generations
        :param city_count: Number of cities
        :param dist_mtx: Distance matrix
        """
        self.gene_length = city_count
        self.population_size = population_size  # 种群大小
        self.cross_size = cross_size  # 交叉数
        self.mutation_size = mutation_size  # 变异数
        self.evolution_size = evolution_size  # 每一代进化的个数
        self.max_generation = max_generation  # 最大进化代数
        self.generations = 0  # 目前的代数
        self.elites = None
        self.best_gene = None  # 最优解
        self.best_length = np.zeros(max_generation)  # 每一代最优路径的长度
        self.dist_mtx = dist_mtx
        self.population = self.init_population()  # 种群
        self.evaluate()  # 对当前种群进行评估

    def init_population(self):
        """
        Initialize the population.
        :return: Initialized population
        """
        # 随机生成
        population = []
        for _ in range(self.population_size):
            gene = np.arange(self.gene_length)
            np.random.shuffle(gene)
            # 确保没有重复的种群
            population.append(Life(gene))
        return population

    def distance(self, path):
        """
        Calculate the distance of a path.
        :param path: Path
        :return: Distance of the path
        """
        dist = 0
        for i in range(self.gene_length - 1):
            dist += self.dist_mtx[path[i]][path[i + 1]]
        dist += self.dist_mtx[path[-1]][path[0]]
        return dist

    def evaluate(self):
        """
        Evaluate the fitness of the population and select elites.
        """
        for p in self.population:
            p.score = 1 / self.distance(p.gene) * 10000
        # 获得最优解
        self.elites = sorted(self.population, key=lambda x: x.score, reverse=True)
        if self.best_gene is None:
            self.best_gene = self.elites[0].gene
        # 如果当前最优解更优，则更新最优解
        if self.distance(self.elites[0].gene) < self.distance(self.best_gene):
            self.best_gene = self.elites[0].gene
            print("update best gene")

    def crossover(self, parent1, parent2):
        """
        Use partially mapped crossover.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :return: Offspring after crossover
        """
        # 确定交叉的起始位置
        start = np.random.randint(0, self.gene_length - 1)
        child1, child2 = np.concatenate(
            (parent2[:start], parent2[start:][::-1])
        ), np.concatenate((parent1[start:], parent1[:start][::-1]))
        return child1, child2

    def mutation(self, gene):
        """
        Randomly select two positions and perform swap mutation.
        :param gene: Gene before mutation
        :return: Gene after mutation
        """
        i1, i2 = np.random.randint(0, self.gene_length - 1), np.random.randint(
            0, self.gene_length - 1
        )
        gene[i1], gene[i2] = gene[i2], gene[i1]
        return gene

    def selection(self, candidate):
        """
        Select an individual based on fitness using roulette wheel selection.
        :return: Selected individual
        """
        adaptability_sum = sum((p.score for p in candidate))
        # 适应度比例
        adaptability_ratio = [p.score / adaptability_sum for p in candidate]
        # 按照适应度比例，采用轮盘赌法选择
        p = np.random.choice(candidate, p=adaptability_ratio)
        return p

    def generation(self):
        """
        Generate a new generation using selection, crossover, and mutation.
        """
        next_generation = []
        # 选择
        for _ in range(self.evolution_size):
            p1 = self.selection(self.population)
            next_generation.append(p1)
        for _ in range(self.cross_size // 2):
            # 交叉
            p1 = self.selection(next_generation)
            p2 = self.selection(next_generation)
            gene1, gene2 = self.crossover(p1.gene, p2.gene)
            next_generation.append(Life(gene1))
            next_generation.append(Life(gene2))
        for _ in range(self.mutation_size):
            # 变异
            p1 = self.selection(next_generation)
            gene = self.mutation(p1.gene)
            next_generation.append(Life(gene))
        self.population = next_generation
        self.generations += 1
        # 评估新一代种群的适应度
        self.evaluate()
        self.best_length[self.generations - 1] = self.distance(self.best_gene)

    def is_finished(self) -> bool:
        """
        Check if the algorithm has reached the maximum number of generations.
        :return: True or False
        """
        return self.generations >= self.max_generation

    def run(self) -> None:
        """run the algorithm"""
        while not self.is_finished():
            self.generation()
            print(f"generation: {self.generations}")
            # print(f"best gene: {self.best_gene}")
            print(f"distance:, {self.distance(self.best_gene)}")
        print(f"best gene: {self.best_gene}")


class Life(object):
    """
    individual
    :param gene: gene
    """

    def __init__(self, gene) -> None:
        self.gene = gene
        self.score = 0


class AntColonyAlgorithm(object):
    """
    Ant colony algorithm.
    :param dist_mtx: Distance matrix
    :param path_count: Length of ant paths
    :param ant_count: Number of ants
    :param city_count: Number of cities
    :param max_iteration: Maximum number of iterations
    :param info_rho: Pheromone evaporation rate
    :param alpha: Importance of pheromones
    :param beta: Importance of heuristic factor
    :param Q: Intensity of pheromone increase
    """

    def __init__(
        self,
        dist_mtx=None,
        path_count=100,
        ant_count=100,
        city_count=100,
        max_iteration=1000,
        info_rho=0.5,
        alpha=1,
        beta=1,
        Q=1,
    ) -> None:
        self.dist_mtx = dist_mtx
        self.phenomenon_mtx = np.ones_like(dist_mtx)  # 信息素矩阵
        self.ant_mtx = np.zeros((ant_count, path_count)).astype(int)  # 记录蚂蚁的路径
        self.etab_mtx = 1 / (dist_mtx + 1e-10)  # 启发式因子矩阵
        self.best_path = np.zeros(path_count).astype(int)  # 每一代的最优路径
        self.best_length = np.zeros(max_iteration)  # 每一代最优路径的长度
        self.max_iteration = max_iteration  # 最大迭代次数
        self.info_rho = info_rho  # 信息素挥发率
        self.alpha = alpha  # 信息素重要程度
        self.Q = Q  # 信息素增加强度
        self.beta = beta  # 启发式因子重要程度
        self.city_count = city_count

    def distance(self, path):
        """
        Calculate the distance of a path.
        :param path: Path
        :return: Distance of the path
        """
        dist = 0
        for i in range(self.city_count - 1):
            dist += self.dist_mtx[path[i]][path[i + 1]]
        dist += self.dist_mtx[path[-1]][path[0]]
        return dist

    def update_phenomenon_mtx(self):
        """
        Update the pheromone matrix.
        """
        ant_count = self.ant_mtx.shape[0]
        change_phenomenon_mtx = np.zeros_like(self.phenomenon_mtx)
        for i in range(ant_count):
            for j in range(self.city_count - 1):
                # 计算信息素增量
                change_phenomenon_mtx[self.ant_mtx[i][j]][self.ant_mtx[i][j + 1]] += (
                    self.Q / self.dist_mtx[self.ant_mtx[i][j]][self.ant_mtx[i][j + 1]]
                )
            change_phenomenon_mtx[self.ant_mtx[i][j + 1]][self.ant_mtx[i][0]] += (
                self.Q / self.dist_mtx[self.ant_mtx[i][j + 1]][self.ant_mtx[i][0]]
            )
        # 更新信息素矩阵
        self.phenomenon_mtx = (
            1 - self.info_rho
        ) * self.phenomenon_mtx + change_phenomenon_mtx

    def selection(
        self,
        unvisited: set,
        visiting: int,
    ):
        """
        Use roulette wheel selection to choose the next city.
        :param unvisited: Unvisited cities
        :param visiting: Current visiting city
        :return: Next visiting city
        """
        list_unvisited = list(unvisited)
        # 计算概率
        prob = np.zeros(len(list_unvisited))
        for i, _ in enumerate(list_unvisited):
            prob[i] = np.power(self.phenomenon_mtx[visiting][i], self.alpha) * np.power(
                self.etab_mtx[visiting][i], self.beta
            )
        prob /= np.sum(prob)
        # 选择下一个城市
        next_city = np.random.choice(list_unvisited, p=prob)
        return next_city

    def run(self) -> None:
        """run the algorithm"""
        count = 0
        ant_count = self.ant_mtx.shape[0]
        # 开始迭代
        while count < self.max_iteration:
            length = np.zeros(ant_count)  # 记录每只蚂蚁的路径长度
            # 随机生成蚂蚁的起始位置
            for i in range(ant_count):
                visiting = self.ant_mtx[i][0] = np.random.randint(0, self.city_count)
                unvisited = set(range(self.city_count))
                unvisited.remove(visiting)  # 删除已经访问过的城市
                for j in range(1, self.city_count):
                    # 选择下一个城市
                    next_city = self.ant_mtx[i][j] = self.selection(unvisited, visiting)
                    unvisited.remove(next_city)
                    length[i] += self.dist_mtx[visiting][next_city]
                    visiting = next_city
            # 更新最优路径
            if count == 0:
                # 更新最优路径的长度
                self.best_length[count] = np.min(length)
                self.best_path = self.ant_mtx[np.argmin(length)]
            else:
                # 如果当前路径更优，则更新最优路径
                if np.min(length) < self.best_length[count - 1]:
                    self.best_length[count] = np.min(length)
                    self.best_path = self.ant_mtx[np.argmin(length)]
                else:
                    self.best_length[count] = self.best_length[count - 1]
            # 更新信息素矩阵
            self.update_phenomenon_mtx()
            print(f"min path length: {self.best_length[count]}")
            print(f"iteration: {count+1}")
            count += 1
        print(f"best path: {self.best_path}")
