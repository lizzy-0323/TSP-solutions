# coding: utf-8
"""
@author: laziyu
@date: 2023/11/12
using genetic algorithm and ant colony algorithm to solve TSP problem
"""
import time
import numpy as np
from algorithm import GeneticAlgorithm, AntColonyAlgorithm
import matplotlib.pyplot as plt


class TSP(object):
    """
    TSP
    """

    def __init__(self) -> None:
        self.city = self.read_file()
        self.dist_mtx = self.init_dist()

    def read_file(self, filename="a280.tsp"):
        """
        read tsp format file
        :param filename: tsp format file
        """
        # tsp格式数据
        city = []
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines[:100]:
                location = line.strip().split()
                city.append([int(location[1]), int(location[2])])
        return city

    def init_dist(self):
        """
        init distance matrix
        :return: distance matrix
        """
        n = len(self.city)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i][j] = dist[j][i] = np.sqrt(
                    (self.city[i][0] - self.city[j][0]) ** 2
                    + (self.city[i][1] - self.city[j][1]) ** 2
                )
        return dist

    def solve(self, method="GA"):
        """
        select method to solve the problem
        :param method: GA or AC
        """
        start_time = time.time()
        if method == "AC":
            ac = AntColonyAlgorithm(dist_mtx=self.dist_mtx, max_iteration=10)
            ac.run()
            end_time = time.time()
            print("AntColony Algorithm run time: ", end_time - start_time)
            self.plot(ac.best_length, "AC")
        elif method == "GA":
            ga = GeneticAlgorithm(dist_mtx=self.dist_mtx, max_generation=10)
            ga.run()
            end_time = time.time()
            print("GeneticAlgorithm run time", end_time - start_time)
            self.plot(ga.best_length, "GA")
        else:
            assert False, "don not support this method"

    def plot(self, length, method="GA"):
        """
        plot the path
        :param path: the best path
        """
        plt.figure()
        plt.plot(length)
        plt.xlabel("generation")
        plt.ylabel("distance")
        plt.title(f"{method} algorithm")
        # plt.show()
        plt.savefig(f"{method}.png")


if __name__ == "__main__":
    tsp = TSP()
    tsp.solve("GA")
    tsp.solve("AC")
