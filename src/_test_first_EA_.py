import random
import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd


# generation function
# param: gen_num means the number of solution
def generate(gen_num):
    population_list = []
    for i in range(gen_num):
        individual = []
        for j in range(15):
            individual.append(random.randint(0, 1))
        # ind[0] = individual , ind[1] = fitness
        ind_with_fit = [individual, fitness(individual)]
        population_list.append(ind_with_fit)

    return population_list


def fitness(individual):
    global fitness_num
    fitness_num += 1
    return sum(individual)


# tour_size can not larger than population_length
def tournament(tour_size, population_list):
    tour_list: list = []
    for i in range(tour_size):
        rd_num = random.randint(0, len(population_list) - 1)
        tour_list.append(population_list[rd_num])

    parent = max(tour_list, key=lambda individual: individual[1]).copy()

    return parent


def crossover(i1, i2, pro=1):
    rd_pro = np.random.rand()
    if (rd_pro < pro):
        check_point = random.randint(0, len(i1[0]) - 1)
        new_i1 = i1[0][0:check_point] + i2[0][check_point:]
        new_i2 = i2[0][0:check_point] + i1[0][check_point:]
        new_i1_with_fit = [new_i1, fitness(new_i1)]
        new_i2_with_fit = [new_i2, fitness(new_i2)]
        return [new_i1_with_fit, new_i2_with_fit]
    else:
        return [i1, i2]


def mutate(individual, size=1, pro=1):
    new_individual = individual[0].copy()
    for i in range(size):
        rd_pro = np.random.rand()
        if (rd_pro < pro):
            rd_mut_point = random.randint(0, len(new_individual) - 1)
            new_individual[rd_mut_point] = 1 - new_individual[rd_mut_point]
        else:
            continue
    new_ind_with_fit = [new_individual, fitness(new_individual)]
    return new_ind_with_fit


def replacement(population_list, individual):
    population_list.sort(key=lambda ind: ind[1])
    # replace the worst individual
    if individual[1] > population_list[0][1]:
        population_list[0] = individual


# to find out the max fitness in the population
def get_population_max_fitness(population_list):
    return max(individual[1] for individual in population_list)


# tournament_size should be even
def first_EA(pop_size,t_size,m_rate,m_size,cross_rate):
    global fitness_num

    sum_record_iter = 0
    record_iter_list = []
    success_time = 0
    # simulate 100 time ,calculate the probability of success
    for i in range(100):
        fitness_num = 0
        record_iter = 0
        pop_list = generate(pop_size)
        while True:
            # generation

            # tournament -- pick parents
            parent_list = []
            for i in range(t_size):
                parent_list.append(tournament(t_size, pop_list))

            # crossover
            child_list = []
            for i in range(0, len(parent_list), 2):
                child_list.extend(crossover(parent_list[i], parent_list[i + 1],cross_rate))

            # mutation
            mu_child_list = []
            for child in child_list:
                mu_child_list.append(mutate(child, size=m_size,pro=m_rate))

            # replacement
            for mu_child in mu_child_list:
                replacement(pop_list, mu_child)

            # get_population_max_fitness
            max_population_fitness = get_population_max_fitness(pop_list)

            if max_population_fitness == 15 and record_iter < 1000 and fitness_num < 100000:
                record_iter_list.append(record_iter)
                sum_record_iter += record_iter
                success_time += 1
                break
            elif record_iter > 1000 or fitness_num > 100000:
                break
            else:
                record_iter += 1





    np_record_iter_list = np.array(record_iter_list)

    # print(f"mean={sum(record_iter_list) / 100},success_pro={success_time / 100},std={np.std(np_record_iter_list)}")
    # print(fitness_num)


    success_rate = success_time / 100
    mean = sum_record_iter / success_time
    if success_rate > 0.95:
        # return np_record_iter_list
        return mean
    else :
        return -1


class second_EA:
    def __init__(self):
        self.rank = []

    def generate_population(self, size):
        # chromosome will be [pop_size,t_size,m_rate,m_size,cross_rate
        # pop_size between 5 and 20
        # t_size between 2 and 10
        # m_rate between 0 and 1
        # m_size between 1 and 5
        # crossover rate between 0 and 1
        pop_with_fit = []
        population_list = [
            [np.random.randint(10, 50), 2 * np.random.randint(1, 5), np.random.rand(), np.random.randint(1, 5),
             np.random.rand()] for i in range(size)]
        for individual in population_list:
            individual_with_fitness = [individual, self.fitness(individual)]
            pop_with_fit.append(individual_with_fitness)
        return pop_with_fit

    # def clean_meanless(self,pop_list):
    #     is_pop_meanless = True
    #     while is_pop_meanless:
    #         have_ind_meanless = True
    #         for individual in pop_list:
    #
    #             if individual[1] == -1:
    #                 have_ind_meanless = False
    #                 new_individual = [np.random.randint(10, 50), 2 * np.random.randint(1, 5), np.random.rand(), np.random.randint(1, 5),
    #                  np.random.rand()]
    #                 individual = [new_individual,self.fitness(new_individual)]
    #         if have_ind_meanless == True:
    #             break

    def fitness(self, individual):
        # fitness of an individual is 1000-mean_num_iterations.

        # print("evaluating individual, ", individual)
        mean = first_EA(individual[0], individual[1], individual[2], individual[3], individual[4])
        # print("iterations generated")
        # Taking absolute value so we don't end up with negatives ranking higher than the rest
        # mean = abs(1000 - np.mean(iterations))
        # std = np.std(iterations)
        return mean

    def tournament(self, population_list, tour_size=2):
        tour_list: list = []
        for i in range(tour_size):
            rd_num = random.randint(0, len(population_list) - 1)
            tour_list.append(population_list[rd_num])

        parent = max(tour_list, key=lambda individual: individual[1]).copy()

        return parent

    def tournament_parent_size(self, parent_size, population_list):
        parents = []
        for i in range(parent_size):
            parents.append(self.tournament(population_list))

        return parents

    def crossover(self, parents, pro=1):
        children = []
        for parent_index in range(0, len(parents), 2):
            rd_pro = np.random.rand()
            if (rd_pro < pro):
                check_point = random.randint(0, len(parents[parent_index]) - 1)
                new_child1 = parents[parent_index][0:check_point] + parents[parent_index + 1][check_point:]
                new_child2 = parents[parent_index + 1][0:check_point] + parents[parent_index][check_point:]
                children.append(new_child1)
                children.append(new_child2)
            else:
                children.append(parents[parent_index], parents[parent_index + 1])

        return children

    def mutation(self, individual, size=1, pro=1):
        new_individual = individual[0].copy()
        # some parameter have already been mutate
        exit_mutate_para_list = []
        rd_num_list = random.sample(range(0, 4), size)
        for rd_num in rd_num_list:
            rd_pro = np.random.rand()
            if (rd_pro < pro):
                # population_size change
                if rd_num == 0:
                    change_factor = np.random.randint(10, 50)
                    # make sure the change_factor is not equal to old population_size
                    while change_factor == new_individual[0]:
                        change_factor = np.random.randint(10, 50)
                    new_individual[rd_num] = change_factor

                # t_size change
                elif rd_num == 1:
                    change_factor = 2 * np.random.randint(1, 5)
                    # make sure the change_factor is not equal to old population_size
                    while change_factor == new_individual[0]:
                        change_factor = 2 * np.random.randint(1, 5)
                    new_individual[rd_num] = change_factor

                # m_rate change
                elif rd_num == 2:
                    change_factor = np.random.rand()
                    # make sure the change_factor is not equal to old population_size
                    while change_factor == new_individual[0]:
                        change_factor = np.random.rand()
                    new_individual[rd_num] = change_factor
                # m_size
                elif rd_num == 3:
                    change_factor = np.random.randint(1, 5)
                    # make sure the change_factor is not equal to old population_size
                    while change_factor == new_individual[0]:
                        change_factor = np.random.randint(1, 5)
                    new_individual[rd_num] = change_factor
                # cross_rate
                else:
                    change_factor = np.random.rand()
                    # make sure the change_factor is not equal to old population_size
                    while change_factor == new_individual[0]:
                        change_factor = np.random.rand()
                    new_individual[rd_num] = change_factor

        new_individual_with_fit = [new_individual, self.fitness(new_individual)]
        return new_individual_with_fit

    def replacement(self, population_list, individual):
        population_list.sort(key=lambda individual: individual[1])
        # replace the worst individual

        if individual[1] > population_list[0][1]:
            population_list[0] = individual

    def get_population_max_fitness(self, population_list):
        population_list.sort(reverse=True, key=lambda individual: individual[1])
        # print("population_list")
        # print(population_list)
        return population_list[0]
        # return max(individual[1] for individual in population_list)


if __name__ == '__main__':
    # chromosome will be [pop_size,t_size,m_rate,m_size,cross_rate
    # pop_size between 5 and 20
    # t_size between 2 and 10
    # m_rate between 0 and 1
    # m_size between 0 and 15
    # cross_rate between 0 and 1
    iter_list = []
    para_com_list = []
    pop_size_list = []
    t_size_list = []
    m_rate_list = []
    m_size_list = []
    cross_rate_list = []
    mean_value_list = []
    iter_num = 0
    for pop_size in range(10, 20, 2):
        print(pop_size)
        for t_size in range(2, 10, 2):
            print(t_size)
            for m_rate in np.arange(0.2, 1, 0.2):
                print(m_rate)
                for m_size in range(7,15):
                    print(m_size)
                    for cross_rate in np.arange(0.2,1,0.2):
                        print(cross_rate)
                        mean_value = first_EA(pop_size, t_size, m_rate, m_size, cross_rate)
                        print("---------------")
                        print(mean_value)
                        iter_list.append(iter_num)
                        pop_size_list.append(pop_size)
                        t_size_list.append(t_size)
                        m_rate_list.append(m_rate)
                        m_size_list.append(m_size)
                        cross_rate_list.append(cross_rate)
                        mean_value_list.append(mean_value)
                        print(iter_num)
    output_data = {
        "iter": iter_list,
        'population_size': pop_size_list,
        'tournament_size': t_size_list,
        'mutation_rate': m_rate_list,
        'mutation_size': m_size_list,
        'crossover_rate': cross_rate_list,
        'mean_value': mean_value_list,
    }
    output_csv = pd.DataFrame(data=output_data)
    output_csv.to_excel('../data/output-1ea_differentparameter.xls')


