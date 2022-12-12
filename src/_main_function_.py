import random
import numpy as np
from collections import Counter



# generation function
# param: gen_num means the number of solution
def generate(gen_num):
    population_list = []
    for i in range(gen_num):
        individual = []
        for j in range(15):
            individual.append(random.randint(0, 1))
        # ind[0] = individual , ind[1] = fitness
        ind_with_fit= [individual,fitness(individual)]
        population_list.append(ind_with_fit)

    return population_list

def fitness(individual):
    global fitness_num
    fitness_num+=1
    return sum(individual)

# tour_size can not larger than population_length
def tournament(tour_size, population_list):
    tour_list :list = []
    for i in range(tour_size):
        rd_num = random.randint(0, len(population_list)-1)
        tour_list.append(population_list[rd_num])

    parent = max(tour_list, key=lambda individual: individual[1]).copy()
    return parent

def crossover(i1,i2,pro=1):
    rd_pro = np.random.rand()
    if (rd_pro < pro):
        check_point = random.randint(0,len(i1[0])-1)
        new_i1 = i1[0][0:check_point] + i2[0][check_point:]
        new_i2 = i2[0][0:check_point] + i1[0][check_point:]
        new_i1_with_fit = [new_i1, fitness(new_i1)]
        new_i2_with_fit = [new_i2, fitness(new_i2)]
        return [new_i1_with_fit, new_i2_with_fit]
    else:
        return [i1,i2]


def mutate(individual,size=1,pro=1):
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

def replacement(population_list,individual):
    population_list.sort(key=lambda individual:individual[1])
    # replace the worst individual
    if individual[1] > population_list[0][1]:
        population_list[0] = individual

# to find out the max fitness in the population
def get_population_max_fitness(population_list):
    return max(individual[1] for individual in population_list)

# tournament_size should be even
def first_EA(pop_size,t_size,m_rate,m_size,cross_point):
    global fitness_num
    fitness_num = 0
    sum_record_iter = 0
    record_iter_list = []
    success_time = 0
    # simulate 100 time ,calculate the probability of success
    for i in range(100):
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
                child_list.extend(crossover(parent_list[i], parent_list[i + 1],cross_point))

            # mutation
            mu_child_list = []
            for child in child_list:
                mu_child_list.append(mutate(child, size=m_size,pro=m_rate))

            # replacement
            for mu_child in mu_child_list:
                replacement(pop_list, mu_child)

            # get_population_max_fitness
            max_population_fitness = get_population_max_fitness(pop_list)

            if max_population_fitness == 15:
                break
            else:
                record_iter += 1

        record_iter_list.append(record_iter)

        sum_record_iter += record_iter
        if (record_iter < 1000 and fitness_num < 100000):
            success_time += 1
    np_record_iter_list = np.array(record_iter_list)
    print(f"mean={sum(record_iter_list) / 100},success_pro={success_time / 100},std={np.std(np_record_iter_list)}")
    print(fitness_num)


    success_rate = success_time / 100
    mean = sum_record_iter / 100
    if success_rate > 0.95:
        return np_record_iter_list
        # return (mean,np.std(np_record_iter_list))
    else :
        return []


class second_EA:
    def __init__(self):
        self.rank = []

    def generate_population(self,size):
        # chromosome will be [pop_size,t_size,m_rate,m_size,cross_rate
        # pop_size between 5 and 20
        # t_size between 2 and 10
        # m_rate between 0 and 1
        # m_size between 0 and 15
        # crossover rate between 0 and 1

        return [[np.random.randint(10, 100), 2 * np.random.randint(1, 5), np.random.rand(), np.random.randint(0, 15),
                 np.random.rand()] for i in range(size)]

    def fitness(self,individual):
        # fitness of an individual is 1000-mean_num_iterations.

        print("evaluating individual, ", individual)
        iterations = first_EA(individual[0], individual[1], individual[2], individual[3], individual[4])
        print("iterations generated")
        # Taking absolute value so we don't end up with negatives ranking higher than the rest
        mean = abs(1000 - np.mean(iterations))
        std = np.std(iterations)

        return (mean, std)

    def assign_ranks(self,population):
        # Takes a population as a list of bits and returns a list of tuples (bitstring,rank)

        population_fitness = []
        for candidate in population:
            fit = self.fitness(candidate)
            population_fitness.append((candidate, fit[0], fit[1]))

        # sorting by the mean ascending
        population_fitness.sort(key=lambda a: a[1])

        ranked_pop = []
        rank = 1

        while True:

            # indexes of the candidates we'll assign this rank to
            pareto_front = []
            compc = population_fitness[0]
            pareto_front.append(0)
            ranked_pop.append((rank, compc[0], compc[1], compc[2]))

            # finding candidates also in this pareto front
            for i in range(1, len(population_fitness)):

                candidate = population_fitness[i]
                if candidate[2] < compc[2]:
                    compc = candidate
                    pareto_front.append(i)
                    ranked_pop.append((rank, candidate[0], candidate[1], candidate[2]))

            # these have been added to
            pareto_front.sort(reverse=True)
            for i in pareto_front:
                population_fitness.pop(i)

            if len(population_fitness) != 0:
                # Now calculate the next pareto front
                rank += 1
                compc = population_fitness[0]
            else:
                break

        return ranked_pop

    def tournament(self,ranked_population, t_size=2):
        # selecting parents based on tournament selection with no replacement

        if t_size >= len(ranked_population):
            print("tournament size greater than or equal to population size")
            return 0

        # creating the probabilities of being selected
        ranks = list(np.array(ranked_population, dtype=object)[:, 0])
        c = Counter(ranks)
        ranks = list(set(ranks))
        rank_sum = sum(ranks)
        rank_prob = {ranks[i]: ranks[i - 1] / rank_sum for i in range(len(ranks))}

        # There are a lot of subltities here because candidates can share ranks and therefore probabilities
        probabilities = [rank_prob[ranked_population[i][0]] / c[ranked_population[i][0]] for i in
                         range(len(ranked_population))]

        parents = []

        for r in range(2):
            tourn = []
            pool = ranked_population.copy()

            # picks random index based on rank probabilities and pops it, adding it to the tournament
            to_pop = np.random.choice(np.arange(len(pool)), size=t_size, replace=False, p=probabilities)
            for index in to_pop:
                tourn.append(pool.pop(index))

            # Pick a candidate to compare the others to from what is left in the pool
            compc = pool[np.random.randint(len(pool))]

            final = []

            for candidate in tourn:
                if candidate[0] > compc[0]:
                    final.append(candidate)

            if len(final) > 1:
                # split the tie randomly between winners
                parents.append(final[np.random.randint(len(final))])
            elif len(final) == 1:
                # We have only one winner
                parents.append(final[0])
            else:
                # We have no winners, so take randomly from those in the tournament
                parents.append(tourn[np.random.randint(len(tourn))])

        return parents

    def crossover(self,parents,pro=1):
        children = []
        for parent_index in range(0,len(parents),2):
            rd_pro = np.random.rand()
            if(rd_pro < pro):
                check_point = random.randint(0,len(parents[parent_index])-1)
                new_child1 = parents[parent_index][0:check_point] + parents[parent_index + 1][check_point:]
                new_child2 = parents[parent_index+1][0:check_point] + parents[parent_index][check_point:]
                children.append(new_child1)
                children.append(new_child2)
            else :
                children.append(parents[parent_index],parents[parent_index+1])

        return children

    def mutation(self,individual,size=1,pro=1):
        new_individual = individual.copy()
        #some parameter have already been mutate
        exit_mutate_para_list = []
        rd_num_list = random.sample(range(0, 4), size)
        for rd_num in rd_num_list:
            # population_size change
            if rd_num == 0:
                change_factor = np.random.randint(10, 100)
                # make sure the change_factor is not equal to old population_size
                while change_factor == new_individual[0]:
                    change_factor = np.random.randint(10, 100)
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
                change_factor = np.random.randint(0, 15)
                # make sure the change_factor is not equal to old population_size
                while change_factor == new_individual[0]:
                    change_factor = np.random.randint(0, 15)
                new_individual[rd_num] = change_factor
            # cross_rate
            else:
                change_factor = np.random.rand()
                # make sure the change_factor is not equal to old population_size
                while change_factor == new_individual[0]:
                    change_factor = np.random.rand()
                new_individual[rd_num] = change_factor


        return new_individual








if __name__ == '__main__':
    # chromosome will be [pop_size,t_size,m_rate,m_size,cross_rate
    # pop_size between 5 and 20
    # t_size between 2 and 10
    # m_rate between 0 and 1
    # m_size between 0 and 15
    # cross_rate between 0 and 1

    pop_size = 100
    t_size = 2
    m_rate = 0.5
    m_size = 1
    cross_rate = 0.5
    test_individual = [pop_size, t_size, m_rate, m_size, cross_rate]
    print(first_EA(46, 6, 0.5445998382417272, 6, 0.9151311211065971))

    # print(second_EA.fitness(test_individual))
    s_EA = second_EA()
    print(s_EA.fitness([46, 6, 0.5445998382417272, 6, 0.9151311211065971]))

    population_list = s_EA.generate_population(10)

    rank_pop = s_EA.assign_ranks(population_list)

    print(rank_pop)
    # print(s_EA.mutation(individual=test_individual,size=2))

