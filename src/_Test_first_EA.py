import random
import numpy as np
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
        return mean

        # return (mean,np.std(np_record_iter_list))
    else :
        return []
if __name__ == '__main__':
    test_parameter = [10, 2, 0.026345243938668328, 2, 0.1758735078865602]
    # [[15, 2, 0.008166671853144014, 4, 0.1922731586416837], 425.32]
    mean = first_EA(15, 2, 0.008166671853144014, 4, 0.1922731586416837)