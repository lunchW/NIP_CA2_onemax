# NIP_CA2_onemax

## environment

This program needs to be run in pycharm, the packages needed are matplotlib and pandas, tqdm, numpy

## 1.first EA(simple-max solver)

In the first EA, pop_size, t_size, m_rate, m_size, cross_rate are passed in.

The program will use this set of parameters to simulate 100 times and record whether it succeeds or not. The success condition is that it must not exceed 1000 iterations and the fitness function must not run more than 100000 times. Finally, if the success rate of the simulation is higher than 95, the number of successful simulations is added up and divided by the number of successes to obtain the average number of iterations for the 100 simulations and return. If the success rate of this experiment is less than 95, then -1 is returned. Use this read_excel.py to output this table as an iterative vs mean_value

### 1.1 generate function

parameter ------(gen_num)

Generate a list of N(gen_num) randomly generated 15-digit 0 or 1 numbers based on the incoming parameters.

### 1.2 fitness function

parameter ------(individual)

Add up all the numbers of an individual to become the fitness value

### 1.3 tournament function

parameter ------(tour-size,population_list)

N individuals were selected from the population, and the best one was chosen as the parent according to their fitness value.

### 1.4 crossover function

parameter ------(i1,i2,pro)

Two parents were selected and nodes were randomly chosen to interchange their genes for recombination

### 1.5 mutate function

parameter ------(individual,size,pro)

According to the incoming parameters, N genes(size) in an individual are randomly selected for mutation, and the chance of mutation is S(pro)

### 1.6 replacement function

parameter ------(population_list,individual)

The incoming population and the mutated recombinant individual are compared with the worst in the population and replaced if they are better, otherwise discarded

## 2. second EA

Fix second_pop_size,second_m_size,second_m_rate for testing, then we can set the number of terminations,Then the number of iterations for each iteration, and the average of the iterations obtained from this simulation are stored in the list. After N runs, the table is output.

```
termination_criterion = 10000
```

### 2.1 generate function

parameter ------(size)

5 randomly generated parameters within a fixed range are stored in an array and treated as an individual. Then generate N such individuals

### 2.2 fitness function

parameter ------(individual)

Substitute the individual parameters into the first_EA function to obtain an iterative average, the larger the value the better

### 2.3 tournament function

parameter ------(population_list,tour_size,)

N individuals were selected from the population, and the best one was chosen as the parent according to their fitness value.

### 2.4 crossover function

parameter ------(i1,i2,pro)

Two parents were selected and nodes were randomly chosen to interchange their genes for recombination

### 2.5 mutate function

parameter ------(individual,size,pro)

According to the incoming parameters, N genes(size) in an individual are randomly selected for mutation, and the chance of mutation is S(pro)

### 2.6 replacement function

parameter ------(population_list,individual)

The incoming population and the mutated recombinant individual are compared with the worst in the population and replaced if they are better, otherwise discarded

