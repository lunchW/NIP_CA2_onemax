import pandas as pd
import matplotlib.pyplot as plt
if __name__ == '__main__':

    input_data = pd.read_excel("../data/output-2EA4.xls")
    print(input_data)
    input_data_iter_mean = input_data
    # input_data.plot()
    # print(input_data)
    print(input_data_iter_mean[:])
    # print(input_data["mutation_rate"])
    x_list = []
    y_list = []
    for x in input_data_iter_mean["iter"]:
        x_list.append(x)
    for y in input_data_iter_mean["mean_value"]:
        y_list.append(y)
    # plt.scatter(x_list,y_list)
    # tournament_size(2-6) step = 2 repeat 5 time v mean_value
    plt.plot(input_data_iter_mean["iter"],input_data_iter_mean["mean_value"])
    plt.title("(40,2,0.8,2,1)repeat 5000 iteration v mean_value_generation_first_EA")
    plt.xlabel("iter")
    plt.ylabel("mean_value_generation_first_EA")
    plt.show()

