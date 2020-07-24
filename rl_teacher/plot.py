import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import datetime, sys

def read_csv(filename):
    x = []
    y = []
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            x_i = eval(row["Step"])
            y_i = eval(row["Value"])
            if x_i <= 31000:
                x.append(x_i)
                y.append(y_i)
    
    return [x,[y]]
            

def get_data(rst_type,scenario):
    a = read_csv('./csv/{0}/{1}/0.csv'.format(rst_type,scenario))
    b = read_csv('./csv/{0}/{1}/1.csv'.format(rst_type,scenario))
    c = read_csv('./csv/{0}/{1}/2.csv'.format(rst_type,scenario))
    d = read_csv('./csv/{0}/{1}/3.csv'.format(rst_type,scenario))
    if rst_type == 'shprl':
        e = read_csv('./csv/{0}/{1}/4.csv'.format(rst_type,scenario))
    else:
        e = [None,[]]
    return a,b,c,d,e

def plot_graph(rst_type,scenario):
    results = get_data(rst_type,scenario)
    fig = plt.figure()
    fig.set_size_inches(8,5)
    if rst_type == 'shprl':
        label = ['Baseline', 'Fixed 1400 Labels', 'Scaled 1400 Labels', 'Fixed 700 Labels', 'Scaled 700 Labels']
    else:
        label = ['Baseline', 'Scaled 1400 Labels (No agent predict)', r'Scaled 1400 Labels (30% agent predict)', r'Scaled 1400 Labels (50% agent predict)']
    sns.tsplot(time=results[0][0], data=results[0][1], condition=label[0], color='k', linestyle='-')
    sns.tsplot(time=results[1][0], data=results[1][1], condition=label[1], color='C0', linestyle='-')
    sns.tsplot(time=results[2][0], data=results[2][1], condition=label[2], color='C1', linestyle='-')
    sns.tsplot(time=results[3][0], data=results[3][1], condition=label[3], color='C2', linestyle='-')
    if rst_type == 'shprl':
        sns.tsplot(time=results[4][0], data=results[4][1], condition=label[4], color='C3', linestyle='-')

    plt.title(scenario, fontsize=20)
    plt.ylabel("Reward Value", fontsize=15)
    plt.xlabel(r'Timesteps $\times 10^7$', fontsize=15, labelpad = 0)
    plt.legend(loc='bottom right')
    plt.savefig('./output/{0}_{1}.png'.format(rst_type,scenario), dpi=200)

plot_graph('shprl','hopper')
plot_graph('shprl','swimmer')
plot_graph('shprl','walker')
plot_graph('shprl','cheetah')
plot_graph('shprl','ant')
plot_graph('harl','hopper')
plot_graph('harl','swimmer')
plot_graph('harl','walker')
plot_graph('harl','cheetah')
plot_graph('harl','ant')