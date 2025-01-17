import math

import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
congressional_voting_records = fetch_ucirepo(id=105)

# data (as pandas dataframes)
X = congressional_voting_records.data.features
y = congressional_voting_records.data.targets

percentages = []


def rep_or_dem(roww, n_dems, n_reps, dems, reps, q_dems, q_reps, vapros):
    chance_rep = 1
    chance_dem = 1
    for i, (column, value) in enumerate(roww[:-1].items()):
        if value == 'n':
            chance_rep *= (n_reps[i] + 1) / (reps_num + 2)
            chance_dem *= (n_dems[i] + 1) / (dems_num + 2)
        elif value == 'y':
            chance_rep *= (reps[i] + 1) / (reps_num + 2)
            chance_dem *= (dems[i] + 1) / (dems_num + 2)
        elif value == '?' and vapros:
            chance_rep *= (q_reps[i] + 1) / (reps_num + 2)
            chance_dem *= (q_dems[i] + 1) / (dems_num + 2)
        elif value == '?' and not vapros:
            chance_rep *= ((n_reps[i] + 1) / (reps_num + 2)) + ((reps[i] + 1) / (reps_num + 2))
            chance_dem *= ((n_dems[i] + 1) / (dems_num + 2)) + ((dems[i] + 1) / (dems_num + 2))
    chance_rep *= (reps_num + 1) / (train_size + 2)
    chance_dem *= (dems_num + 1) / (train_size + 2)
    if chance_rep > chance_dem:
        return "republican"
    return "democrat"


data = pd.concat([X, y], axis=1)
data = data.sample(frac=1).reset_index(drop=True)

reps_num = 0
dems_num = 0

for idx, row in data.iterrows():
    if row['Class'] == "democrat":
        dems_num += 1
    if row['Class'] == "republican":
        reps_num += 1

train_list = []
test_list = []

train_constraint_reps = 0.8 * reps_num
train_constraint_dems = 0.8 * dems_num

counter_dems = 0
counter_reps = 0
for idx, row in data.iterrows():
    if row['Class'] == "democrat" and counter_dems < train_constraint_dems:
        train_list.append(row)
        counter_dems += 1
    elif row['Class'] == "republican" and counter_reps < train_constraint_reps:
        train_list.append(row)
        counter_reps += 1
    else:
        test_list.append(row)

train_set = pd.DataFrame(train_list)
test_set = pd.DataFrame(test_list)
train_size = len(train_set)


def NaiveBayes(train_set, test_set, vapros):
    num_features = X.shape[1]
    reps = [0] * num_features
    n_reps = [0] * num_features
    dems = [0] * num_features
    n_dems = [0] * num_features
    q_dems = [0] * num_features
    q_reps = [0] * num_features

    for index, row in train_set.iterrows():
        if row['Class'] == "republican":
            for idx, (column, value) in enumerate(row[:-1].items()):
                if row[column] == 'y':
                    reps[idx] += 1
                elif row[column] == 'n':
                    n_reps[idx] += 1
                elif row[column] == '?':
                    q_reps[idx] += 1
        if row['Class'] == "democrat":
            for idx, (column, value) in enumerate(row[:-1].items()):
                if row[column] == 'y':
                    dems[idx] += 1
                elif row[column] == 'n':
                    n_dems[idx] += 1
                elif row[column] == '?':
                    q_dems[idx] += 1
    counter = 0
    for index, row in test_set.iterrows():
        if row['Class'] == rep_or_dem(row, n_dems, n_reps, dems, reps, q_dems, q_reps, vapros):
            counter += 1

    accuracy = (counter / len(test_set)) * 100
    percentages.append(accuracy)
    print(f"Accuracy: {accuracy:.2f}%")


NaiveBayes(train_set, train_set, 1)
print()

partitioner = train_size / 10

for i in range(10):
    test_data = train_set.iloc[int(i * partitioner):int((i + 1) * partitioner)]
    train_data = train_set.drop(index=train_set.iloc[int(i * partitioner):int((i + 1) * partitioner)].index)
    NaiveBayes(train_data, test_data, True)

mean = 0
for i in range(1, 11):
    mean += percentages[i]

mean /= 10

variance = 0
for i in range(1, 11):
    variance += (percentages[i] - mean) ** 2

variance/=10
variance = math.sqrt(variance)

print(f"Mean: {mean:.2f}%")
print(f"Variance: {variance:.2f}%")
print()

NaiveBayes(train_set, test_set, True)