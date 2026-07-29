# Copyright 2025 Forschungszentrum Juelich GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File author: Lena Krieger

import numpy as np
import pandas as pd


def synthetic_balance(sens_attr, predictions, sensitive):
    predictions = np.array(predictions)
    # distribution of labels
    labels, counts = np.unique(predictions, return_counts=True)
    # number of data points
    C = counts.sum()
    if -1 in labels:
        # number of datapoints - noise points
        C = C - counts[0]
    # get sensitive columns
    if len(sens_attr) == 1:
        if len(sensitive.shape) !=2:
            sensitive_columns = np.expand_dims(sensitive, axis=1)
        else:
            sensitive_columns = sensitive
        G = calculate_group_membership(sensitive_columns)
    else:
        sensitive_columns = sensitive
        G = calculate_group_membership_multiple(sensitive_columns)

    sensitive_groups = pd.DataFrame(
        G, columns=["sensitive_{}".format(x) for x in range(len(G[0]))]
    )
    sens_dict = {}
    counter = 0
    sensitive = np.array(sensitive)
    if len(sens_attr) > 1:
        for i, attr in enumerate(sens_attr):
            num_groups = len(np.unique(sensitive_columns[attr]))
            sens_dict[attr] = sensitive_groups.iloc[:, counter: num_groups + counter]
            counter = counter + num_groups
    else:
        num_groups = len(np.unique(sensitive_columns))
        sens_dict[sens_attr[0]] = sensitive_groups

    # indices for noise points
    indices = [i for i, label in enumerate(predictions) if label == -1]
    # indices = [i for i, label in enumerate(predictions) if label == -1]
    # we remove all data points that are assigned noise points
    predictions = np.array(predictions).reshape(-1)
    return balance(predictions, indices, sens_dict, C, per_cluster=True)


def balance_mixed(dataname,sens_attr, predictions, sensitive, per_cluster =False):
    if 'synthetic' in dataname:
        return synthetic_balance(sens_attr,sensitive, predictions)
    predictions = np.array(predictions)
    # distribution of labels
    labels, counts = np.unique(predictions, return_counts=True)
    # number of data points
    C = counts.sum()
    if -1 in labels:
        # number of datapoints - noise points
        C = C - counts[0]
    # get sensitive columns
    if len(sens_attr) == 1:
        if len(sensitive.shape) !=2:
            sensitive_columns = np.expand_dims(sensitive, axis=1)
        else:
            sensitive_columns = sensitive
        G = calculate_group_membership(sensitive_columns)
    sensitive_groups = pd.DataFrame(
        G, columns=["sensitive_{}".format(x) for x in range(len(G[0]))]
    )


    sens_dict = {}
    counter = 0
    sensitive = np.array(sensitive)
    if len(sens_attr) > 1:
        for i, attr in enumerate(sens_attr):
            num_groups = len(np.unique(sensitive_columns[attr]))
            sens_dict[attr] = sensitive_groups.iloc[:, counter: num_groups + counter]
            counter = counter + num_groups
    else:
        num_groups = len(np.unique(sensitive_columns))
        sens_dict[sens_attr[0]] = sensitive_groups
    # indices for noise points
    indices = np.asarray(predictions == -1).nonzero()[0]
    predictions = np.array(predictions).reshape(-1)
    # indices = [i for i, label in enumerate(predictions) if label == -1]
    # we remove all data points that are assigned noise points
    return balance(predictions, indices, sens_dict, C, per_cluster)

def balance_score(dataname,sens_attr, predictions, sensitive, per_cluster =False):
    if 'synthetic' in dataname:
        return synthetic_balance(sens_attr,sensitive, predictions)
    predictions = np.array(predictions)
    # distribution of labels
    labels, counts = np.unique(predictions, return_counts=True)
    # number of data points
    C = counts.sum()
    if -1 in labels:
        # number of datapoints - noise points
        C = C - counts[0]
    # get sensitive columns

    #print("len sens attr ", len(sens_attr))

    if len(sens_attr) == 1:
        if len(sensitive.shape) !=2:
            sensitive_columns = np.expand_dims(sensitive, axis=1)
        else:
            sensitive_columns = sensitive
        G = calculate_group_membership(sensitive_columns)
    else:
        sensitive_columns = sensitive
        G = calculate_group_membership_multiple(sensitive_columns)

    sensitive_groups = pd.DataFrame(
        G, columns=["sensitive_{}".format(x) for x in range(len(G[0]))]
    )


    sens_dict = {}
    counter = 0
    sensitive = np.array(sensitive)
    if len(sens_attr) > 1:
        for i, attr in enumerate(sens_attr):
            num_groups = len(np.unique(sensitive_columns[attr]))
            sens_dict[attr] = sensitive_groups.iloc[:, counter: num_groups + counter]
            counter = counter + num_groups
    else:
        num_groups = len(np.unique(sensitive_columns))
        sens_dict[sens_attr[0]] = sensitive_groups
    # indices for noise points
    indices = np.asarray(predictions == -1).nonzero()[0]
    predictions = np.array(predictions).reshape(-1)
    # indices = [i for i, label in enumerate(predictions) if label == -1]
    # we remove all data points that are assigned noise points
    return balance(predictions, indices, sens_dict, C, per_cluster)
def balance(predictions, indices, sens_dict,C, per_cluster=False):

    assigned_labels = np.delete(predictions, indices, axis=0)

    balance_per_group_per_cluster = {}
    labels, counts = np.unique(assigned_labels, return_counts=True)
    balance_per_attr_per_cluster = {}
    balance_per_attribute = {}
    for attribute in sens_dict:
        balance_per_group_per_cluster[attribute] = {}
        sensitive_groups = sens_dict[attribute]
        balance_per_cluster = []
        #with WorkerPool(n_jobs=1, shared_objects=(sensitive_groups, indices, assigned_labels,C)) as pool:
        #    balance_per_cluster = pool.map(balance_mapping, zip(labels, counts),iterable_len=len(counts), progress_bar=True)
        for label, count in zip(labels, counts):
            if per_cluster:
                balances, p = balance_mapping((sensitive_groups, indices, assigned_labels,C, True), label, count)
                balance_per_group_per_cluster[attribute][label] = p
            else:
                balances = balance_mapping((sensitive_groups, indices, assigned_labels,C, per_cluster), label, count)

            balance_per_cluster.append(balances)


        if len(balance_per_cluster) == 0:
            balance_per_attribute[attribute] = 0
            balance_per_attr_per_cluster[attribute] = 0
        else:
            balance_per_attribute[attribute] = sum(balance_per_cluster) / len(
                balance_per_cluster
            )
            balance_per_attr_per_cluster[attribute] = balance_per_cluster

    if len(balance_per_attribute) == 1:
        if per_cluster:
            for key in balance_per_attribute:
                return balance_per_attribute[key], balance_per_attr_per_cluster[key], balance_per_group_per_cluster[key]
        else:
            for key in balance_per_attribute:
                return balance_per_attribute[key]
    else:
        if per_cluster:
            return balance_per_attribute, balance_per_attr_per_cluster, balance_per_group_per_cluster
        return balance_per_attribute



def balance_mapping(shared, label, count):
    sensitive_groups, indices, assigned_labels, C, per_cluster = shared
    balance_per_group = []

    for sensitive_group in sensitive_groups:
        
        # print('min: {}'.format(mi, '.2f'))
        # get column corresponding to group sensitive group
        sensitive_values = np.array(sensitive_groups[sensitive_group])
        # get sensitive values only for non-noise points
        sensitive_values = np.delete(sensitive_values, indices, axis=0)
        # number of points in dataset
        C = C
        # number of points in dataset that belong to group i
        C_i = sum(sensitive_values)
        # mask all elements in cluster f
        n_assigned = len(assigned_labels)
        if len(sensitive_values) > n_assigned:
            sensitive_values = sensitive_values[:n_assigned]
        masked = sensitive_values[assigned_labels == label]
        # number of points in cluster f that belong to group i
        C_f_i = sum(masked)
        # number of points in cluster f
        C_f = count
        # ratio of elements in group i in contrast to dataset
        r_i = C_i / C
        # ratio of elements in group i in contrast to cluster f
        r_f_i = C_f_i / C_f
        if r_f_i == 0:
            balance_per_group.append(0)
        else:
            balance_per_group.append(min(float(r_i / r_f_i), float(r_f_i / r_i)))
    if per_cluster:
        return min(balance_per_group), balance_per_group
    return min(balance_per_group)


def calculate_group_membership(sensitive):
    """
    Data_to_list transforms the data to list.

        Parameters:
            data_loader : DataLoader object encapsulating the data.

        Returns:
            group_membership_vector.
    """
    # we get columns with sensitive values encoded as numbers
    # empty groupmembership- vector
    G = []
    # load dataframe including the sensitive attributes
    sensitive_columns = np.array(sensitive)
    names = ['group_{}'.format(i) for i in range(sensitive.shape[1])]
    columns = pd.DataFrame(sensitive_columns, columns=names)

    for sensitive_column in names:
        column = columns[sensitive_column]
        choice = np.unique(column)
        for c in choice:
            # create a 2D array filled with 0's (size is number of samples * number of sensitive attributes in this column)
            encoded_array = np.zeros(len(sensitive), dtype=int)
            idx = np.asarray(column == c).nonzero()[0]
            # one hot encoding
            encoded_array[idx] = 1
            G.append(encoded_array)
    G = np.stack(G, axis=-1)
    return G

def calculate_group_membership_multiple(sensitive):
    """
    Data_to_list transforms the data to list.

        Parameters:
            data_loader : DataLoader object encapsulating the data.

        Returns:
            group_membership_vector.
    """
    # we get columns with sensitive values encoded as numbers
    # empty groupmembership- vector
    G = []
    # load dataframe including the sensitive attributes
    print("sensitive : ", sensitive.columns)
    for i in sensitive.columns:
        column = sensitive[i]
        choice = np.unique(column)
        for c in choice:
            # create a 2D array filled with 0's (size is number of samples * number of sensitive attributes in this column)
            encoded_array = np.zeros(len(sensitive), dtype=int)
            idx = np.asarray(column == c).nonzero()[0]
            # one hot encoding
            encoded_array[idx] = 1
            G.append(encoded_array)
    G = np.stack(G, axis=-1)
    return G

def balance_wo_centers(data_loader, predictions):
    """
        Checks fairness for each of the clusters defined by the clustering.
        Returns balance using the total and class counts.

            Parameters:
                data_loader : data_loader
                predictions (list) : assigned cluster labels.

            Returns:
                balance (float) : balance of the clustering.
    """

    predictions = np.array(predictions)
    # distribution of labels
    labels, counts = np.unique(predictions, return_counts=True)
    # number of data points
    C = counts.sum()
    if -1 in labels:
       # number of datapoints - noise points
       C = C - counts[0]

    # get sensitive columns
    sensitive_columns = data_loader.get_sensitive_columns()
    # if only one column
    if len(sensitive_columns == 1):
        # G is group membership vector
        G = calculate_group_membership4(data_loader)
        sensitive_groups = pd.DataFrame(G, columns = ['sensitive_{}'.format(x) for x in range(len(G[0]))] )
    # indices for noise points
    indices = [i for i, label in enumerate(predictions) if label == -1]
    # we remove all data points that are assigned noise points
    assigned_labels = np.delete(predictions, indices, axis=0)
    labels, counts = np.unique(assigned_labels, return_counts=True)
    balance_per_cluster = []
    # for each cluster
    for label, count in zip(labels, counts):
        balance_per_group =[]
        # for each sensitive attribute
        for sensitive_group in sensitive_groups:
            # get column corresponding to group sensitive group
            sensitive_values = sensitive_groups[sensitive_group]
            # get sensitive values only for non-noise points
            sensitive_values = np.delete(sensitive_values, indices, axis=0)
            # number of points in dataset
            C = C
            # number of points in dataset that belong to group i
            C_i = sum(sensitive_values)
            # mask all elements in cluster f
            masked = sensitive_values[assigned_labels == label]
            # number of points in cluster f that belong to group i
            C_f_i = sum(masked)
            # number of points in cluster f
            C_f = count
            # ratio of elements in group i in contrast to dataset
            r_i = C_i / C
            # ratio of elements in group i in contrast to cluster f
            r_f_i = C_f_i / C_f
            #print("Cluster {}".format(label) )
            #print('Sensitive Label: ',sensitive_group)
            #print('C_i: {}'.format(C_i, '.2f'))
            #print('C: {}'.format(C, '.2f'))
            #print('r_i: {}'.format(r_i))

            #print('C_f_i: {}'.format(C_f_i, '.2f'))
            #print('C_f: {}'.format(C_f, '.2f'))
            #print('r_f_i: {}'.format(r_f_i))
            if r_f_i == 0:
                #print('Zero occured')
                balance_per_group.append(0)
            else:
                mi = min(float(r_i / r_f_i), float(r_f_i / r_i))
                #print('min: {}'.format(mi, '.2f'))
                balance_per_group.append(mi)
        balance_per_cluster.append(min(balance_per_group))
    if len(balance_per_cluster) == 0:
        return 0
    return sum(balance_per_cluster)/len(balance_per_cluster)


def calculate_group_membership4(data_loader):
    """
            Data_to_list transforms the data to list.

                Parameters:
                    data_loader : DataLoader object encapsulating the data.

                Returns:
                    group_membership_vector.
    """
    # we get columns with sensitive values encoded as numbers
    # empty groupmembership- vector
    G = []
    # load dataframe including the sensitive attributes
    sensitive_columns = data_loader.get_sensitive_columns()
    # for each sensitive column
    
    for sensitive_column in sensitive_columns:
        # take the column
        column = sensitive_columns[sensitive_column]
        # create a 2D array filled with 0's (size is number of samples * number of sensitive attributes in this column)
        encoded_array = np.zeros((column.size, column.max() + 1), dtype=int)
        # one hot encoding
        encoded_array[np.arange(column.size), column] = 1
        G.append(encoded_array)
    G = np.concatenate(G, axis=1)
    return G