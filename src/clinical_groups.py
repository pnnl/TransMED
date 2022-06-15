import sys

import torch


class ClinicalGroupMapper:
    def __init__(self, group_file, embed_size):
        self.group_dict = self.get_groups(group_file)
        self.num_groups = len(self.group_dict)
        self.embed_size = embed_size
        if(embed_size < self.num_groups):
            print(f"Min embedding size needed to encode groups {self.num_groups}")
            sys.exit(1)
        self.group_id = dict()
        for label, i in zip(self.group_dict.keys(), range(self.num_groups)):
            self.group_id[label] = i
        print(self.group_id)

    def map_to_groups(self, patient_row):
        """
        takes in a patient row in dataframe
        and generates group presence or absence vector of size embed_size
        """
        group_vector = torch.zeros(self.embed_size)
        for group_label, features in self.group_dict.items():
            group_presence = sum([patient_row.feature for feature in features])
            group_vector[self.group_id[group_label]] = group_presence
        return group_vector

    def get_groups(self, filename):
        """
        returns a group_label to feature dictionary
        by parsing a csv file
        {group_name : [list_of_features]}
        """
        groups_dict = dict()
        with open(filename, "r") as f:
            header = f.readline()
            for line in f.readline():
                arr = line.strip().split(",")
                groups_dict[arr[0]] = arr[1:]
        return groups_dict
