import sys

import torch
import torch.nn.functional as F


class DemoMapper:
    def __init__(self, bin_age):
        self.bin_age = bin_age
        num_age_classes = 120
        if self.bin_age:
            num_age_classes = 11
        #print("Number of age bins =", num_age_classes)
        num_sex_classes = 3
        num_race_classes = 8
        num_ethnicity_classes = 3
        self.num_classes = {
            "age": num_age_classes,
            "race": num_race_classes,
            "sex": num_sex_classes,
            "ethnicity": num_ethnicity_classes,
        }
        self.min_size = sum(self.num_classes.values())

    def map_age(self, age):
        if not self.bin_age:
            return int(age)

        if age <= 1:
            return 1
        elif age <= 4:
            return 2
        elif age <= 14:
            return 3
        elif age <= 24:
            return 4
        elif age <= 34:
            return 5
        elif age <= 44:
            return 6
        elif age <= 54:
            return 7
        elif age <= 64:
            return 8
        elif age <= 74:
            return 9
        elif age <= 84:
            return 10
        else:
            return 0

    def map_race(self, race):
        race = str(race).lower()
        if "american indian" in race:
            return 1
        elif "asian" in race:
            return 2
        elif "black" in race or "african american" in race:
            return 3
        elif "native hawaiian" in race:
            return 4
        elif "white" in race:
            return 5
        elif "hispanic" in race:
            return 6
        elif "other" in race:
            return 7
        else:  # refused or unknown
            #if race:
            #    print(f"Unhandled race: {race}")
            return 0

    def map_sex(self, sex):
        sex = sex.lower()
        if sex == "female":
            return 1
        elif sex == "male":
            return 2
        else:
            #if sex:
            #    print(f"Unhandled sex: {sex}")
            return 0


    def map_ethnicity(self, eth):
        eth = eth.lower()
        if eth == "hispanic or latino":
            return 1
        elif eth == "not hispanic or latino":
            return 2
        else:
            #if eth:
            #    print(f"Unhandled ethnicity: {eth}")
            return 0

    def get_demographics_embed(self, patient_row, embed_size, args):
        """
        encode sex, race, ethnicity and age (refer to cdc age groups for covid)
        returns Embedding = concat(embed(static_attributes))
        """
        if(embed_size < self.min_size):
            print(" Minimum embedding size needed, embed_size", self.min_size,\
                  embed_size)
            sys.exit(1)

        if "demo" in args.features:
            age = self.map_age(int(patient_row.age))
            race = self.map_race(patient_row.race)
            eth = self.map_ethnicity(patient_row.ethnicity)
            sex = self.map_sex(patient_row.sex)
            e1 = F.one_hot(torch.tensor(age), num_classes=self.num_classes["age"])
            e2 = F.one_hot(torch.tensor(race), num_classes=self.num_classes["race"])
            e3 = F.one_hot(torch.tensor(sex), num_classes=self.num_classes["sex"])
            e4 = F.one_hot(torch.tensor(eth), num_classes=self.num_classes["ethnicity"])
        else:
            # Padding
            e1 = torch.zeros(self.num_classes["age"])
            e2 = torch.zeros(self.num_classes["race"])
            e3 = torch.zeros(self.num_classes["sex"])
            e4 = torch.zeros(self.num_classes["ethnicity"])

        # if(str(include_risk_factors.lower()) == "true"):
        if "nlp" in args.features:
            e5 = torch.tensor(patient_row.risk_factors)
            num_pads = embed_size - (
                self.num_classes["age"]
                + self.num_classes["race"]
                + self.num_classes["sex"]
                + self.num_classes["ethnicity"]
                + len(patient_row.risk_factors)
            )
            assert num_pads >= 0
            mytensor = F.pad(
                torch.cat([e1, e2, e3, e4, e5]), (0, num_pads), "constant", 0
            )
        else:
            num_pads = embed_size - (
                self.num_classes["age"]
                + self.num_classes["race"]
                + self.num_classes["sex"]
                + self.num_classes["ethnicity"]
            )

            assert num_pads >= 0
            mytensor = F.pad(torch.cat([e1, e2, e3, e4]), (0, num_pads), "constant", 0)
        # print(patient_data['age'], patient_data['race'], patient_data['ethnicity'],\
        #      self.num_classes['ethnicity'], e1, e2, e3, e4, mytensor)
        # print("demographics.size", mytensor.size())
        return mytensor


class DemoMapper_MACE:
    def __init__(self, bin_age):
        self.bin_age = bin_age
        num_age_classes = 120
        if self.bin_age:
            num_age_classes = 11
        print(f"Number of age bins={num_age_classes}")
        num_sex_classes = 3
        num_race_classes = 9
        num_ethnicity_classes = 5
        self.num_classes = {
            "age": num_age_classes,
            "race": num_race_classes,
            "sex": num_sex_classes,
            "ethnicity": num_ethnicity_classes,
        }
        self.min_size = sum(self.num_classes.values())

    def map_age(self, age):
        if not self.bin_age:
            return int(age)

        if age <= 1:
            return 1
        elif age <= 4:
            return 2
        elif age <= 14:
            return 3
        elif age <= 24:
            return 4
        elif age <= 34:
            return 5
        elif age <= 44:
            return 6
        elif age <= 54:
            return 7
        elif age <= 64:
            return 8
        elif age <= 74:
            return 9
        elif age <= 84:
            return 10
        else:
            return 0

    def map_age_group(self, age_group):
        if age_group == 0:
            return 'default'
        elif age_group == 1:
            return 'Age <= 1'
        elif age_group == 2:
            return 'Age <= 4'
        elif age_group == 3:
            return 'Age <= 14'
        elif age_group == 4:
            return 'Age <= 24'
        elif age_group == 5:
            return 'Age <= 34'
        elif age_group == 6:
            return 'Age <= 44'
        elif age_group == 7:
            return 'Age <= 54'
        elif age_group == 8:
            return 'Age <= 64'
        elif age_group == 9:
            return 'Age <= 74'
        elif age_group == 10:
            return 'Age <= 84'
        else:
            print(f"Unhandled age_group: {age_group}")
            return 'ERROR'

    def map_race(self, race):
        try:
            race = race.lower()
        except AttributeError:
            race = None
        if race == "american indian or alaska native":
            return 1
        elif race == "asian":
            return 2
        elif race == "black or african american":
            return 3
        elif race == "declined to answer":
            return 4
        elif race == "native hawaiian or other pacific islander":
            return 5
        elif race == "unknown by patient":
            return 6
        elif race == "white not of hisp orig":
            return 7
        elif race == "white":
            return 8
        else:
            if race:
                print(f"Unhandled race: {race}")
            return 0

    def map_sex(self, sex):
        try:
            sex = sex.lower()
        except AttributeError:
            sex = None
        if sex == 'f':
            return 1
        elif sex == 'm':
            return 2
        else:
            if sex:
                print(f"Unhandled sex: {sex}")
            return 0

    def map_ethnicity(self, eth):
        try:
            eth = eth.lower()
        except AttributeError:
            eth = None
        if eth == "declined to answer":
            return 1
        elif eth == "hispanic or latino":
            return 2
        elif eth == "not hispanic or latino":
            return 3
        elif eth == "unknown by patient":
            return 4
        else:
            if eth:
                print(f"Unhandled ethnicity: {eth}")
            return 0

    def get_demographics_embed(self, patient_row, embed_size, args):
        """
        encode sex, race, ethnicity and age (refer to cdc age groups for covid)
        returns Embedding = concat(embed(static_attributes))
        """

        if "demo" in args.features:
            try:
                age = self.map_age(int(patient_row.age))
            except ValueError:
                age = 0
            race = self.map_race(patient_row.race)
            eth = self.map_ethnicity(patient_row.ethnicity)
            sex = self.map_sex(patient_row.sex)
            e1 = F.one_hot(torch.tensor(age), num_classes=self.num_classes["age"])
            e2 = F.one_hot(torch.tensor(race), num_classes=self.num_classes["race"])
            e3 = F.one_hot(torch.tensor(sex), num_classes=self.num_classes["sex"])
            e4 = F.one_hot(torch.tensor(eth), num_classes=self.num_classes["ethnicity"])
        else:
            # Padding
            e1 = torch.zeros(self.num_classes["age"])
            e2 = torch.zeros(self.num_classes["race"])
            e3 = torch.zeros(self.num_classes["sex"])
            e4 = torch.zeros(self.num_classes["ethnicity"])

        # if(str(include_risk_factors.lower()) == "true"):
        if "nlp" in args.features:
            e5 = torch.tensor(patient_row.risk_factors)
            num_pads = embed_size - (
                self.num_classes["age"]
                + self.num_classes["race"]
                + self.num_classes["sex"]
                + self.num_classes["ethnicity"]
                + len(patient_row.risk_factors)
            )
            assert num_pads >= 0
            mytensor = F.pad(
                torch.cat([e1, e2, e3, e4, e5]), (0, num_pads), "constant", 0
            )
        else:
            num_pads = embed_size - (
                self.num_classes["age"]
                + self.num_classes["race"]
                + self.num_classes["sex"]
                + self.num_classes["ethnicity"]
            )

            assert num_pads >= 0
            mytensor = F.pad(torch.cat([e1, e2, e3, e4]), (0, num_pads), "constant", 0)
        # print(patient_data['age'], patient_data['race'], patient_data['ethnicity'],\
        #      self.num_classes['ethnicity'], e1, e2, e3, e4, mytensor)
        # print("demographics.size", mytensor.size())
        return mytensor


class DemoMapper_MIMIC:
    def __init__(self, bin_age):
        self.bin_age = bin_age
        num_age_classes = 120
        if self.bin_age:
            num_age_classes = 11
        print(f"Number of age bins={num_age_classes}")
        num_sex_classes = 3
        num_race_classes = 1
        num_ethnicity_classes = 39
        self.num_classes = {
            "age": num_age_classes,
            "race": num_race_classes,
            "sex": num_sex_classes,
            "ethnicity": num_ethnicity_classes,
        }
        self.min_size = sum(self.num_classes.values())

    def map_age(self, age):
        if not self.bin_age:
            return int(age)

        if age <= 1:
            return 1
        elif age <= 4:
            return 2
        elif age <= 14:
            return 3
        elif age <= 24:
            return 4
        elif age <= 34:
            return 5
        elif age <= 44:
            return 6
        elif age <= 54:
            return 7
        elif age <= 64:
            return 8
        elif age <= 74:
            return 9
        elif age <= 84:
            return 10
        else:
            return 0

    def map_age_group(self, age_group):
        if age_group == 0:
            return 'default'
        elif age_group == 1:
            return 'Age <= 1'
        elif age_group == 2:
            return 'Age <= 4'
        elif age_group == 3:
            return 'Age <= 14'
        elif age_group == 4:
            return 'Age <= 24'
        elif age_group == 5:
            return 'Age <= 34'
        elif age_group == 6:
            return 'Age <= 44'
        elif age_group == 7:
            return 'Age <= 54'
        elif age_group == 8:
            return 'Age <= 64'
        elif age_group == 9:
            return 'Age <= 74'
        elif age_group == 10:
            return 'Age <= 84'
        else:
            print(f"Unhandled age_group: {age_group}")
            return 'ERROR'

    def map_race(self, race):
        try:
            race = race.lower()
        except AttributeError:
            race = None
        # no race for mimic
        if race:
            print(f"Unhandled race: {race}")
        return 0

    def map_sex(self, sex):
        try:
            sex = sex.lower()
        except AttributeError:
            sex = None
        if sex == 'f':
            return 1
        elif sex == 'm':
            return 2
        else:
            if sex:
                print(f"Unhandled sex: {sex}")
            return 0

    def map_ethnicity(self, eth):
        try:
            eth = eth.lower()
        except AttributeError:
            eth = None
        if eth == "white":
            return 1
        elif eth == "black/african american":
            return 2
        elif eth == "asian":
            return 3
        elif eth == "unknown/not specified":
            return 4
        elif eth == "hispanic or latino":
            return 5
        elif eth == "patient declined to answer":
            return 6
        elif eth == "multi race ethnicity":
            return 7
        elif eth == "other":
            return 8
        elif eth == "white - russian":
            return 9
        elif eth == "hispanic/latino - puerto rican":
            return 10
        elif eth == "asian - chinese":
            return 11
        elif eth == "asian - asian indian":
            return 12
        elif eth == "asian - cambodian":
            return 13
        elif eth == "hispanic/latino - salvadoran":
            return 14
        elif eth == "american indian/alaska native":
            return 15
        elif eth == "hispanic/latino - central american (other)":
            return 16
        elif eth == "portuguese":
            return 17
        elif eth == "black/cape verdean":
            return 18
        elif eth == "white - brazilian":
            return 19
        elif eth == "black/haitian":
            return 20
        elif eth == "middle eastern":
            return 21
        elif eth == "asian - vietnamese":
            return 22
        elif eth == "native hawaiian or other pacific islander":
            return 23
        elif eth == "hispanic/latino - guatemalan":
            return 24
        elif eth == "hispanic/latino - dominican":
            return 25
        elif eth == "white - other european":
            return 26
        elif eth == "hispanic/latino - cuban":
            return 27
        elif eth == "unable to obtain":
            return 28
        elif eth == "hispanic/latino - colombian":
            return 29
        elif eth == "asian - other":
            return 30
        elif eth == "asian - filipino":
            return 31
        elif eth == "hispanic/latino - mexican":
            return 32
        elif eth == "american indian/alaska native federally recognized tribe":
            return 33
        elif eth == "caribbean island":
            return 34
        elif eth == "black/african":
            return 35
        elif eth == "white - eastern european":
            return 36
        elif eth == "asian - thai":
            return 37
        elif eth == "asian - korean":
            return 38
        else:
            if eth:
                print(f"Unhandled ethnicity: {eth}")
            return 0

    def get_demographics_embed(self, patient_row, embed_size, args):
        """
        encode sex, race, ethnicity and age (refer to cdc age groups for covid)
        returns Embedding = concat(embed(static_attributes))
        """

        if "demo" in args.features:
            try:
                age = self.map_age(int(patient_row.age))
            except ValueError:
                age = 0
            race = self.map_race(patient_row.race)
            eth = self.map_ethnicity(patient_row.ethnicity)
            sex = self.map_sex(patient_row.sex)
            e1 = F.one_hot(torch.tensor(age), num_classes=self.num_classes["age"])
            e2 = F.one_hot(torch.tensor(race), num_classes=self.num_classes["race"])
            e3 = F.one_hot(torch.tensor(sex), num_classes=self.num_classes["sex"])
            e4 = F.one_hot(torch.tensor(eth), num_classes=self.num_classes["ethnicity"])
        else:
            # Padding
            e1 = torch.zeros(self.num_classes["age"])
            e2 = torch.zeros(self.num_classes["race"])
            e3 = torch.zeros(self.num_classes["sex"])
            e4 = torch.zeros(self.num_classes["ethnicity"])

        # if(str(include_risk_factors.lower()) == "true"):
        if "nlp" in args.features:
            e5 = torch.tensor(patient_row.risk_factors)
            num_pads = embed_size - (
                self.num_classes["age"]
                + self.num_classes["race"]
                + self.num_classes["sex"]
                + self.num_classes["ethnicity"]
                + len(patient_row.risk_factors)
            )
            assert num_pads >= 0
            mytensor = F.pad(
                torch.cat([e1, e2, e3, e4, e5]), (0, num_pads), "constant", 0
            )
        else:
            num_pads = embed_size - (
                self.num_classes["age"]
                + self.num_classes["race"]
                + self.num_classes["sex"]
                + self.num_classes["ethnicity"]
            )

            assert num_pads >= 0
            mytensor = F.pad(torch.cat([e1, e2, e3, e4]), (0, num_pads), "constant", 0)
        # print(patient_data['age'], patient_data['race'], patient_data['ethnicity'],\
        #      self.num_classes['ethnicity'], e1, e2, e3, e4, mytensor)
        # print("demographics.size", mytensor.size())
        return mytensor
