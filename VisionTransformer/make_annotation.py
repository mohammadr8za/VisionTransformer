import os
import glob
from os.path import join, split
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def write_to_csv(save_path, train_data, valid_data, test_data):
    train_data_dict = {"path": [],
                       "folder": [],
                       "data": []
                       }
    test_data_dict = {"path": [],
                      "folder": [],
                      "data": []
                      }
    valid_data_dict = {"path": [],
                       "folder": [],
                       "data": []
                       }
    for data in train_data:
        train_data_dict["data"].append(split(data)[-1])
        train_data_dict["folder"].append(split(split(data)[0])[-1])
        train_data_dict["path"].append(split(split(data)[0])[0])

    for data in valid_data:
        valid_data_dict["data"].append(split(data)[-1])
        valid_data_dict["folder"].append(split(split(data)[0])[-1])
        valid_data_dict["path"].append(split(split(data)[0])[0])

    for data in test_data:
        test_data_dict["data"].append(split(data)[-1])
        test_data_dict["folder"].append(split(split(data)[0])[-1])
        test_data_dict["path"].append(split(split(data)[0])[0])

        # Create Data Frame for All Sets
    train_data_frame = pd.DataFrame(train_data_dict)
    valid_data_frame = pd.DataFrame(valid_data_dict)
    test_data_frame = pd.DataFrame(test_data_dict)

    # Write Data Frames into .csv files
    train_data_frame.to_csv(join(save_path, "train_annotation.csv"))
    valid_data_frame.to_csv(join(save_path, "valid_annotation.csv"))
    test_data_frame.to_csv(join(save_path, "test_annotation.csv"))


def make_annotation(root, mode="train_test_folder"):

    if mode == "train_test_folder":

        train_path = join(root, "train")
        test_path = join(root, "test")

        train_data_list = glob.glob(join(train_path, "*/*.jpg"))
        # Separate Validation Set from Train Set
        train_data_array, valid_data_array = train_test_split(np.array(train_data_list), train_size=0.8, test_size=0.2)

        train_data_list, valid_data_list = list(train_data_array), list(valid_data_array)
        test_data_list = glob.glob(join(test_path, "*/*.jpg"))

        write_to_csv(save_path=root,
                     train_data=train_data_list,
                     valid_data=valid_data_list,
                     test_data=test_data_list)

        # Create a .csv file to contain all data
        all_data_list = glob.glob(join(root, "*/*/*.jpg"))
        all_data_dict = {"path": [],
                         "folder": [],
                         "data": []
                         }
        for data in all_data_list:
            all_data_dict["data"].append(split(data)[-1])
            all_data_dict["folder"].append(split(split(data)[0])[-1])
            all_data_dict["path"].append(split(split(data)[0])[0])

        all_data_frame = pd.DataFrame(all_data_dict)
        all_data_frame.to_csv(join(root, "all_in_one_annotation.csv"))

    if mode == "all_in_one_folder":
        data_list = glob.glob(join(root, "*/*.jpg"))
        data_dict = {"path": [],
                     "folder": [],
                     "data": []}

        train_data, test_valid_data = train_test_split(np.array(data_list), train_size=0.7, test_size=0.3)
        valid_data, test_data = train_test_split(test_valid_data, train_size=0.6, test_size=0.4)

        write_to_csv(save_path=root,
                     train_data=train_data,
                     valid_data=valid_data,
                     test_data=test_data)

        # Create a .csv file to contain all data
        for data in data_list:
            data_dict["data"].append(split(data)[-1])
            data_dict["folder"].append(split(split(data)[0])[-1])
            data_dict["path"].append(split(split(data)[0])[0])

        all_data_frame = pd.DataFrame(data_dict)
        all_data_frame.to_csv(join(root, "all_in_one_annotation.csv"))


if __name__ == "__main__":
    root = r"D:\mreza\TestProjects\Python\DL\ViT\Data\pizza_sushi_steak"
    make_annotation(root)
