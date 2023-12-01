# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:03 2023

@author: GZ57NM
"""

import os

def create_folders(path, limit):
    for i in range(6, limit + 1):
        folder_name = f'Experiment {i}'
        try:
            os.mkdir(os.path.join(path, folder_name))
            print(f"Folder '{folder_name}' created successfully.")
        except FileExistsError:
            print(f"Folder '{folder_name}' already exists.")

if __name__ == '__main__':
    try:
        path = input("Enter the path where you want to create the folders: ")
        limit = int(input("Enter the number of folders to create: "))
        create_folders(path, limit)
    except ValueError:
        print("Please enter a valid number.")