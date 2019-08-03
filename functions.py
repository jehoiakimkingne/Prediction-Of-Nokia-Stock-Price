# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:40:24 2019

@author: Jéhoiakim KINGNE
"""

# Convert data to a csv file

#IMPORTATIONS
import csv
import requests

#FUNCTIONS DEFINITION

def WriteListToCSV(csv_file,csv_columns,data_list):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(csv_columns)
            for data in data_list:
                writer.writerow(data)
    except IOError:
            print("error")    
    return csv_file  

def extract(URL):
    myrequest = requests.get(url = URL)       
    quandlData = myrequest.json()
    return quandlData