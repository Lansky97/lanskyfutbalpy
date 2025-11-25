import csv
import pandas as pd

def read_results(file_path):
    results = []
    with open(file_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            results.append(row)
    return results

def get_points(home_goals, away_goals):
    if home_goals > away_goals:
        return 3, 0
    elif home_goals < away_goals:
        return 0, 3
    else:
        return 1, 1
    