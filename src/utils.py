import pandas as pd

def load_standard_data(path: "data/students_performance_dataset.csv"):
    df = pd.read_csv(path)
    allowed_columns = [
        'Attendance',
        'Midterm',
        'final',
        'Projects',
        'Study_Hours',
    ]
    return df[allowed_columns]

def load_biased_data(path: "data/students_performance_dataset.csv"):
    df = pd.read_csv(path)
    biased_allowed_columns = [
        'Attendance',
        'Midterm',
        'final',
        'Projects',
        'Study_Hours',
        'Gender',
        'Parental_Level_of_Education',
        'Internet_Access',
    ]
    return df[biased_allowed_columns]