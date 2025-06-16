import pandas as pd
import numpy as np

np.random.seed(42)


data = {
    'name': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Emma Brown',
             'Frank Miller', 'Grace Lee', 'Henry Garcia', 'Isabel Martinez', 'Jack Taylor'],
    'subject': ['Math', 'Science', 'English', 'History', 'Math',
                'Science', 'English', 'History', 'Math', 'Science'],
    'score': np.random.randint(50, 101, 10),  
    'grade': [''] * 10  
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\n" + "="*50 + "\n")

def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['grade'] = df['score'].apply(assign_grade)

print("DataFrame with assigned grades:")
print(df)
print("\n" + "="*50 + "\n")

grade_counts = df['grade'].value_counts().sort_index()
print("Grade Distribution:")
print(grade_counts)
print("\n" + "="*50 + "\n")

print("Score Statistics:")
print(f"Average Score: {df['score'].mean():.2f}")
print(f"Highest Score: {df['score'].max()}")
print(f"Lowest Score: {df['score'].min()}")
print(f"Standard Deviation: {df['score'].std():.2f}")

print("\n" + "="*60 + "\n")

df_sorted = df.sort_values('score', ascending=False)

print("DataFrame sorted by score (descending order):")
print(df_sorted)

print("\n" + "="*60 + "\n")

subject_averages = df.groupby('subject')['score'].mean()

print("Average score for each subject:")
for subject, avg_score in subject_averages.items():
    print(f"{subject}: {avg_score:.2f}")

print("\nSubject averages (sorted by average score):")
subject_averages_sorted = subject_averages.sort_values(ascending=False)
for subject, avg_score in subject_averages_sorted.items():
    print(f"{subject}: {avg_score:.2f}")

print("\n" + "="*60 + "\n")

def pandas_filter_pass(dataframe):
    """
    Filter dataframe to return only records with grades A or B.
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe with a 'grade' column
    
    Returns:
    pd.DataFrame: Filtered dataframe containing only A and B grades
    """
    return dataframe[dataframe['grade'].isin(['A', 'B'])]

df_pass = pandas_filter_pass(df)

print("Students with passing grades (A or B):")
print(df_pass)

print(f"\nNumber of students with A or B grades: {len(df_pass)}")
print(f"Percentage of students with A or B grades: {len(df_pass)/len(df)*100:.1f}%")