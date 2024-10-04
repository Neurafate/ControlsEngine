import time
import pandas as pd
from preprocess import process
from classifier import process_files
from classified_to_dict import run_comparison
from results import merge_results_with_framework1

frame1=str(input("Enter the path of user org framework"))
frame2=str(input("Enter the path of the service org framework"))
frameworks=[frame1,frame2]
process(frameworks)

file1='test1.xlsx'
file2='test2.xlsx'

time.sleep(10)
classifiedfiles=process_files([file1,file2])

sheet1=classifiedfiles[0]
sheet2=classifiedfiles[1]
run_comparison(sheet1_path=sheet1,sheet2_path=sheet2)

df = pd.read_excel(frame1)
csv_file_path = 'original.csv'  # Replace with the desired output file path
df.to_csv(csv_file_path, index=False)

original='original.csv'
comparison='control_comparisons.csv'
output_path = 'framework1_with_results.csv'
merge_results_with_framework1(original, comparison, output_path)