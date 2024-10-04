import pandas as pd

def process(input_files):
    for idx, input_file in enumerate(input_files, start=1):
            # Read the Excel file
            df = pd.read_excel(input_file)
            
            # Select only the third column (column C) and rename it to 'control'
            df_control = df.iloc[:, [2]].copy()  # iloc[:, 2] selects the third column
            df_control.columns = ['control']     # Rename it to 'control'
            
            # Create a unique output file name (test1.xlsx, test2.xlsx, etc.)
            output_file = f'test{idx}.xlsx'
            
            # Save the result to a new Excel file
            df_control.to_excel(output_file, index=False)