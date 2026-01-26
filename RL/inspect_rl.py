import csv

# Clean episode_summary.csv
input_file = "sclerostin_biosensor_results_None/rl_logs/episode_summary.csv"
output_file = "sclerostin_biosensor_results_None/rl_logs/episode_summary_cleaned.csv"

print("🧹 Cleaning episode_summary.csv...")

with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        # Read header
        header = infile.readline().strip()
        expected_cols = len(header.split(','))
        print(f"Expected columns: {expected_cols}")
        
        outfile.write(header + '\n')
        
        cleaned_lines = 0
        skipped_lines = 0
        
        for line_num, line in enumerate(infile, start=2):
            # Count columns in this line
            cols = len(line.strip().split(','))
            
            if cols == expected_cols:
                outfile.write(line)
                cleaned_lines += 1
            else:
                print(f"   ⚠️ Skipping line {line_num}: has {cols} columns instead of {expected_cols}")
                skipped_lines += 1

print(f"\n✅ Cleaning complete!")
print(f"   Cleaned lines: {cleaned_lines}")
print(f"   Skipped lines: {skipped_lines}")
print(f"   Output: {output_file}")

# Rename cleaned file
import os
if skipped_lines > 0:
    backup = input_file + ".backup"
    os.rename(input_file, backup)
    os.rename(output_file, input_file)
    print(f"\n✅ Replaced original file (backup: {backup})")