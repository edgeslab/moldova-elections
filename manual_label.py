import pandas as pd

# Define the 8 label columns
label_columns = [
    "pro-Stoianoglo", "anti-Stoianoglo", "neutral-Stoianoglo",
    "pro-Sandu", "anti-Sandu", "neutral-Sandu",
    "non-relevant to Stoianoglo", "non-relevant to Sandu"
]

# Load the text file (assuming it's already in variable form or you can load from a file)
file_to_read = input("input which text file to read: ")
with open(file_to_read, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# Initialize empty dataframe
data = []

# For each line, prompt the user for binary input on each label
for idx, line in enumerate(lines):
    print(f"\nTweet # {idx+1}:\n{line}\n")
    row = {"translatedContentText": line}
    for label in label_columns:
        while True:
            val = input(f"{label} (0=No, 1=Yes): ").strip()
            if val in {"0", "1"}:
                row[label] = "Yes" if val == "1" else "No"
                break
            else:
                print("Please enter 0 or 1.")
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("labeled_stances_2.csv", index=False)

df.head()
