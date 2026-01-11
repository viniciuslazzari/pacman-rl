import pandas as pd

# Load CSV
df = pd.read_csv("comparison_results/experiments_summary.csv")

# Print as Markdown table
def df_to_markdown(df):
    # Header
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in df.values]
    return "\n".join([header, separator] + rows)

markdown_table = df_to_markdown(df)
print(markdown_table)
