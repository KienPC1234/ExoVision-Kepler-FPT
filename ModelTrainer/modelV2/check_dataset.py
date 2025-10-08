import pyarrow.parquet as pq

parquet_file = pq.ParquetFile("data/koi_lightcurves.parquet")

columns = parquet_file.schema.names
num_cols = len(columns)
num_rows = parquet_file.metadata.num_rows

print("=== HEADER (Column Name) ===")
print(columns)

print("\n=== Statistics ===")
print("Number of columns (parameters):", num_cols)
print("Number of rows (celestial):", num_rows)
print("Total number of data cells (rows x columns):", num_rows * num_cols)
