import pyarrow.parquet as pq

parquet_file = pq.ParquetFile("data/koi_lightcurves.parquet")

columns = parquet_file.schema.names
num_cols = len(columns)
num_rows = parquet_file.metadata.num_rows

print("=== HEADER (Tên cột) ===")
print(columns)

print("\n=== Thống kê ===")
print("Số cột (tham số):", num_cols)
print("Số hàng (thiên thể):", num_rows)
print("Tổng số ô dữ liệu (hàng x cột):", num_rows * num_cols)
