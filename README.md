# PRIMO-Pre-interview-fulltime

## เทคนิคที่ใช้ในแต่ละส่วน

1. **สคริปต์สร้างข้อมูล (`Scripts/generate_data.py`)**
   - ใช้ `numpy.random.default_rng` เพื่อสุ่มข้อมูลตามสัดส่วน (probability weights) ของสาขา แบรนด์ และประเภทสินค้า ที่ได้จากการสำรวจในโน้ตบุ๊ก `explore.ipynb` จึงคง distribution ให้ใกล้เคียงข้อมูลจริง
   - สร้าง SKU catalog ล่วงหน้าแล้วสุ่มแบบ vectorized เพื่อรันได้เร็ว พร้อมมีการ drift ประเภทสินค้าเล็กน้อยเพื่อเพิ่มความแปรผันตามฤดูกาล
   - สร้างข้อมูลเป็นก้อน (chunk) ขนาด 500k แถว แล้ว downcast คอลัมน์ (เช่น `amount` -> `int32`, `quantity` -> `int8`, ค่าเชิงหมวดหมู่ -> `Categorical`) ก่อนเขียนลง Parquet
   - ใช้ `pyarrow.ParquetWriter` สตรีมข้อมูลลงไฟล์ `large_order_data_2024/large_order_data_10M_2024.parquet` พร้อมบีบอัดแบบ `snappy` และแสดงสถานะผ่าน `tqdm`

2. **ชุดข้อมูล 10 ล้านแถว (`large_order_data_2024/large_order_data_10M_2024.parquet`)**
   - ได้รับการจัดเก็บแบบ columnar ทำให้การอ่านเฉพาะคอลัมน์หรือกรองข้อมูลในภายหลังทำได้รวดเร็ว
   - ขนาดไฟล์เล็กลงเพราะใช้ทั้งการบีบอัดระดับไฟล์และการเข้ารหัสแบบ dictionary (จาก dtype หมวดหมู่) สอดคล้องกับผลการ explore ที่พบว่าค่ามีการซ้ำสูง

3. **สคริปต์สกัด Top Spender (`Scripts/find_top_spenders.py`)**
   - ใช้ `pyarrow.parquet.ParquetFile.iter_batches` ในการอ่านไฟล์ Parquet ทีละ batch (ควบคุมด้วย `CHUNK_SIZE`) เพื่อลดการใช้หน่วยความจำเมื่อทำงานกับ 10M แถว
   - หลังอ่านแต่ละ batch แปลงเป็น DataFrame แล้วใช้ `pandas.groupby` รวมยอดใช้จ่ายรายเดือนต่อ `customer_no`, จากนั้นรวมผลทุก batch และจัดอันดับด้วย `rank(method="first")`
   - ผลลัพธ์ถูก downcast (`rank` -> `int16`, `month` -> `int8`) ก่อนเขียนกลับเป็น Parquet (`large_order_data_2024/top_10_spenders_per_month.parquet`) เพื่อให้ไฟล์ผลลัพธ์มีขนาดเล็ก
   - มี progress bar ของ `tqdm` คอยบอกสถานะการประมวลผลแต่ละ batch

4. **ไฟล์ผลลัพธ์ Top Spender (`large_order_data_2024/top_10_spenders_per_month.parquet`)**
   - บันทึกข้อมูลที่ผ่านการจัดอันดับแล้ว โดยมีคอลัมน์ `year_month` (string), `month`, `month_name`, `rank`, `customer_no`, `total_spent`
   - โครงสร้างข้อมูลเอื้อต่อการนำไปทำ visualization หรือวิเคราะห์ต่อ เช่น pivot สร้าง heatmap หรือสรุปตามปี/เดือน

## วิธีใช้งาน

1. ติดตั้งไลบรารีที่จำเป็น:
   ```bash
   pip install -r requirements.txt  # ต้องมี numpy, pandas, pyarrow, tqdm, gdown
   ```
2. สร้างชุดข้อมูล 10 ล้านแถว:
   ```bash
   python Scripts/generate_data.py
   ```
3. คำนวณ Top Spender รายเดือน:
   ```bash
   python Scripts/find_top_spenders.py
   ```
4. เปิดดูผลลัพธ์ด้วย pandas หรือโปรแกรมที่รองรับไฟล์ Parquet ได้ตามต้องการ

**หรือหากต้องการใช้ Data เลยก็สามารถ run scripts download.py ได้ ใน folder large_order_data_2024**


> หมายเหตุ: การกระจายของแบรนด์ สาขา ประเภทสินค้า และช่วงราคา ถูกตั้งค่าตามข้อมูลเชิงลึกที่ได้จาก `explore.ipynb` เพื่อให้โมเดลการจำลองสอดคล้องกับข้อมูลจริง
