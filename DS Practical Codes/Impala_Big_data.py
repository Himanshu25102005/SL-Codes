# Generated from: Impala_Big_data.ipynb
# Converted at: 2026-04-23T03:19:42.037Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Title: Big Data Analytics (Impala)


# PROBLEM STATEMENT:
# Create databases and tables, insert small amounts of data, and run simple queries using 
# Impala


import sqlite3

conn = sqlite3.connect('student_db.db')
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE students (
    id INTEGER,
    name TEXT,
    age INTEGER
)
""")
conn.commit()

cursor.execute("INSERT INTO students VALUES (1, 'Laukik', 21)")
cursor.execute("INSERT INTO students VALUES (2, 'Nikhil', 22)")
cursor.execute("INSERT INTO students VALUES (3, 'Sakshi', 20)")
conn.commit()

cursor.execute("SELECT * FROM students")
rows = cursor.fetchall()

for row in rows:
    print(row)

cursor.execute("SELECT COUNT(*) FROM students")
print(cursor.fetchone())

conn.close()