import sqlite3

conn = sqlite3.connect('files.db')
c = conn.cursor()

# Bảng lưu file
c.execute('''
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    owner_email TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_type TEXT,
    k_value INTEGER,
    size_before INTEGER,
    size_after INTEGER
)
''')

# Bảng lưu người được cấp quyền truy cập file
c.execute('''
CREATE TABLE IF NOT EXISTS file_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    user_email TEXT,
    FOREIGN KEY(file_id) REFERENCES files(id)
)
''')

conn.commit()
conn.close()
print("✅ Database và bảng 'files' + 'file_access' đã được tạo thành công.")
