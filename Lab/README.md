# 🧪 Hướng Dẫn Chạy Lab

## Môi trường chạy

### Google Colab (Khuyên dùng)
1. Mở file `.ipynb` trong thư mục `src/`
2. Upload lên [colab.research.google.com](https://colab.research.google.com)
3. Vào **Runtime → Change runtime type → GPU (T4)**
4. Chạy từng cell

### Cài đặt local (tuỳ chọn)
```bash
pip install cupy-cuda12x numpy matplotlib
```

## 📋 Danh sách bài tập

| Bài | Tên | Công nghệ | Trạng thái |
|---|---|---|---|
| HW01 | GPU-Accelerated Image Statistics with CuPy | CuPy | 🔄 |

## 📌 Ghi chú
- File `.ipynb` chạy trên **Google Colab với GPU**
- Thư mục `data/` chứa ảnh/dữ liệu mẫu cho từng bài
