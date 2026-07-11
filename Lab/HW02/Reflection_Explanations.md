# HW02 — Giải thích Reflection Questions

Ghi chú giải thích logic đằng sau các câu trả lời reflection questions trong `src/HW02.ipynb`, dựa trên code và số liệu đo thực tế trong notebook (K=15, TILE=16, HALF_K=7, SHARED_DIM=30, CY=2, GPU: Tesla T4).

## Câu 1 — Halo loading

**Lập luận:** K=15 → mỗi pixel output cần nhìn ra 7 pixel (`HALF_K=7`) về mỗi phía. Nếu block 16×16 threads chỉ load đúng 16×16 pixel khớp với vị trí thread, thì các thread gần **rìa tile** thiếu dữ liệu 7-pixel viền ngoài để hoàn thành tổng K×K của mình.

**Hậu quả nếu quên:** đọc phải ô `smem` chưa được ghi (rác) hoặc index sai → sai kết quả **lặp lại theo lưới** ở mọi ranh giới tile trên toàn ảnh, chứ không phải lỗi ngẫu nhiên rải rác.

## Câu 2 — `cuda.syncthreads()`

**Sau khi load (bắt buộc):** 256 thread trong block load các phần khác nhau của `smem[30,30]` song song, tốc độ không đều nhau → nếu không có barrier, thread nhanh có thể bắt đầu đọc `smem` để tính convolution trước khi thread chậm ghi xong phần của nó → race condition.

**Trong vòng lặp tính toán (không cần):** lúc này mọi thread chỉ **đọc** `smem` (đã sync xong, không đổi nữa) và ghi vào biến `val` riêng của mình rồi `out[row,col]` (ô global riêng biệt, không đụng nhau) → không còn phụ thuộc chéo giữa các thread nữa nên sync thêm chỉ tốn overhead vô ích.

## Câu 3 — Roofline

Dùng đúng số đo thực tế trong notebook (sau khi đã sửa lỗi thiếu `cuda.syncthreads()` ở Stage 4/5 — xem phần "Bug đã tìm và sửa" bên dưới):
- Ridge point = 25.43 FLOP/byte (tính từ `PEAK_TFLOPS*1000/PEAK_BW_GBS`)
- GFLOP/s đo được: Naive 38.9, Tiled 53.5, Const 62.7, Coarsened 38.3 — tất cả đều **dưới 1% của 8.14 TFLOPS peak**

**Kết luận:** dù AI lý thuyết của tiled/const (32 FLOP/byte) và coarsened (64 FLOP/byte) đã vượt ridge point trên giấy, hiệu năng đo được thực tế vẫn cực thấp so với cả hai trần (compute lẫn memory 320 GB/s) — nghĩa là kernel chưa hề chạm tới đường roofline, tức còn nhiều overhead khác (launch overhead, workload quá nhỏ để bão hòa GPU, coalescing chưa tối ưu) đang kìm hãm hiệu năng chứ không chỉ đơn thuần là bandwidth hay compute. Đây là điểm tinh tế: AI cao không tự động đảm bảo đạt gần roofline nếu kernel không tận dụng hết phần cứng.

## Câu 4 — Thread coarsening tradeoff

Số đo thực tế (sau khi sửa bug): **Coarsened CY=2 (12.32ms) giờ là kernel GPU chậm nhất** — chậm hơn cả Naive (12.14ms), Tiled (8.82ms), và Const (7.53ms). Lý do:
- Mỗi thread cần nhiều thanh ghi hơn để giữ `CY` accumulator
- Mỗi block cần shared memory lớn hơn (`SHARED_ROWS = CY*TILE + 2*HALF_K = 46` thay vì 30)
- Số block giảm theo tỉ lệ `1/CY`

→ Khái niệm phần cứng: **occupancy**. Ít block/warp đồng thời trên SM hơn nghĩa là khi 1 warp bị stall (chờ global memory), SM không có warp khác để switch sang che giấu độ trễ → hiệu năng giảm dù về lý thuyết AI đã tăng gấp đôi.

## Câu 5 — Kết nối với đồ án cuối kỳ (NMS)

Đồ án cuối kỳ: **Topic A4 — Real-Time NMS (Non-Maximum Suppression)** cho object detection.

Map trực tiếp: pixel ↔ box, vùng lân cận K×K ↔ nhóm box cần so sánh IoU, `smem[30,30]` load ảnh 1 lần ↔ load tọa độ (x1,y1,x2,y2,score) của 1 tile box vào shared memory 1 lần. Ý tưởng cốt lõi giống hệt Stage 3: **giảm số lần đọc global memory bằng cách cho nhiều thread dùng chung dữ liệu đã load sẵn trong shared memory**, thay vì mỗi thread tự đọc lại tọa độ box từ global memory cho từng cặp so sánh.

---

## Bug đã tìm và sửa (so với template gốc)

Đối chiếu `HW02-template.ipynb` với bản đã điền, phát hiện:

- **Stage 4 (`conv2d_const_mem`) và Stage 5 (`conv2d_coarsened`)** — cả hai đều **thiếu `cuda.syncthreads()`** sau vòng lặp load shared memory, trước khi đọc `smem` để tính convolution. Đây là race condition thật sự (undefined behavior), nằm trong phần code sinh viên được yêu cầu tự viết (`YOUR CODE HERE` trong template) → đã sửa bằng cách thêm `cuda.syncthreads()` đúng vị trí. Sau khi sửa, `assert np.allclose(...)` ở cả 2 stage vẫn pass, và số liệu ở Stage 7/8 đã được chạy lại (số liệu trong câu 3, câu 4 ở trên là số liệu **sau khi sửa**).
- **Stage 6 (`run_filter`)** — có lỗi `if K == 5` (lẽ ra phải là `if K == 15`, vì `conv2d_tiled_specialized` được build cứng cho K=15). Do không filter nào trong bài có K=5, nhánh dispatch cho Gaussian blur (K=15) không bao giờ chạy, khiến `d_out` (garbage/uninitialized GPU memory) được trả về mà không báo lỗi. Cell này **không có `YOUR CODE HERE`** trong template — là code "provided", theo luật "Do not modify the provided code cells" nên **không tự sửa**, chỉ ghi chú lại để báo giảng viên.

**Lưu ý:** HW02 được chấm tự động và yêu cầu "your genuine understanding" — nên đọc lại, diễn đạt bằng văn phong của bản thân trước khi nộp bài để đúng tinh thần academic integrity của khóa học.
