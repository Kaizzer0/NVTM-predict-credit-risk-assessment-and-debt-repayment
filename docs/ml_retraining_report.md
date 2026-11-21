# Báo cáo huấn luyện lại Home Credit

Phiên bản `home-credit-dt-svm-ann-20261121`, ngày 21/11/2026. Pipeline phục vụ là Artificial Neural Network (ANN).

## Dữ liệu và hợp đồng đầu vào

Nguồn huấn luyện là `Home Credit Default Risk/application_train.csv`, gồm 307.511 hồ sơ và 122 cột. Pandas đọc 65 cột `float64`, 41 cột `int64` và 16 cột phân loại `object`. Có 9.152.465 giá trị thiếu trên 67 cột, 0 giá trị số vô hạn, 0 ID thiếu hoặc trùng. `TARGET=1` biểu thị trễ hơn X ngày ở ít nhất một trong Y kỳ đầu. Home Credit không công bố X và Y.

| Tập | Hồ sơ | TARGET=0 | TARGET=1 |
|---|---:|---:|---:|
| Gốc | 307.511 | 282.686 (91,9271%) | 24.825 (8,0729%) |
| Training | 246.008 | 226.148 | 19.860 |
| Test nội bộ | 61.503 | 56.538 | 4.965 |

| Đặc trưng | Kiểu | Thiếu gốc | Xử lý |
|---|---|---:|---|
| `AMT_CREDIT` | Số | 0 | Giữ thang tiền Home Credit |
| `AMT_ANNUITY` | Số | 12 | Median học từ phần fit |
| `DAYS_BIRTH` | Số | 0 | Ngày âm so với ngày nộp hồ sơ |
| `DAYS_EMPLOYED` | Số | 0 | Chuyển 55.374 mã `365243` thành thiếu rồi điền median |
| `NAME_CONTRACT_TYPE` | Phân loại | 0 | One-hot |
| `OCCUPATION_TYPE` | Phân loại | 96.391 | Mã riêng cho thiếu, one-hot, bỏ qua danh mục mới |

Sáu trường trên lấy từ biểu mẫu đang dùng. `TARGET` chỉ dùng làm nhãn, `SK_ID_CURR` dùng kiểm tra định danh và phân vùng. Các cột về hành vi trả nợ sau thời điểm nộp hồ sơ có nguy cơ mang thông tin kết quả nếu ghép thiếu kiểm soát thời gian. Bản này chỉ dùng `application_train.csv`. Các bảng `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `POS_CASH_balance` và `credit_card_balance` chỉ phù hợp khi có quy trình xác minh thông tin sẵn có tại ngày nộp và ghép đúng thời điểm. CSV Kaggle `application_test.csv` không có nhãn.

## Thiết kế thực nghiệm

- Chia phân tầng một lần bằng `train_test_split(test_size=0.2, random_state=42)`. Mọi cấu hình dùng cùng chỉ mục từ `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` trên Training. Metadata lưu mã băm ID của tập fit và validation từng fold, cùng ID Training và Test ở dạng mã băm.
- Mỗi fold có `Pipeline` mới. Bước làm sạch dùng quy tắc cố định, còn imputer, scaler và encoder được fit trên bốn fold. SVM và ANN chuẩn hóa số, DT giữ thang gốc. `ColumnTransformer` ghép kết quả thành ma trận sparse.
- Mẫu chuẩn bị 256 hồ sơ lấy từ Training cho shape fit `192 × 22`, shape validation `64 × 22`, đầu ra hữu hạn. Kiểm tra nhỏ so sánh đủ 15 lượt fit từ 1.000 hồ sơ Training thật. Chỉ số ở bảng dưới đến từ quy trình toàn bộ dữ liệu.
- Dùng Python 3.12.14, scikit-learn 1.6.0, pandas 2.2.3 và NumPy 2.2.1 trên Windows. Máy có 16 CPU logic, 15,71 GiB RAM và khoảng 1,12 GiB RAM khả dụng ở thời điểm kiểm tra ban đầu. Dữ liệu huấn luyện chỉ đọc các cột cần thiết, lần lượt fit từng cấu hình và giới hạn thư viện tính toán ở hai luồng.

## Cross-Validation trên Training

AP là Average Precision của scikit-learn. Điểm xếp hạng CV lấy từ xác suất lớp 1 của DT/ANN và `decision_function` của Linear SVM. Mỗi hàng hiển thị năm fold của cấu hình tốt nhất của thuật toán. Thời gian là tổng thời gian fit của cả hai cấu hình.

| Mô hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean AP | Std AP | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0,125690 | 0,117831 | 0,119317 | 0,118629 | 0,118938 | 0,120081 | 0,002847 | 18,66 |
| Linear SVM | 0,128662 | 0,122144 | 0,128089 | 0,124376 | 0,122934 | 0,125241 | 0,002664 | 12,68 |
| ANN | 0,138236 | 0,135570 | 0,137956 | 0,137714 | 0,132666 | **0,136428** | 0,002105 | 128,11 |

| Thuật toán | Cấu hình 1 → mean AP | Cấu hình 2 → mean AP |
|---|---|---|
| Decision Tree | Depth 5, split 800, leaf 400 → 0,113173 | Depth 8, split 400, leaf 200 → 0,120081 |
| Linear SVM | `C=0.01` → 0,125228 | `C=1.0` → 0,125241 |
| ANN | `(16,)`, `alpha=0.001` → 0,134167 | `(24,12)`, `alpha=0.01` → 0,136428 |

DT và SVM dùng `class_weight="balanced"`. SVM dùng solver primal, `max_iter=2000` và `tol=1e-5`. ANN dùng ReLU, Adam, batch 256, `max_iter=60`, `tol=1e-4` và dừng theo loss Training khi hết số lượt chờ cải thiện (`n_iter_no_change=7`). Cấu hình này tránh dùng accuracy validation làm tiêu chí dừng nội bộ trên dữ liệu lệch lớp. Có 30 lượt fit CV trong 163,73 giây tổng thời gian. CV và refit đều ghi nhận 0 cảnh báo hội tụ.

## Mô hình được chọn và Test độc lập

ANN `(24,12)` đạt mean AP cao nhất. Lựa chọn và ngưỡng 0,5 được ghi vào `selection_lock.json` trước khi đánh giá. Refit dùng đủ 246.008 hồ sơ Training và mất 74,97 giây, kể cả hiệu chỉnh sigmoid từ ba fold phân tầng nội bộ. `CalibratedClassifierCV(ensemble=False)` lấy điểm out-of-fold để học hiệu chỉnh và fit pipeline cơ sở trên toàn bộ Training. Cả ba bước so sánh, hiệu chỉnh và fit cuối chỉ dùng Training.

| Chỉ số trên 61.503 hồ sơ Test | Giá trị |
|---|---:|
| Average Precision | 0,136888 |
| ROC-AUC | 0,645724 |
| Accuracy | 0,919272 |
| Precision | 0 |
| Recall | 0 |
| F1 | 0 |

| Nhãn thực tế | Dự đoán 0 | Dự đoán 1 |
|---|---:|---:|
| TARGET=0 | 56.538 | 0 |
| TARGET=1 | 4.965 | 0 |

Ở ngưỡng 0,5, mọi hồ sơ Test được dự đoán lớp 0. Accuracy phản ánh tỷ lệ lớp đa số ở ngưỡng này. AP và ROC-AUC đánh giá khả năng xếp hạng theo xác suất. Dữ liệu Test chỉ được chấm một lần cho ANN. Nhóm rủi ro nghiệp vụ thấp/trung bình/cao cần được xác nhận riêng.

## Artifact và tái lập

- `reports/ml-artifacts/versions/home-credit-dt-svm-ann-20261121/` chứa pipeline Joblib, metadata, khóa lựa chọn và toàn bộ tọa độ PR/ROC trong `test_curves.json`. Giao diện lấy tối đa 400 điểm mỗi đường từ API. Manifest đang kích hoạt nằm tại `reports/ml-artifacts/active_model.json`.
- `reports/ml-audit.json` chứa số thiếu và kiểu dữ liệu từng cột. `reports/ml-retraining.json` chứa chỉ số từng fold, thời gian, cấu hình và Test. CSV, artifact, bản ghi đầu vào mẫu và SQLite kiểm thử thuộc các đường dẫn Git bỏ qua.
- Đã tải lại artifact qua quy trình kích hoạt. Một hồ sơ Training trả HTTP 200 và xác suất 0,151259913. Đầu vào thiếu trường bắt buộc hoặc có chứa TARGET/ID trả 422.
- Cách huấn luyện, kích hoạt và khởi chạy có trong [README](../README.md).

## Kiểm thử tích hợp

| Phạm vi | Kết quả |
|---|---|
| Pytest ML Service | 29 đạt. Kiểm tra chia dữ liệu, fit trong fold, AP theo đúng loại score, refit, hiệu chỉnh, nạp artifact, 422, 503 và trạng thái lỗi của tác vụ huấn luyện. |
| Pytest Backend | 37 đạt, bao gồm cầu nối tới ML Service thật, phân quyền, tạo hồ sơ, dự đoán và đọc lịch sử. Một assertion gắn cứng phiên bản v1 đã được cập nhật để đối chiếu với manifest thực tế. |
| Playwright với API thật | 9 kiểm tra đạt. Đăng nhập, bảng CV đối chiếu từng ô AP với API, Test độc lập, hai biểu đồ, dự đoán, lưu và xem lịch sử trên SQLite cách ly. |
| Playwright hồi quy giao diện | 110 kiểm tra đạt trên Chromium và Vite, với API giả lập dành riêng cho kiểm thử giao diện. |
| Khởi động lại | Đã khởi động lại cả backend và ML Service rồi đọc bản ghi đã lưu và dự đoán lại cùng đầu vào. Phiên bản và xác suất khớp với kết quả đã kiểm tra. |
| Kiểm tra tĩnh | TypeScript type-check, Vite build, biên dịch cú pháp Python, kiểm tra cú pháp JavaScript và `pip check` đều đạt. |

Docker Compose/PostgreSQL chưa được chạy trong môi trường này. Máy có thư mục Docker Desktop nhưng shell không tìm được executable `docker`; WSL trả `E_ACCESSDENIED` khi liệt kê distro. Kiểm thử toàn luồng dùng SQLite hỗ trợ sẵn trong backend. Lượt huấn luyện toàn bộ được chạy qua CLI. API huấn luyện đã được kiểm tra quyền, nhận tác vụ, xung đột và lỗi thiếu dữ liệu; không kích hoạt lượt đánh giá Test toàn bộ thứ hai qua giao diện.
