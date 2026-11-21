# Hệ thống hỗ trợ đánh giá rủi ro tín dụng và dự đoán khả năng trả nợ bằng máy học

**Đề tài Niên luận Kỹ thuật phần mềm**  
Hệ thống **hỗ trợ** cán bộ tín dụng ước lượng rủi ro gặp khó khăn thanh toán theo Home Credit bằng Machine Learning, **không** thay thế quyết định cấp tín dụng thực tế và **không** kết nối ngân hàng thật. Hồ sơ và giải thích SHAP mô phỏng cũ được giữ để xem lịch sử.

> **Cảnh báo bắt buộc:** Kết quả chỉ mang tính hỗ trợ đánh giá rủi ro, không thay thế quyết định cấp tín dụng thực tế.  
> **Nguồn dữ liệu:** Bản huấn luyện `home-credit-dt-svm-ann-20261121` dùng 307.511 hồ sơ gốc từ Home Credit Default Risk. Xem [báo cáo huấn luyện lại](docs/ml_retraining_report.md).

Sau khi chạy hệ thống, vào **Khách hàng**, chọn hoặc tạo hồ sơ, rồi bấm **Dự đoán và lưu kết quả**.
Hồ sơ nhập từ CSV dùng mã ẩn danh. TARGET gốc hiển thị riêng và nằm ngoài đầu vào dự đoán.

---

## 1. Kiến trúc hệ thống

```
┌───────────────────────────┐      JWT      ┌─────────────────────────────┐     SQL     ┌────────────┐
│         Frontend          │ ────────────► │           Backend           │ ──────────► │ PostgreSQL │
│ (React + Vite + Tailwind) │               │      (FastAPI + JWT + RBAC) │             └────────────┘
└───────────────────────────┘               └──────────────┬──────────────┘
                                                           │ HTTP /predict
                                                           ▼
                                            ┌─────────────────────────────┐
                                            │         ML Service          │
                                            │ (FastAPI + scikit-learn)    │
                                            └─────────────────────────────┘
```

- **Frontend:** Không gọi trực tiếp ML Service; toàn bộ giao tiếp đi qua Backend.
- **Backend:** Quản lý người dùng, phân quyền RBAC, khách hàng, hồ sơ vay, lưu lịch sử dự đoán và nhật ký kiểm toán (audit log).
- **ML Service:** So sánh Decision Tree, Linear SVM và ANN. Nạp pipeline Joblib để ước lượng xác suất gặp khó khăn thanh toán (`TARGET=1`).

---

## 2. Công nghệ sử dụng

| Phân hệ | Công nghệ |
|---|---|
| **Frontend** | React 18, Vite, TypeScript, TailwindCSS, Axios, Recharts |
| **Backend** | Python 3.11, FastAPI, SQLAlchemy 2.0, Pydantic v2, JWT, Bcrypt |
| **ML Service** | FastAPI, scikit-learn, pandas, NumPy, Joblib |
| **Database** | PostgreSQL 16 (hỗ trợ SQLite cho demo chạy nhẹ) |
| **Container & CI** | Docker, Docker Compose |
| **Testing** | Pytest (Backend & ML Service) |

---

## 3. Danh sách 10 trang giao diện người dùng

1. **Đăng nhập (`/login`):** Xác thực JWT, ghi nhớ phiên làm việc, phân quyền theo vai trò.
2. **Dashboard (`/`):** Thống kê tổng số hồ sơ, số hồ sơ đã đánh giá và tổng lượt dự đoán.
3. **Danh sách khách hàng (`/customers`):** Tìm kiếm, phân trang, thêm mới và xem chi tiết khách hàng.
4. **Danh sách hồ sơ vay (`/loans`):** Tìm kiếm, lọc theo trạng thái dự đoán và mở hồ sơ.
5. **Chi tiết hồ sơ vay (`/loans/:id`):** Xem thông tin Home Credit, kết quả và các lần dự đoán của hồ sơ.
6. **Tạo / Chỉnh sửa hồ sơ vay (`/loans/new`, `/loans/:id/edit`):** Form validation đầy đủ với Pydantic backend.
7. **Đánh giá rủi ro (`/risk-assessment`, `/loans/:id/result`):** Chọn hồ sơ hoặc nhập thông tin Home Credit để nhận xác suất, lớp dự đoán và điểm minh họa.
8. **Lịch sử dự đoán (`/predictions`):** Tìm các lượt đánh giá đã lưu theo hồ sơ và xem lại phiên bản mô hình, đầu vào và kết quả.
9. **Quản lý người dùng (`/users`):** Dành riêng cho `ADMIN` tạo, phân quyền và khóa tài khoản cán bộ.
10. **Thông tin mô hình (`/models`):** Xem tiến độ huấn luyện, AP từng fold, mô hình đang phục vụ, kết quả Test, biểu đồ PR/ROC và registry lịch sử.

---

## 4. Tài khoản demo

Hệ thống chỉ có hai vai trò hoạt động: `ADMIN` và `OFFICER`. Khách hàng là hồ sơ nghiệp vụ, không cần tài khoản đăng nhập. Tài khoản Viewer cũ đã ngừng sử dụng; chi tiết migration và kiểm tra tại [docs/two_roles.md](docs/two_roles.md).

Trang đăng nhập có nút điền nhanh cho hai tài khoản Admin và Officer `.com` dưới đây. Nút chỉ điền thông tin; bấm **Đăng nhập** để xác thực. Nếu mật khẩu đã được đổi, nhập mật khẩu hiện tại.

Backend tạo tài khoản khi chưa có. Với tài khoản demo cũ dùng đuôi `.local`, bước seed chỉ đổi email sang `.com`, giữ ID, vai trò và mật khẩu đã băm. Nếu cả hai địa chỉ cùng tồn tại, dùng `.com` và giữ nguyên cả hai; nếu tài khoản được chọn sai vai trò, seed báo xung đột để kiểm tra. Không cần xóa database hay Docker volume. `ADMIN_EMAIL` vẫn hỗ trợ địa chỉ quản trị tùy chỉnh; cấu hình mặc định dùng `admin@credit-risk.com`.

| Vai trò | Email | Mật khẩu | Quyền hạn |
|---|---|---|---|
| **ADMIN** | `admin@credit-risk.com` | `Admin@123456` | Quản trị người dùng, mô hình, toàn bộ dữ liệu |
| **OFFICER** | `officer@credit-risk.com` | `Officer@123456` | Quản lý khách hàng, tạo hồ sơ vay, chạy đánh giá ML |

---

## 5. Hướng dẫn chạy hệ thống

### Cách 1. Chạy bằng Docker Compose

```bash
# Chạy tại gốc repo. Cấu hình bổ sung có trong .env.example.
docker compose up --build
```

Để phục vụ một bản đã kích hoạt, đưa `active_model.json` và thư mục `versions/<version>/` vào volume
`ml_artifacts`. Dịch vụ báo `model_ready: false` khi chưa có artifact hợp lệ. Để huấn luyện từ giao
diện trong Docker, cấu hình `HOME_CREDIT_DATA_DIR` trên host và khóa chia sẻ `ML_TRAINING_TOKEN`,
sau đó chạy `docker compose -f docker-compose.yml -f docker-compose.home-credit.yml up --build`.

**Các cổng truy cập:**
- **Frontend App:** [http://localhost:3000](http://localhost:3000)
- **Backend API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ML Service Docs (Swagger):** [http://localhost:8001/docs](http://localhost:8001/docs)

---

### Cách 2: Chạy Local độc lập (dành cho phát triển & kiểm thử)

Tạo môi trường và chuẩn bị artifact theo mục 7. Chạy từng dịch vụ trong một terminal, bắt đầu từ gốc repo.

#### 1. ML Service
```powershell
cd ml-service
& ../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8001
```

#### 2. Backend
```powershell
cd backend
# Mặc định sử dụng SQLite tại ./credit_risk_demo.db
& ../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

#### 3. Frontend
```bash
cd frontend
npm ci
npm run dev
# Mở trình duyệt tại http://localhost:5173
```

---

## 6. Chạy kiểm thử tự động (Pytest)

```bash
# Kiểm thử Backend (Auth, RBAC, Customer, Loan, Risk scoring)
cd backend
python -m pytest

# Kiểm thử pipeline, artifact và API. Đặt HOME_CREDIT_DATA_DIR để chạy kiểm thử với CSV thật.
cd ../ml-service
python -m pytest
```

---

## 7. Huấn luyện và phục vụ Home Credit

Từ `application_train.csv` có nhãn, chia phân tầng 80% Training và 20% Test với
`random_state=42`. Cả ba thuật toán dùng cùng 5 fold Training, mỗi lượt fit lại
median imputation, mã hóa one-hot và bước chuẩn hóa dành cho SVM/ANN.
Pipeline chuyển mã ngày làm việc `365243` thành giá trị thiếu và xử lý danh mục mới.

- Decision Tree thử hai cấu hình độ sâu và kích thước nút với trọng số cân bằng lớp.
- Linear SVM dùng `LinearSVC` cho dữ liệu lớn, thử hai giá trị `C` và tính AP từ `decision_function`.
- ANN dùng `MLPClassifier` để học quan hệ phi tuyến, thử hai kiến trúc ReLU/Adam có regularization và giới hạn số epoch.

Average Precision (AP) tổng hợp precision qua các mức recall và phù hợp với phép so sánh
khả năng xếp hạng trên dữ liệu mất cân bằng. Chọn cấu hình và thuật toán có mean AP cao nhất.
Fit lại pipeline đã chọn trên đủ Training, hiệu chỉnh sigmoid từ dự đoán out-of-fold trên
Training, rồi đánh giá một lần trên Test với ngưỡng định trước 0,5. Báo cáo gồm AP, ROC-AUC,
accuracy, precision, recall, F1, ma trận nhầm lẫn và tọa độ đường PR/ROC.

```powershell
# Từ gốc repo.
python -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r ml-service/requirements.txt -r backend/requirements.txt
$env:HOME_CREDIT_DATA_DIR = 'D:/path/to/home-credit-default-risk'
& ./.venv/Scripts/python.exe ml-service/training/train.py --mode prepare --report reports/ml-audit.json
& ./.venv/Scripts/python.exe ml-service/training/train.py --mode cv-smoke --sample-size 1000
& ./.venv/Scripts/python.exe ml-service/training/train.py --mode train --version home-credit-repro --report reports/ml-repro.json
Push-Location ml-service
& ../.venv/Scripts/python.exe -m training.activate --version home-credit-repro
Pop-Location
```

Lệnh tự tìm CSV trong thư mục `home-credit-default-risk` bên trong hoặc cạnh repo;
có thể truyền `--application-csv`, `--description-csv`, `--data-dir` hoặc biến môi trường.
`--mode prepare` kiểm tra dữ liệu và shape tiền xử lý. `--mode cv-smoke` dùng một mẫu
lấy từ Training thật. `--mode cv` so sánh CV trên toàn bộ Training. `--mode train` chạy đầy đủ
với tên phiên bản mới. Mặc định artifact cục bộ nằm trong `reports/ml-artifacts/versions/<version>/`.
Dùng cùng `ARTIFACT_DIR` cho huấn luyện, kích hoạt và phục vụ nếu đổi đường dẫn.
Lệnh kích hoạt xác minh pipeline và dự đoán qua API trước khi cập nhật manifest.

Để huấn luyện trong `/models`, đặt cùng một giá trị `ML_TRAINING_TOKEN` cho backend và ML
Service, cung cấp đường dẫn dữ liệu cho ML Service và dùng tài khoản `ADMIN`.
Trang hiển thị tiến độ trong phiên dịch vụ, và phiên bản mới tự được nạp sau khi xác minh.

Mô hình chỉ nhận sáu trường trong hợp đồng giao diện. Số tiền dùng thang Home Credit gốc.
Kết quả thực nghiệm ở ngưỡng 0,5 có recall bằng 0. Các nhóm thấp/trung bình/cao cần được
xác nhận riêng. Xem [báo cáo huấn luyện lại](docs/ml_retraining_report.md).

---

## 8. Kịch bản Demo Niên luận (5 phút)

1. **Đăng nhập:** Mở `http://localhost:3000`, đăng nhập bằng `officer@credit-risk.com` / `Officer@123456`.
2. **Dashboard:** Xem số hồ sơ và số kết quả đã lưu.
3. **Quản lý khách hàng:** Vào menu **Khách hàng**, bấm xem hoặc tạo mới khách hàng mẫu.
4. **Hồ sơ vay:** Vào menu **Hồ sơ vay** để xem thông tin và trạng thái dự đoán.
5. **Dự đoán theo khách hàng:** Vào Khách hàng hoặc Hồ sơ và dự đoán, chọn hồ sơ Home Credit,
   kiểm tra 6 đầu vào, bấm Dự đoán và lưu kết quả. Mỗi lần lưu gắn đúng hồ sơ, snapshot đầu
   vào và phiên bản mô hình; mở Lịch sử dự đoán để xem lại. Không mặc định số tiền là VND.
6. **Lịch sử dự đoán:** Vào menu **Lịch sử dự đoán**, tìm theo hồ sơ và mở chi tiết kết quả.
7. **Phân quyền:** Đăng xuất và đăng nhập tài khoản `admin@credit-risk.com` / `Admin@123456` để kiểm tra trang **Người dùng** và **Mô hình**.
