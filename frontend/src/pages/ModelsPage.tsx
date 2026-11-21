import { isAxiosError } from "axios";
import { useCallback, useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, errorMessage } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Loading } from "../components/Loading";
import { DataTable, EmptyState, ErrorState, PageHeader } from "../components/UI";
import { useReadRequest } from "../hooks/useReadRequest";
import type { ModelInfo, ServingModel } from "../types";

const names: Record<string, string> = {
  DecisionTree: "Decision Tree",
  LinearSVM: "Linear SVM",
  ANN: "Artificial Neural Network (ANN)",
};
const metrics = {
  ap: "Average Precision", roc_auc: "ROC-AUC", accuracy: "Accuracy",
  precision: "Precision", recall: "Recall", f1: "F1-score",
} as const;

interface TrainingStatus {
  status: "idle" | "running" | "failed" | "succeeded";
  enabled: boolean;
  version: string | null;
  message: string | null;
  completed_folds: number;
}

interface Curves {
  version: string;
  pr: { recall: number[]; precision: number[] };
  roc: { fpr: number[]; tpr: number[] };
}

function EvaluationCurve({ title, xLabel, yLabel, xValues, yValues }: {
  title: string; xLabel: string; yLabel: string; xValues: number[]; yValues: number[];
}) {
  const points = xValues.map((x, index) => ({ x, y: yValues[index] }));
  return (
    <figure>
      <figcaption className="mb-2 font-semibold">{title}</figcaption>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={points} margin={{ bottom: 24, right: 18 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} label={{ value: xLabel, position: "insideBottom", offset: -14 }} />
          <YAxis type="number" domain={[0, 1]} label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
          <Tooltip labelFormatter={(value) => Number(value).toFixed(4)} formatter={(value) => [Number(value).toFixed(4), yLabel]} />
          <Line type="linear" dataKey="y" dot={false} stroke="#2563eb" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </figure>
  );
}

export function ModelsPage() {
  const { user } = useAuth();
  const { data, loading, error, run } = useReadRequest<{
    current: ServingModel | null; models: ModelInfo[]; curves: Curves | null;
  }>();
  const [job, setJob] = useState<TrainingStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [jobError, setJobError] = useState("");
  const [starting, setStarting] = useState(false);
  const lastActivated = useRef<string | null>(null);
  const load = useCallback(() => run(async (signal) => {
    const [current, models] = await Promise.all([
      api.get<ServingModel>("/api/models/current", { signal }).then((res) => res.data).catch((err) => {
        if (isAxiosError(err) && err.response?.status === 503) return null;
        throw err;
      }),
      api.get<ModelInfo[]>("/api/models", { signal }),
    ]);
    const curves = current ? (await api.get<Curves>("/api/models/current/curves", { signal })).data : null;
    return { current, models: models.data, curves: curves?.version === current?.version ? curves : null };
  }), [run]);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const { data: status } = await api.get<TrainingStatus>("/api/models/training", { signal: controller.signal });
        if (controller.signal.aborted) return;
        setJob(status);
        setStatusError("");
        if (status.status === "succeeded" && status.version !== lastActivated.current) {
          lastActivated.current = status.version;
          void load();
        }
      } catch (err) {
        if (!controller.signal.aborted) setStatusError(errorMessage(err));
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(poll, 3000);
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [load]);

  async function train() {
    setStarting(true);
    setJobError("");
    try {
      setJob((await api.post<TrainingStatus>("/api/models/training")).data);
    } catch (err) {
      setJobError(errorMessage(err));
    } finally {
      setStarting(false);
    }
  }

  const current = data?.current;
  const test = current?.test_metrics;

  return (
    <div className="space-y-6">
      <PageHeader title="Thông tin mô hình" description="Huấn luyện, so sánh 5 fold và kết quả Test độc lập." />
      <section className="card space-y-3">
        <h2>Huấn luyện Home Credit</h2>
        <p>{Object.values(names).join(" · ")}</p>
        {job && <p role="status">
          {job.status === "idle" ? "Chưa có lượt huấn luyện trong phiên dịch vụ." : job.message}
          {job.status === "running" && ` · Đã đánh giá ${job.completed_folds} fold trên các cấu hình`}
        </p>}
        {(jobError || statusError) && <ErrorState message={jobError || statusError} />}
        {user?.role === "ADMIN" && (
          job?.enabled
            ? <button className="btn-primary" disabled={starting || job.status === "running"} onClick={() => void train()}>Bắt đầu huấn luyện</button>
            : job && <p className="text-sm text-muted">Quản trị viên có thể huấn luyện bằng CLI hoặc cấu hình huấn luyện qua API.</p>
        )}
      </section>
      {loading ? <Loading label="Đang tải thông tin mô hình…" /> : error ? <ErrorState message={error} retry={load} /> : (
        <>
          {current && test ? (
            <section className="card space-y-5">
              <h2>Mô hình đang phục vụ · {current.version}</h2>
              <p>Mô hình được chọn <strong>{names[current.algorithm] ?? current.algorithm}</strong> theo AP trung bình cao nhất trên 5 fold của Training.</p>
              <p>{current.training_dataset} · Ngưỡng phân loại {current.classification_threshold} · Xác suất {current.probability_calibrated ? "đã hiệu chỉnh trên Training" : "chưa hiệu chỉnh"}.</p>
              <h3>Cross-Validation trên Training</h3>
              <DataTable label="Kết quả 5-fold Cross-Validation">
                <table className="table">
                  <thead><tr>
                    <th scope="col">Mô hình</th>
                    {[1, 2, 3, 4, 5].map((fold) => <th key={fold} scope="col">Fold {fold}</th>)}
                    <th scope="col">Mean AP</th><th scope="col">Std AP</th><th scope="col">Thời gian fit (s)</th>
                  </tr></thead>
                  <tbody>
                    {Object.entries(current.cv_results.models).map(([name, result]) => (
                      <tr key={name}>
                        <th scope="row" className="normal-case">{names[name] ?? name}</th>
                        {result.folds.map((fold) => <td key={fold.fold}>{fold.ap.toFixed(6)}</td>)}
                        <td>{result.summary.ap_mean.toFixed(6)}</td>
                        <td>{result.summary.ap_std.toFixed(6)}</td>
                        <td>{result.total_fit_seconds.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </DataTable>
              <h3>Test độc lập · {test.rows.toLocaleString("vi-VN")} hồ sơ</h3>
              <dl className="detail-grid">
                {Object.entries(metrics).map(([key, label]) => <div key={key}>
                  <dt>{label}</dt><dd>{test[key as keyof typeof metrics].toFixed(6)}</dd>
                </div>)}
              </dl>
              {test.recall === 0 && <p className="info-note">
                Ở ngưỡng {current.classification_threshold}, mô hình bỏ sót toàn bộ hồ sơ có TARGET=1 trong tập Test.
              </p>}
              <DataTable label="Ma trận nhầm lẫn trên Test">
                <table className="table">
                  <caption>Ma trận nhầm lẫn</caption>
                  <thead><tr><th scope="col">Nhãn thực tế</th><th scope="col">Dự đoán 0</th><th scope="col">Dự đoán 1</th></tr></thead>
                  <tbody>{test.confusion_matrix.map((row, label) => <tr key={label}>
                    <th scope="row">{label}</th><td>{row[0]}</td><td>{row[1]}</td>
                  </tr>)}</tbody>
                </table>
              </DataTable>
              {data.curves && <div className="grid gap-4 lg:grid-cols-2">
                <EvaluationCurve title="PR Curve trên Test" xLabel="Recall" yLabel="Precision" xValues={data.curves.pr.recall} yValues={data.curves.pr.precision} />
                <EvaluationCurve title="ROC Curve trên Test" xLabel="False positive rate" yLabel="True positive rate" xValues={data.curves.roc.fpr} yValues={data.curves.roc.tpr} />
              </div>}
            </section>
          ) : <EmptyState description="Chưa có phiên bản Home Credit sẵn sàng dự đoán." />}
          <section className="card space-y-4">
            <h2>Registry lịch sử</h2>
            {!data?.models.length ? <EmptyState description="Chưa có phiên bản trong registry." /> : (
              <DataTable label="Registry mô hình lịch sử">
                <table className="table">
                  <thead><tr>
                    <th scope="col">Tên / Tập dữ liệu</th><th scope="col">Phiên bản</th>
                    <th scope="col">Thuật toán</th><th scope="col">Trạng thái đã lưu</th>
                  </tr></thead>
                  <tbody>{data.models.map((model) => <tr key={model.id}>
                    <td><p className="font-medium">{model.model_name}</p><p className="text-xs text-muted">{model.training_dataset}</p></td>
                    <td>{model.version}</td><td>{model.algorithm}</td><td><span className="badge">{model.status}</span></td>
                  </tr>)}</tbody>
                </table>
              </DataTable>
            )}
          </section>
        </>
      )}
    </div>
  );
}
