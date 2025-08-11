import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";

// ---------- Config ----------
const PREDICTION_API = import.meta.env.VITE_PREDICTION_API || "http://localhost:8000";
const TRAINING_API   = import.meta.env.VITE_TRAINING_API   || "http://localhost:8001";


// ---------- Tiny UI primitives (Tailwind) ----------
const Card = ({ children, className = "" }) => (
  <div className={`rounded-2xl bg-white/70 dark:bg-zinc-900/60 backdrop-blur shadow-xl border border-zinc-200/50 dark:border-zinc-800 p-5 ${className}`}>{children}</div>
);

const SectionTitle = ({ title, desc, icon }) => (
  <div className="mb-4">
    <div className="flex items-center gap-2">
      <span className="text-xl">{icon}</span>
      <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
    </div>
    {desc && <p className="text-sm text-zinc-500 mt-1">{desc}</p>}
  </div>
);

const Button = ({ children, onClick, type = "button", disabled = false, variant = "primary", className = "" }) => {
  const base = "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 font-medium transition";
  const styles = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-60",
    ghost: "bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-200",
    danger: "bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-60",
    outline: "border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800",
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled} className={`${base} ${styles[variant]} ${className}`}>
      {children}
    </button>
  );
};

const Spinner = () => (
  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
  </svg>
);

const Badge = ({ tone = "zinc", children }) => (
  <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-${tone}-100 text-${tone}-700 dark:bg-${tone}-900/50 dark:text-${tone}-200`}>{children}</span>
);

// Safe dynamic tone badge (Tailwind needs explicit classes)
const SentimentBadge = ({ val }) => {
  if (typeof val !== "number") return <Badge tone="zinc">{String(val)}</Badge>;
  return val === 1 ? (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200">Positive</span>
  ) : (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-200">Negative</span>
  );
};

// ---------- App ----------
export default function App() {
  // Inputs / outputs
  const [review, setReview] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  const [batchResults, setBatchResults] = useState([]);

  // Statuses / metrics
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingBatch, setLoadingBatch] = useState(false);
  const [trainStatus, setTrainStatus] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState("");
  const [metrics, setMetrics] = useState("");
  const [dvcPushStatus, setDvcPushStatus] = useState(null);
  const [dvcStatus, setDvcStatus] = useState(null);

  const eventSourceRef = useRef(null);
  const logEndRef = useRef(null);

  // Auto-scroll training logs when updated
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [trainingLogs]);

  // Handle single review prediction
  const handlePredict = async (e) => {
    e.preventDefault();
    setLoadingPredict(true);
    setPrediction(null);
    try {
      const response = await axios.post(`${PREDICTION_API}/predict`, { reviews: [review] });
      setPrediction(response.data.predictions[0]);
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setPrediction(`Error: ${detail}`);
    } finally {
      setLoadingPredict(false);
    }
  };

  // Handle batch prediction
  const handleBatchPredict = async (e) => {
    e.preventDefault();
    setLoadingBatch(true);
    setBatchResults([]);
    if (!csvFile) {
      alert("Please upload a CSV file!");
      setLoadingBatch(false);
      return;
    }
    const reader = new FileReader();
    reader.onload = async (event) => {
      const text = event.target.result;
      const reviews = text.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
      try {
        const response = await axios.post(`${PREDICTION_API}/batch_predict`, { reviews });
        setBatchResults(response.data.predictions);
      } catch (error) {
        const detail = error.response?.data?.detail || error.message;
        setBatchResults([`Error: ${detail}`]);
      } finally {
        setLoadingBatch(false);
      }
    };
    reader.readAsText(csvFile);
  };

  // Trigger non-streaming training (fallback)
  const handleTrain = async () => {
    setTrainStatus(null);
    try {
      const response = await axios.post(`${TRAINING_API}/train`);
      setTrainStatus(response.data.detail || "Training started!");
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setTrainStatus(`Training error: ${detail}`);
    }
  };

  // Stream training logs (preferred)
  const handleStreamTraining = () => {
    if (streaming && eventSourceRef.current) return;
    setTrainingLogs("");
    setTrainStatus("Streaming training logs...");
    setStreaming(true);

    const es = new EventSource(`${TRAINING_API}/train_stream`);
    eventSourceRef.current = es;

    es.onmessage = (e) => setTrainingLogs((prev) => prev + e.data + "\n");

    es.addEventListener("end", () => {
      setTrainingLogs((prev) => prev + "[TRAINING COMPLETE]\n");
      setTrainStatus("Training finished");
      setStreaming(false);
      es.close();
      eventSourceRef.current = null;
    });

    es.onerror = (e) => {
      setTrainingLogs((prev) => prev + `[STREAM ERROR] ${JSON.stringify(e)}\n`);
      setTrainStatus("Streaming error, check logs");
      setStreaming(false);
      es.close();
      eventSourceRef.current = null;
    };
  };

  // Fetch Prometheus metrics (text)
  const handleFetchMetrics = async () => {
    try {
      const response = await axios.get(`${PREDICTION_API}/metrics`, { responseType: "text" });
      setMetrics(response.data);
    } catch (error) {
      const detail = error.response?.data || error.message;
      setMetrics(`Error fetching metrics: ${detail}`);
    }
  };

  // DVC push / status
  const handleDvcPush = async () => {
    setDvcPushStatus(null);
    try {
      const response = await axios.post(`${TRAINING_API}/dvc_push`);
      setDvcPushStatus(response.data.detail || "Push completed!");
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setDvcPushStatus(`DVC Push error: ${detail}`);
    }
  };

  const handleDvcStatus = async () => {
    setDvcStatus(null);
    try {
      const response = await axios.get(`${TRAINING_API}/dvc_status`);
      setDvcStatus(response.data.detail || "Checked status!");
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setDvcStatus(`DVC Status error: ${detail}`);
    }
  };

  // Cleanup SSE
  useEffect(() => () => eventSourceRef.current?.close(), []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-rose-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950 text-zinc-900 dark:text-zinc-50">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur bg-white/60 dark:bg-zinc-900/50 border-b border-zinc-200/60 dark:border-zinc-800">
        <div className="max-w-6xl mx-auto px-5 py-4 flex items-center justify-between">
          <motion.h1 initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="text-xl md:text-2xl font-bold tracking-tight">
            IMDB Sentiment Dashboard
          </motion.h1>
          <div className="text-xs text-zinc-500">API: {PREDICTION_API.replace(/^https?:\/\//, "")} | Train: {TRAINING_API.replace(/^https?:\/\//, "")}</div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-5 py-6 grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left column */}
        <section className="lg:col-span-2 space-y-5">
          {/* Single Review */}
          <Card>
            <SectionTitle title="Single Review Prediction" desc="Type a review and get instant sentiment" icon={<span>🎯</span>} />
            <form onSubmit={handlePredict} className="space-y-3">
              <textarea
                rows={5}
                value={review}
                onChange={(e) => setReview(e.target.value)}
                placeholder="Type an IMDB review..."
                className="w-full resize-y rounded-xl border border-zinc-300/70 dark:border-zinc-700 bg-white/70 dark:bg-zinc-900/50 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <div className="flex items-center gap-3 flex-wrap">
                <Button type="submit" disabled={loadingPredict || !review.trim()}>
                  {loadingPredict ? (<><Spinner /> Predicting…</>) : (<>Predict Sentiment</>)}
                </Button>
                <Button variant="ghost" onClick={() => { setReview(""); setPrediction(null); }}>Clear</Button>
                <AnimatePresence>
                  {prediction !== null && (
                    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} className="ml-auto">
                      <div className="text-sm text-zinc-600 dark:text-zinc-300">Result:</div>
                      <div className="mt-1">
                        <SentimentBadge val={prediction} />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </form>
          </Card>

          {/* Training */}
          <Card>
            <SectionTitle title="Train Model" desc="Start training and watch logs live" icon={<span>🧪</span>} />
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleTrain}>Start Training (blocking)</Button>
              <Button onClick={handleStreamTraining} disabled={streaming} variant="outline">
                {streaming ? (<><Spinner /> Streaming…</>) : (<>Start Training & Stream Logs</>)}
              </Button>
              {trainStatus && <span className="text-sm text-zinc-500">{trainStatus}</span>}
            </div>
            <div className="mt-4">
              <div className="text-sm font-medium mb-2">Live Training Logs</div>
              <div className="relative rounded-xl bg-zinc-950 text-zinc-100 border border-zinc-800 p-3 max-h-80 overflow-y-auto font-mono text-[12px] leading-relaxed">
                <AnimatePresence>
                  {trainingLogs.split("\n").map((line, idx) => (
                    <motion.div key={idx} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.08, delay: idx * 0.01 }} className="whitespace-pre-wrap">
                      {line}
                    </motion.div>
                  ))}
                </AnimatePresence>
                <div ref={logEndRef} />
                <div className="absolute bottom-2 right-3 text-[10px] text-zinc-500 animate-pulse">live</div>
              </div>
            </div>
          </Card>

          {/* Metrics */}
          <Card>
            <SectionTitle title="Prometheus Metrics" desc="Fetch exporter text for diagnostics" icon={<span>📈</span>} />
            <div className="flex items-center gap-3">
              <Button onClick={handleFetchMetrics}>{metrics ? "Refresh" : "Fetch Metrics"}</Button>
              <Button variant="ghost" onClick={() => setMetrics("")}>Clear</Button>
            </div>
            <div className="mt-3">
              <pre className="rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-3 overflow-auto max-h-80 text-[12px]">{metrics || "No metrics fetched yet."}</pre>
            </div>
          </Card>
        </section>

        {/* Right column */}
        <section className="space-y-5">
          {/* Batch Prediction */}
          <Card>
            <SectionTitle title="Batch Predict from CSV" desc="Upload a newline-separated CSV of reviews" icon={<span>🗂️</span>} />
            <form onSubmit={handleBatchPredict} className="space-y-3">
              <label className="block">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setCsvFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:px-4 file:py-2 file:text-white hover:file:bg-indigo-700 file:transition"
                />
              </label>
              <div className="flex items-center gap-3">
                <Button type="submit" disabled={loadingBatch || !csvFile}>
                  {loadingBatch ? (<><Spinner /> Predicting…</>) : (<>Batch Predict</>)}
                </Button>
                <Button variant="ghost" onClick={() => { setCsvFile(null); setBatchResults([]); }}>Clear</Button>
              </div>
            </form>

            <AnimatePresence>
              {batchResults.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="mt-4">
                  <div className="text-sm font-medium mb-2">Batch Results</div>
                  <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden">
                    <table className="min-w-full text-sm">
                      <thead className="bg-zinc-50 dark:bg-zinc-900">
                        <tr>
                          <th className="text-left px-3 py-2 font-medium">#</th>
                          <th className="text-left px-3 py-2 font-medium">Prediction</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.map((res, i) => (
                          <tr key={i} className="odd:bg-white even:bg-zinc-50 dark:odd:bg-zinc-950 dark:even:bg-zinc-900">
                            <td className="px-3 py-2">{i + 1}</td>
                            <td className="px-3 py-2">
                              {typeof res === "number" ? <SentimentBadge val={res} /> : <span className="text-rose-600 dark:text-rose-400">{String(res)}</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>

          {/* DVC Controls */}
          <Card>
            <SectionTitle title="DVC Controls" desc="Manage artifacts and status" icon={<span>🧱</span>} />
            <div className="flex flex-wrap gap-3">
              <Button onClick={handleDvcPush}>Push Artifacts</Button>
              <Button onClick={handleDvcStatus} variant="outline">Check Status</Button>
            </div>
            <div className="mt-3 space-y-2 text-sm">
              {dvcPushStatus && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2">
                  <span className="text-zinc-500">Push:</span>
                  <span className="font-medium">{dvcPushStatus}</span>
                </motion.div>
              )}
              {dvcStatus && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2">
                  <span className="text-zinc-500">Status:</span>
                  <span className="font-medium">{dvcStatus}</span>
                </motion.div>
              )}
            </div>
          </Card>

          {/* Tips */}
          <Card>
            <SectionTitle title="Tips" desc="Pro settings for smooth local dev" icon={<span>💡</span>} />
            <ul className="list-disc pl-5 text-sm space-y-1 text-zinc-600 dark:text-zinc-300">
              <li>Configure <code>REACT_APP_PREDICTION_API</code> and <code>REACT_APP_TRAINING_API</code> to point at your services.</li>
              <li>Tailwind classes assume a global Tailwind setup (PostCSS + tailwind.config.cjs).</li>
              <li>Framer Motion powers the subtle fades, pulses, and streaming log reveals.</li>
            </ul>
          </Card>
        </section>
      </main>

      {/* Footer */}
      <footer className="max-w-6xl mx-auto px-5 py-8 text-center text-xs text-zinc-500">
        Built with <a className="underline decoration-dotted" href="https://tailwindcss.com/" target="_blank" rel="noreferrer">Tailwind</a> + <a className="underline decoration-dotted" href="https://www.framer.com/motion/" target="_blank" rel="noreferrer">Framer Motion</a>
      </footer>
    </div>
  );
}
