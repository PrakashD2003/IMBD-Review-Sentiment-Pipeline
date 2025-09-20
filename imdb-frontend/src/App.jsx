// imdb-frontend/src/App.jsx

import React, { useState, useRef, useEffect, useMemo } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";

// --- Configuration ---
// These are read from environment variables at build time (e.g., in a .env file)
const PREDICTION_API = import.meta.env.VITE_PREDICTION_API || "http://localhost:8000";
const TRAINING_API   = import.meta.env.VITE_TRAINING_API   || "http://localhost:8001";

// --- Design System Primitives (Updated) ---
// Note: Assumes 'Inter' and 'JetBrains Mono' fonts are imported in your main CSS file (e.g., index.css).

const Card = ({ children, className = "" }) => (
  <div className={`rounded-2xl bg-white dark:bg-slate-900 shadow-lg border border-slate-200/80 dark:border-slate-800 p-6 ${className}`}>
    {children}
  </div>
);

const SectionTitle = ({ title, desc, icon }) => (
  <div className="mb-5">
    <div className="flex items-center gap-2">
      <span className="text-xl">{icon}</span>
      <h3 className="text-lg font-semibold tracking-tight text-slate-800 dark:text-slate-100">{title}</h3>
    </div>
    {desc && <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{desc}</p>}
  </div>
);

const Button = ({ children, onClick, type = "button", disabled = false, variant = "primary", className = "" }) => {
  const base = "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-slate-900";
  const styles = {
    primary: "bg-teal-600 text-white hover:bg-teal-700 disabled:opacity-50 focus:ring-teal-500",
    ghost: "bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 disabled:opacity-50 focus:ring-slate-400",
    danger: "bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-50 focus:ring-rose-500",
    outline: "border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 focus:ring-teal-500",
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

const SentimentBadge = ({ val }) => {
  if (typeof val !== "number") {
    return (
      <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
        {String(val)}
      </span>
    );
  }
  return val === 1 ? (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-200">Positive</span>
  ) : (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-200">Negative</span>
  );
};

const ProgressBar = ({ value = 0, label = "Progress", sublabel = "" }) => (
  <div>
    <div className="flex justify-between items-end text-xs mb-1">
      <div className="font-medium text-slate-600 dark:text-slate-300">{label}</div>
      <div className="text-slate-500">{Math.round(value)}%</div>
    </div>
    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 overflow-hidden">
      <div className="bg-teal-600 h-2 rounded-full transition-[width] duration-300 ease-out" style={{ width: `${Math.min(100, Math.max(0, value))}%` }} />
    </div>
    {sublabel && <div className="text-[11px] text-slate-500 mt-1">{sublabel}</div>}
  </div>
);

const CollapsiblePre = ({ text, defaultLines = 200, className = "", label = "Output" }) => {
  const [expanded, setExpanded] = useState(false);
  const lines = useMemo(() => (text || "").split("\n"), [text]);
  const isLong = lines.length > defaultLines;
  const shown = expanded || !isLong ? lines : lines.slice(-defaultLines);
  const hiddenCount = Math.max(0, lines.length - shown.length);

  const handleCopy = async () => { await navigator.clipboard.writeText(text || ""); };

  return (
    <div className={className}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="text-slate-500 text-sm">
          {label}
          {isLong && !expanded && <span className="ml-2 text-xs">(showing last {defaultLines} lines)</span>}
        </div>
        <div className="flex items-center gap-2">
          {isLong && <Button variant="outline" onClick={() => setExpanded((v) => !v)}>{expanded ? "Collapse" : `Show All ${lines.length}`}</Button>}
          <Button variant="ghost" onClick={handleCopy}>Copy</Button>
        </div>
      </div>
      <pre className="font-mono rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-3 overflow-auto max-h-80 text-[12px] text-slate-700 dark:text-slate-300">
        {shown.join("\n")}
      </pre>
    </div>
  );
};

const TabButton = ({ children, isActive, onClick }) => (
    <button
        onClick={onClick}
        className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors duration-200
            ${isActive
                ? 'bg-teal-600 text-white'
                : 'text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800'
            }`}
    >
        {children}
    </button>
);


// --- Tab Components ---

function InferenceTab() {
    const [review, setReview] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [loadingPredict, setLoadingPredict] = useState(false);
    const [csvFile, setCsvFile] = useState(null);
    const [batchResults, setBatchResults] = useState([]);
    const [loadingBatch, setLoadingBatch] = useState(false);

    const handlePredict = async (e) => {
        e.preventDefault();
        setLoadingPredict(true);
        setPrediction(null);
        try {
            const { data } = await axios.post(`${PREDICTION_API}/predict`, { reviews: [review] });
            setPrediction(data.predictions[0]);
        } catch (error) {
            setPrediction(`Error: ${error?.response?.data?.detail || error.message}`);
        } finally {
            setLoadingPredict(false);
        }
    };

    const handleBatchPredict = async (e) => {
        e.preventDefault();
        if (!csvFile) return;
        setLoadingBatch(true);
        setBatchResults([]);
        const reader = new FileReader();
        reader.onload = async (event) => {
            const reviews = String(event.target.result).split("\n").map(l => l.trim()).filter(Boolean);
            try {
                const { data } = await axios.post(`${PREDICTION_API}/batch_predict`, { reviews });
                setBatchResults(data.predictions);
            } catch (error) {
                setBatchResults([`Error: ${error?.response?.data?.detail || error.message}`]);
            } finally {
                setLoadingBatch(false);
            }
        };
        reader.readAsText(csvFile);
    };

    return (
        <div className="space-y-6">
            <Card>
                <SectionTitle title="Single Review Prediction" desc="Get instant sentiment for a single review" icon="🎯" />
                <form onSubmit={handlePredict} className="space-y-4">
                    <textarea
                        rows={5}
                        value={review}
                        onChange={(e) => setReview(e.target.value)}
                        placeholder="e.g., 'This movie was a masterpiece, the acting was superb!'"
                        className="w-full resize-y rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800/50 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-teal-500"
                    />
                    <div className="flex items-center gap-3 flex-wrap">
                        <Button type="submit" disabled={loadingPredict || !review.trim()}>
                            {loadingPredict ? <><Spinner /> Predicting...</> : "Predict Sentiment"}
                        </Button>
                        <Button variant="ghost" onClick={() => { setReview(""); setPrediction(null); }}>Clear</Button>
                        <AnimatePresence>
                            {prediction !== null && (
                                <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="ml-auto flex items-center gap-2">
                                    <span className="text-sm text-slate-600 dark:text-slate-300">Result:</span>
                                    <SentimentBadge val={prediction} />
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </form>
            </Card>

            <Card>
                <SectionTitle title="Batch Prediction" desc="Upload a newline-separated CSV of reviews for bulk processing" icon="🗂️" />
                <form onSubmit={handleBatchPredict} className="space-y-4">
                     <input
                        type="file"
                        accept=".csv,.txt"
                        onChange={(e) => setCsvFile(e.target.files?.[0] || null)}
                        className="block w-full text-sm text-slate-500 file:mr-4 file:rounded-lg file:border-0 file:bg-teal-600 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-teal-700 file:transition"
                    />
                    <div className="flex items-center gap-3">
                        <Button type="submit" disabled={loadingBatch || !csvFile}>
                            {loadingBatch ? <><Spinner /> Predicting...</> : "Run Batch Predict"}
                        </Button>
                        <Button variant="ghost" onClick={() => { setCsvFile(null); setBatchResults([]); }}>Clear</Button>
                    </div>
                </form>

                 <AnimatePresence>
                    {batchResults.length > 0 && (
                        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="mt-4 overflow-hidden">
                            <h4 className="text-sm font-medium mb-2">Batch Results ({batchResults.length})</h4>
                            <div className="rounded-xl border border-slate-200 dark:border-slate-800 max-h-60 overflow-y-auto">
                                <table className="min-w-full text-sm">
                                    <thead className="bg-slate-50 dark:bg-slate-950/50 sticky top-0">
                                        <tr>
                                            <th className="text-left px-4 py-2 font-medium">#</th>
                                            <th className="text-left px-4 py-2 font-medium">Prediction</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                                        {batchResults.map((res, i) => (
                                            <tr key={i} className="odd:bg-white even:bg-slate-50 dark:odd:bg-slate-900 dark:even:bg-slate-800/50">
                                                <td className="px-4 py-2 text-slate-500">{i + 1}</td>
                                                <td className="px-4 py-2"><SentimentBadge val={res} /></td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </Card>
        </div>
    );
}

function MlopsTab() {
    // Component states...
    const [trainStatus, setTrainStatus] = useState(null);
    const [streaming, setStreaming] = useState(false);
    const [trainingLogs, setTrainingLogs] = useState("");
    const [dvcStatus, setDvcStatus] = useState(null);
    const [trainingHistory, setTrainingHistory] = useState([]);
    const [reproduceId, setReproduceId] = useState("");
    const [reproducingRunId, setReproducingRunId] = useState(null);
    const [reproduceResult, setReproduceResult] = useState(null);

    // Progress states...
    const [totalStages, setTotalStages] = useState(null);
    const [completedStages, setCompletedStages] = useState(0);
    const [currentStage, setCurrentStage] = useState("");
    const [progress, setProgress] = useState(0);

    const eventSourceRef = useRef(null);
    useEffect(() => () => eventSourceRef.current?.close(), []);

    // --- Handlers ---

    const handleDvcStatus = async () => {
        setDvcStatus({ loading: "Fetching status..." });
        try {
            const { data } = await axios.get(`${TRAINING_API}/dvc_status`);
            setDvcStatus(data.dvc_status || data);
        } catch (error) {
            const detail = error?.response?.data?.detail || error.message;
            setDvcStatus({ error: `DVC Status error: ${detail}` });
        }
    };

    const handleTrainingHistory = async () => {
        try {
            const { data } = await axios.get(`${TRAINING_API}/training_history`);
            setTrainingHistory(Array.isArray(data.training_runs) ? data.training_runs : []);
        } catch (error) {
            const detail = error?.response?.data?.detail || error.message;
            setTrainingHistory([`Error: ${detail}`]);
        }
    };

    const handleReproduce = async (id) => {
        const target = (id || reproduceId || "").trim();
        if (!target) return;
        setReproducingRunId(target);
        setReproduceResult(null);
        try {
            const { data } = await axios.post(`${TRAINING_API}/reproduce/${encodeURIComponent(target)}`);
            setReproduceResult(data);
        } catch (error) {
            const detail = error?.response?.data?.detail || error.message;
            setReproduceResult({ error: detail });
        } finally {
            setReproducingRunId(null);
        }
    };

    const handleStreamTraining = async () => {
        if (streaming) return;
        setTrainingLogs("");
        setTrainStatus("Streaming training logs…");
        setStreaming(true);
        setProgress(0);
        setCurrentStage("");
        setCompletedStages(0);

        const es = new EventSource(`${TRAINING_API}/train_stream`);
        eventSourceRef.current = es;

        es.onmessage = (e) => setTrainingLogs((prev) => prev + (e.data || "") + "\n");
        es.addEventListener("end", () => {
            setTrainStatus("Training finished");
            setStreaming(false);
            es.close();
        });
        es.onerror = () => {
            setTrainStatus("Streaming error");
            setStreaming(false);
            es.close();
        };
    };

    return (
        <div className="space-y-6">
            <Card>
                <SectionTitle title="Train Model" desc="Trigger the DVC pipeline and stream logs in real-time" icon="🧪" />
                <div className="flex flex-wrap items-center gap-3">
                    <Button onClick={handleStreamTraining} disabled={streaming}>
                        {streaming ? <><Spinner /> Streaming...</> : "Start Training & Stream Logs"}
                    </Button>
                    {trainStatus && <span className="text-sm text-slate-500">{trainStatus}</span>}
                </div>

                <div className="mt-4">
                  <ProgressBar value={progress} label="Training Progress" sublabel={currentStage} />
                </div>

                <div className="mt-4">
                  <CollapsiblePre text={trainingLogs} label="Live Training Logs" />
                </div>
            </Card>

            <Card>
                <SectionTitle title="MLOps Controls" desc="Manage artifacts, status, and reproducibility" icon="🧱" />
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* DVC Status & History */}
                    <div className="space-y-4">
                        <Button onClick={handleDvcStatus} variant="outline" className="w-full">Check DVC Status</Button>
                        {dvcStatus && <CollapsiblePre text={JSON.stringify(dvcStatus, null, 2)} label="DVC Status" />}
                        <Button onClick={handleTrainingHistory} variant="outline" className="w-full">Fetch Training History</Button>
                        {trainingHistory.length > 0 && (
                            <div>
                                <h4 className="text-sm font-medium mb-2">Training Runs</h4>
                                <ul className="space-y-2 max-h-60 overflow-y-auto pr-2">
                                    {trainingHistory.map(id => (
                                        <li key={id} className="flex items-center gap-2 text-sm">
                                            <span className="flex-1 font-mono text-xs truncate">{id}</span>
                                            <Button size="sm" variant="ghost" onClick={() => handleReproduce(id)} disabled={!!reproducingRunId}>
                                                {reproducingRunId === id ? <Spinner/> : "Reproduce"}
                                            </Button>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                    {/* Reproduce Run */}
                    <div className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Reproduce by ID</label>
                            <input
                                value={reproduceId}
                                onChange={(e) => setReproduceId(e.target.value)}
                                placeholder="Enter training_id..."
                                className="w-full rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800/50 px-3 py-2 text-sm"
                            />
                            <Button onClick={() => handleReproduce(reproduceId)} disabled={!reproduceId.trim() || !!reproducingRunId}>
                                {reproducingRunId === reproduceId ? <><Spinner /> Reproducing...</> : "Reproduce"}
                            </Button>
                        </div>
                        {reproduceResult && <CollapsiblePre text={JSON.stringify(reproduceResult, null, 2)} label="Reproduction Result" />}
                    </div>
                </div>
            </Card>
        </div>
    );
}

function SystemHealthTab() {
    const [metrics, setMetrics] = useState("");
    const handleFetchMetrics = async () => {
        try {
            const { data } = await axios.get(`${TRAINING_API}/metrics`, { responseType: "text" });
            setMetrics(data);
        } catch (error) {
            setMetrics(`Error: ${error?.response?.data || error.message}`);
        }
    };

    return (
        <Card>
            <SectionTitle title="System Health" desc="Fetch Prometheus metrics for diagnostics" icon="📈" />
            <div className="flex items-center gap-3">
                <Button onClick={handleFetchMetrics}>{metrics ? "Refresh Metrics" : "Fetch Metrics"}</Button>
                <Button variant="ghost" onClick={() => setMetrics("")}>Clear</Button>
            </div>
            <div className="mt-4">
                <CollapsiblePre text={metrics} label="Prometheus Metrics" defaultLines={500} />
            </div>
        </Card>
    );
}


// ---------- Main App Component ----------
export default function App() {
  const [activeTab, setActiveTab] = useState("inference");

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-50 font-sans">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur-lg bg-white/80 dark:bg-slate-900/80 border-b border-slate-200/60 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <motion.h1 initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="text-xl md:text-2xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
            IMDB Sentiment MLOps Dashboard
          </motion.h1>
          <div className="text-xs text-slate-500 hidden sm:block">
            Prediction API: {PREDICTION_API}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Tab Navigation */}
        <div className="mb-6 p-1.5 bg-slate-100 dark:bg-slate-900 rounded-xl flex items-center gap-2 w-full sm:w-auto">
            <TabButton isActive={activeTab === 'inference'} onClick={() => setActiveTab('inference')}>Inference</TabButton>
            <TabButton isActive={activeTab === 'mlops'} onClick={() => setActiveTab('mlops')}>MLOps</TabButton>
            <TabButton isActive={activeTab === 'health'} onClick={() => setActiveTab('health')}>System Health</TabButton>
        </div>

        {/* Tab Content */}
        <div>
            {activeTab === 'inference' && <InferenceTab />}
            {activeTab === 'mlops' && <MlopsTab />}
            {activeTab === 'health' && <SystemHealthTab />}
        </div>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-6 py-8 text-center text-xs text-slate-500">
        Professional MLOps Interface | Powered by React, Tailwind CSS, and FastAPI
      </footer>
    </div>
  );
}