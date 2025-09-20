import React, { useState, useRef, useEffect, useMemo } from "react";
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

// Safe sentiment badge (no dynamic Tailwind strings)
const SentimentBadge = ({ val }) => {
  if (typeof val !== "number") {
    return (
      <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-zinc-100 text-zinc-700 dark:bg-zinc-900/50 dark:text-zinc-200">
        {String(val)}
      </span>
    );
  }
  return val === 1 ? (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200">Positive</span>
  ) : (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-200">Negative</span>
  );
};

// Progress bar
const ProgressBar = ({ value = 0, label = "Progress", sublabel = "" }) => (
  <div>
    <div className="flex justify-between items-end text-xs mb-1">
      <div className="text-zinc-600 dark:text-zinc-400">{label}</div>
      <div className="text-zinc-500">{Math.round(value)}%</div>
    </div>
    <div className="w-full bg-zinc-200 dark:bg-zinc-800 rounded-full h-2 overflow-hidden">
      <div className="bg-indigo-600 h-2 transition-[width] duration-300 ease-out" style={{ width: `${Math.min(100, Math.max(0, value))}%` }} />
    </div>
    {sublabel && <div className="text-[11px] text-zinc-500 mt-1">{sublabel}</div>}
  </div>
);

// --- Collapsible preformatted text for very long logs ---
const CollapsiblePre = ({
  text,
  defaultLines = 200,
  className = "",
  label = "Output",
}) => {
  const [expanded, setExpanded] = useState(false);
  const lines = useMemo(() => (text || "").split("\n"), [text]);
  const isLong = lines.length > defaultLines;
  const shown = expanded || !isLong ? lines : lines.slice(-defaultLines);
  const hiddenCount = Math.max(0, lines.length - shown.length);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text || "");
    } catch {
      /* ignore */
    }
  };

  return (
    <div className={className}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="text-zinc-500 text-sm">
          {label}
          {isLong && !expanded && (
            <span className="ml-2 text-xs">
              (showing last {defaultLines} lines, hidden {hiddenCount})
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isLong && (
            <Button variant="outline" onClick={() => setExpanded((v) => !v)}>
              {expanded ? "Show last lines" : "Show all"}
            </Button>
          )}
          <Button variant="ghost" onClick={handleCopy}>Copy</Button>
        </div>
      </div>

      <pre className="rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-3 overflow-auto max-h-60 text-[12px]">
        {shown.join("\n")}
      </pre>
    </div>
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
  const [trainingId, setTrainingId] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState("");
  const [metrics, setMetrics] = useState("");
  const [dvcStatus, setDvcStatus] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [pipelineInfo, setPipelineInfo] = useState(null);

  // Progress state
  const [totalStages, setTotalStages] = useState(null);
  const [completedStages, setCompletedStages] = useState(0);
  const [currentStage, setCurrentStage] = useState("");
  const [progress, setProgress] = useState(0);

  // Reproduce UI
  const [reproduceId, setReproduceId] = useState("");
  const [reproducingRunId, setReproducingRunId] = useState(null); // Changed state
  const [reproduceResult, setReproduceResult] = useState(null);

  const eventSourceRef = useRef(null);
  const logEndRef = useRef(null);

  // Auto-scroll anchor for short bursts
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [trainingLogs]);

  // --- Helpers for progress ---
  const fetchStageCount = async () => {
    try {
      const { data } = await axios.get(`${TRAINING_API}/pipeline_info`);
      const count = Array.isArray(data?.dag?.nodes) ? data.dag.nodes.length : null;
      setPipelineInfo(data);
      setTotalStages(count);
      setCompletedStages(0);
      setCurrentStage("");
      setProgress(0);
    } catch {
      setTotalStages(null); // unknown => indeterminate
      setCompletedStages(0);
      setCurrentStage("");
      setProgress(0);
    }
  };

  const updateProgressFromLog = (line) => {
    if (!line) return;

    // detect current running stage
    const runMatch = line.match(/Running stage '([^']+)'/i);
    if (runMatch && runMatch[1]) {
      setCurrentStage(runMatch[1]);
    }

    // stage completed (finished) or skipped due to cache
    const cachedMatch = line.match(/Stage '([^']+)' is cached/i);
    const finishedMatch = line.match(/Stage '([^']+)' finished/i);

    if (cachedMatch || finishedMatch) {
      setCompletedStages((prev) => {
        const next = (prev || 0) + 1;
        if (totalStages && totalStages > 0) {
          setProgress(Math.min(100, (next / totalStages) * 100));
        }
        return next;
      });
    }

    // if DVC prints lock update at the very end, force 100% for safety
    if (/Updating lock file/i.test(line) || /Pipeline completed/i.test(line)) {
      if (totalStages && completedStages >= totalStages - 1) {
        setCompletedStages(totalStages);
        setProgress(100);
      }
    }
  };

  // Handle single review prediction
  const handlePredict = async (e) => {
    e.preventDefault();
    setLoadingPredict(true);
    setPrediction(null);
    try {
      const response = await axios.post(`${PREDICTION_API}/predict`, { reviews: [review] });
      setPrediction(response.data.predictions[0]);
    } catch (error) {
      const detail = error?.response?.data?.detail || error.message;
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
      const reviews = String(text).split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
      try {
        const response = await axios.post(`${PREDICTION_API}/batch_predict`, { reviews });
        setBatchResults(response.data.predictions);
      } catch (error) {
        const detail = error?.response?.data?.detail || error.message;
        setBatchResults([`Error: ${detail}`]);
      } finally {
        setLoadingBatch(false);
      }
    };
    reader.readAsText(csvFile);
  };

  // Non-streaming training (blocking HTTP)
  const handleTrain = async () => {
    setTrainStatus(null);
    setTrainingId(null);
    setProgress(0);
    setCurrentStage("");
    setCompletedStages(0);
    await fetchStageCount(); // initialize stage count

    try {
      const { data } = await axios.post(`${TRAINING_API}/train`);
      setTrainStatus(data.status || "Training completed");
      setTrainingId(data.training_id || null);
      // blocking endpoint returns when done; mark complete
      setCompletedStages(totalStages || completedStages);
      setProgress(100);
    } catch (error) {
      const detail = error?.response?.data?.detail || error.message;
      setTrainStatus(`Training error: ${detail}`);
    }
  };

  // Streaming training logs with progress
  const handleStreamTraining = async () => {
    if (streaming || eventSourceRef.current) return;
    setTrainingLogs("");
    setTrainStatus("Streaming training logs…");
    setStreaming(true);

    await fetchStageCount(); // fetch total stages before streaming

    const es = new EventSource(`${TRAINING_API}/train_stream`);
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      const line = e.data || "";
      setTrainingLogs((prev) => prev + line + "\n");
      updateProgressFromLog(line);
    };

    es.addEventListener("end", () => {
      setTrainingLogs((prev) => prev + "[TRAINING COMPLETE]\n");
      setTrainStatus("Training finished");
      setStreaming(false);
      setProgress(100);
      if (totalStages && completedStages < totalStages) {
        setCompletedStages(totalStages);
      }
      es.close();
      eventSourceRef.current = null;
    });

    es.onerror = (e) => {
      setTrainingLogs((prev) => prev + `[STREAM ERROR] ${JSON.stringify(e)}\n`);
      setTrainStatus("Streaming error, check backend logs");
      setStreaming(false);
      es.close();
      eventSourceRef.current = null;
    };
  };

  // Fetch Prometheus metrics
  const handleFetchMetrics = async () => {
    try {
      const response = await axios.get(`${TRAINING_API}/metrics`, { responseType: "text" });
      setMetrics(response.data);
    } catch (error) {
      const detail = error?.response?.data || error.message;
      setMetrics(`Error fetching metrics: ${detail}`);
    }
  };

  // DVC status
  const handleDvcStatus = async () => {
    try {
      const { data } = await axios.get(`${TRAINING_API}/dvc_status`);
      setDvcStatus(data.dvc_status || data);
    } catch (error) {
      const detail = error?.response?.data?.detail || error.message;
      setDvcStatus({ error: `DVC Status error: ${detail}` });
    }
  };

  // Training history
  const handleTrainingHistory = async () => {
    try {
      const { data } = await axios.get(`${TRAINING_API}/training_history`);
      setTrainingHistory(Array.isArray(data.training_runs) ? data.training_runs : []);
    } catch (error) {
      const detail = error?.response?.data?.detail || error.message;
      setTrainingHistory([`Error: ${detail}`]);
    }
  };

  // Pipeline info
  const handlePipelineInfo = async () => {
    try {
      const { data } = await axios.get(`${TRAINING_API}/pipeline_info`);
      setPipelineInfo(data);
    } catch (error) {
      const detail = error?.response?.data?.detail || error.message;
      setPipelineInfo({ error: detail });
    }
  };

  // Reproduce run
  const handleReproduce = async (id) => {
    const target = (id || reproduceId || "").trim();
    if (!target) return;
    setReproducingRunId(target); // Set the specific ID
    setReproduceResult(null);
    try {
      const { data } = await axios.post(`${TRAINING_API}/reproduce/${encodeURIComponent(target)}`);
      setReproduceResult(data);
    } catch (error) {
      const detail = error?.response?.data?.detail || error.message;
      setReproduceResult({ error: detail });
    } finally {
      setReproducingRunId(null); // Clear the ID
    }
  };

  // Cleanup SSE
  useEffect(() => () => eventSourceRef.current?.close(), []);

  // Labels for progress
  const progressLabel =
    totalStages ? `Stages: ${completedStages}/${totalStages}` : streaming ? "Streaming…" : "Preparing…";
  const progressSub =
    currentStage ? `Current stage: ${currentStage}` : totalStages ? "" : "Stage count not available";

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-rose-50 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950 text-zinc-900 dark:text-zinc-50">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur bg-white/60 dark:bg-zinc-900/50 border-b border-zinc-200/60 dark:border-zinc-800">
        <div className="max-w-6xl mx-auto px-5 py-4 flex items-center justify-between">
          <motion.h1 initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="text-xl md:text-2xl font-bold tracking-tight">
            IMDB Sentiment Dashboard
          </motion.h1>
          <div className="text-xs text-zinc-500">
            API: {PREDICTION_API.replace(/^https?:\/\//, "")} | Train: {TRAINING_API.replace(/^https?:\/\//, "")}
          </div>
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
              {trainStatus && <span className="text-sm text-zinc-500">{trainStatus}{trainingId ? ` • id: ${trainingId}` : ""}</span>}
            </div>

            {/* Progress bar */}
            <div className="mt-4">
              <ProgressBar value={progress} label={progressLabel} sublabel={progressSub} />
            </div>

            {/* Live logs (collapsible viewer) */}
            <div className="mt-3">
              <CollapsiblePre
                text={trainingLogs}
                defaultLines={400}
                label="Live Training Logs"
              />
              <div ref={logEndRef} />
              <div className="mt-1 text-right text-[10px] text-zinc-500 animate-pulse">live</div>
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

          {/* DVC Controls + Reproduce */}
          <Card>
            <SectionTitle title="DVC Controls" desc="Manage artifacts, status, and reproduction" icon={<span>🧱</span>} />
            <div className="flex flex-wrap gap-3">
              <Button onClick={handleDvcStatus}>Check Status</Button>
              <Button onClick={handleTrainingHistory} variant="outline">Training History</Button>
              <Button onClick={handlePipelineInfo} variant="outline">Pipeline Info</Button>
            </div>

            {/* Reproduce run controls */}
            <div className="mt-4 space-y-2">
              <div className="text-sm font-medium">Reproduce a run</div>
              <div className="flex gap-2 flex-wrap">
                <input
                  value={reproduceId}
                  onChange={(e) => setReproduceId(e.target.value)}
                  placeholder="Enter training_id (e.g., 20250813-abc12345)"
                  className="flex-1 min-w-[240px] rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white/70 dark:bg-zinc-900/50 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <Button onClick={() => handleReproduce()} disabled={!reproduceId.trim() || reproducingRunId !== null}>
                  {reproducingRunId === reproduceId ? (<><Spinner /> Reproducing…</>) : (<>Reproduce</>)}
                </Button>
                <Button variant="ghost" onClick={() => { setReproduceId(""); setReproduceResult(null); }}>Clear</Button>
              </div>

              {reproduceResult && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  {(() => {
                    const { output, ...rest } = reproduceResult || {};
                    return (
                      <>
                        <div className="text-zinc-500 mb-1">Reproduce Metadata:</div>
                        <pre className="rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-3 overflow-auto max-h-60 text-[12px]">
                          {JSON.stringify(rest, null, 2)}
                        </pre>
                        {typeof output === "string" && output.length > 0 && (
                          <CollapsiblePre
                            text={output}
                            defaultLines={200}
                            label="Reproduction Output"
                            className="mt-3"
                          />
                        )}
                      </>
                    );
                  })()}
                </motion.div>
              )}
            </div>

            {/* One-click reproduce from history */}
            <div className="mt-4">
              {trainingHistory && trainingHistory.length > 0 && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <div className="text-sm font-medium mb-2">Training Runs</div>
                  <ul className="space-y-2">
                    {trainingHistory.map((id, idx) => {
                      // Check if THIS specific run is the one being reproduced
                      const isThisRunLoading = reproducingRunId === id;
                      return (
                        <li key={idx} className="flex items-center gap-2">
                          <span className="flex-1 break-all font-mono text-[12px]">{id}</span>
                          <Button
                            variant="outline"
                            onClick={() => handleReproduce(id)}
                            // Disable all buttons if any run is in progress
                            disabled={reproducingRunId !== null}
                            className="shrink-0"
                          >
                            {isThisRunLoading ? <><Spinner /> </> : null}
                            {isThisRunLoading ? "Running..." : "Reproduce"}
                          </Button>
                        </li>
                      );
                    })}
                  </ul>
                </motion.div>
              )}
            </div>

            {/* Status & Pipeline info blocks */}
            <div className="mt-4 space-y-3 text-sm">
              {dvcStatus && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <div className="text-zinc-500 mb-1">Status:</div>
                  <pre className="rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-3 overflow-auto max-h-60 text-[12px]">
                    {JSON.stringify(dvcStatus, null, 2)}
                  </pre>
                </motion.div>
              )}
              {pipelineInfo && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <div className="text-zinc-500 mb-1">Pipeline Info:</div>
                  <pre className="rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-3 overflow-auto max-h-60 text-[12px]">
                    {JSON.stringify(pipelineInfo, null, 2)}
                  </pre>
                </motion.div>
              )}
            </div>
          </Card>

          {/* Tips */}
          <Card>
            <SectionTitle title="Tips" desc="Pro settings for smooth local dev" icon={<span>💡</span>} />
            <ul className="list-disc pl-5 text-sm space-y-1 text-zinc-600 dark:text-zinc-300">
              <li>Set <code>VITE_PREDICTION_API</code> and <code>VITE_TRAINING_API</code> in your <code>.env</code>.</li>
              <li>Ensure CORS in backend allows this origin in production.</li>
              <li>Server-Sent Events require your ingress/proxy to support streaming.</li>
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