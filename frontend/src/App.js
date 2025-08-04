import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

function App() {
  // Inputs / outputs
  const [review, setReview] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  const [batchResults, setBatchResults] = useState([]);

  // Statuses / metrics
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingBatch, setLoadingBatch] = useState(false);
  const [trainStatus, setTrainStatus] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState('');
  const [metrics, setMetrics] = useState('');
  const [dvcPushStatus, setDvcPushStatus] = useState(null);
  const [dvcStatus, setDvcStatus] = useState(null);

  const eventSourceRef = useRef(null);
  const logEndRef = useRef(null);

  // Auto-scroll training logs when updated
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainingLogs]);

  // Handle single review prediction
  const handlePredict = async (e) => {
    e.preventDefault();
    setLoadingPredict(true);
    setPrediction(null);
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        reviews: [review]
      });
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
      const reviews = text
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
      try {
        const response = await axios.post('http://localhost:8000/batch_predict', {
          reviews: reviews
        });
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
      const response = await axios.post('http://localhost:8001/train');
      setTrainStatus(response.data.detail || "Training started!");
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setTrainStatus(`Training error: ${detail}`);
    }
  };

  // Stream training logs (preferred)
  const handleStreamTraining = () => {
    // If already streaming, ignore or reset
    if (streaming && eventSourceRef.current) {
      return;
    }

    setTrainingLogs('');
    setTrainStatus('Streaming training logs...');
    setStreaming(true);

    const es = new EventSource('http://localhost:8001/train_stream');
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      // Regular log line
      setTrainingLogs(prev => prev + e.data + '\n');
    };

    es.addEventListener('end', () => {
      setTrainingLogs(prev => prev + '[TRAINING COMPLETE]\n');
      setTrainStatus('Training finished'); 
      setStreaming(false);
      es.close();
      eventSourceRef.current = null;
    });

    es.onerror = (e) => {
      setTrainingLogs(prev => prev + `[STREAM ERROR] ${JSON.stringify(e)}\n`);
      setTrainStatus('Streaming error, check logs'); 
      setStreaming(false);
      es.close();
      eventSourceRef.current = null;
    };
  };

  // Fetch Prometheus metrics (text)
  const handleFetchMetrics = async () => {
    try {
      const response = await axios.get('http://localhost:8000/metrics', {
        responseType: 'text',
      });
      setMetrics(response.data);
    } catch (error) {
      const detail = error.response?.data || error.message;
      setMetrics(`Error fetching metrics: ${detail}`);
    }
  };

  // Trigger DVC push
  const handleDvcPush = async () => {
    setDvcPushStatus(null);
    try {
      const response = await axios.post('http://localhost:8001/dvc_push');
      setDvcPushStatus(response.data.detail || "Push completed!");
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setDvcPushStatus(`DVC Push error: ${detail}`);
    }
  };

  // Trigger DVC status
  const handleDvcStatus = async () => {
    setDvcStatus(null);
    try {
      const response = await axios.get('http://localhost:8001/dvc_status');
      setDvcStatus(response.data.detail || "Checked status!");
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setDvcStatus(`DVC Status error: ${detail}`);
    }
  };

  // Cleanup on unmount: close SSE
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return (
    <div style={{ margin: "2rem", fontFamily: "system-ui, sans-serif" }}>
      <h1>IMDB Review Sentiment Analysis</h1>

      {/* Single Review Prediction */}
      <form onSubmit={handlePredict} style={{ marginBottom: "1.5rem" }}>
        <h3>Single Review Prediction</h3>
        <textarea
          rows={4}
          cols={60}
          value={review}
          onChange={e => setReview(e.target.value)}
          placeholder="Type an IMDB review..."
        />
        <br />
        <button type="submit" disabled={loadingPredict}>Predict Sentiment</button>
        {prediction !== null && (
          <div style={{ marginTop: "0.5rem" }}>
            <strong>Sentiment:</strong>{" "}
            {typeof prediction === "number"
              ? prediction === 1
                ? "Positive"
                : "Negative"
              : prediction}
          </div>
        )}
      </form>

      {/* Batch Prediction */}
      <form onSubmit={handleBatchPredict} style={{ marginBottom: "1.5rem" }}>
        <h3>Batch Predict from CSV</h3>
        <input
          type="file"
          accept=".csv"
          onChange={e => setCsvFile(e.target.files[0])}
        />
        <button type="submit" disabled={loadingBatch}>Batch Predict</button>
        {batchResults.length > 0 && (
          <div style={{ marginTop: "0.5rem" }}>
            <strong>Batch Results:</strong>
            <ul>
              {batchResults.map((res, i) => (
                <li key={i}>
                  {`Review ${i + 1}: ${
                    typeof res === "number"
                      ? res === 1
                        ? "Positive"
                        : "Negative"
                      : res
                  }`}
                </li>
              ))}
            </ul>
          </div>
        )}
      </form>

      {/* Training Controls */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h3>Train Model</h3>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          <button onClick={handleTrain}>Start Training (blocking)</button>
          <button onClick={handleStreamTraining} disabled={streaming}>
            {streaming ? "Streaming..." : "Start Training & Stream Logs"}
          </button>
        </div>
        {trainStatus && (
          <div style={{ marginTop: "0.5rem" }}>
            <strong>Status:</strong> {trainStatus}
          </div>
        )}
        <div style={{ marginTop: "0.5rem" }}>
          <strong>Live Training Logs:</strong>
          <div
            style={{
              background: "#1e1e1e",
              color: "#f5f5f5",
              padding: "0.75rem",
              borderRadius: 6,
              maxHeight: 300,
              overflowY: "auto",
              fontFamily: "monospace",
              whiteSpace: "pre-wrap",
              position: "relative",
            }}
          >
            {trainingLogs || <div style={{ opacity: 0.6 }}>No logs yet.</div>}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>

      {/* DVC Push */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h3>DVC Push</h3>
        <button onClick={handleDvcPush}>Push Artifacts</button>
        {dvcPushStatus && (
          <div style={{ marginTop: "0.5rem" }}>
            <strong>Status:</strong> {dvcPushStatus}
          </div>
        )}
      </div>

      {/* DVC Status */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h3>DVC Status</h3>
        <button onClick={handleDvcStatus}>Check Status</button>
        {dvcStatus && (
          <div style={{ marginTop: "0.5rem" }}>
            <strong>Status:</strong> {dvcStatus}
          </div>
        )}
      </div>

      {/* Metrics */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h3>Prometheus Metrics</h3>
        <button onClick={handleFetchMetrics}>Fetch Metrics</button>
        <pre
          style={{
            background: "#f0f0f0",
            padding: "1rem",
            borderRadius: 6,
            overflowX: "auto",
            maxHeight: 300,
          }}
        >
          {metrics || "No metrics fetched yet."}
        </pre>
      </div>
    </div>
  );
}

export default App;
