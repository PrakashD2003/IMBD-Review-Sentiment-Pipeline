import React, { useState } from 'react';

export default function TrainPage() {
  const [message, setMessage] = useState('');
  const [logs, setLogs] = useState([]);
  let eventSourceRef = null;
  
  const triggerTrain = () => {
    setMessage('Starting training...');
    setLogs([]);
    const es = new EventSource('/train_stream');
    eventSourceRef = es;
    es.onmessage = (e) => {
      setLogs((l) => [...l, e.data]);
    };
    es.addEventListener('end', () => {
      setMessage('Training completed');
      es.close();
    });
    es.onerror = () => {
      setMessage('Error during training');
      es.close();
    };
  };

  const triggerPush = async () => {
    setMessage('Pushing artifacts...');
    const resp = await fetch('/dvc_push', { method: 'POST' });
    const text = await resp.text();
    if (resp.ok) setMessage(text || 'Push completed');
    else setMessage('Error: ' + text);
  };

  const checkStatus = async () => {
    setMessage('Checking DVC status...');
    const resp = await fetch('/dvc_status');
    const text = await resp.text();
    if (resp.ok) setMessage(text);
    else setMessage('Error: ' + text);
  };


  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white p-6 rounded-2xl shadow-lg w-full max-w-xl">
        <h1 className="text-2xl font-bold mb-6 text-center">Model Training</h1>
        <div className="space-y-4">
          <button onClick={triggerTrain} className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600">
            Run Training Pipeline
          </button>
          <button onClick={triggerPush} className="w-full bg-green-500 text-white py-2 rounded-lg hover:bg-green-600">
            Push to DVC Remote
          </button>
          <button onClick={checkStatus} className="w-full bg-purple-500 text-white py-2 rounded-lg hover:bg-purple-600">
            Check DVC Status
          </button>
          {message && <p className="text-center mt-4">{message}</p>}
          {logs.length > 0 && (
            <pre className="bg-black text-white p-2 h-64 overflow-y-scroll mt-2 rounded-lg text-sm">
              {logs.join('\n')}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}