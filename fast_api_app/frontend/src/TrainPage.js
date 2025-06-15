import React, { useState } from 'react';

export default function TrainPage() {
  const [message, setMessage] = useState('');

  const triggerTrain = async () => {
    setMessage('Starting training...');
    const resp = await fetch('/train', { method: 'POST' });
    const text = await resp.text();
    if (resp.ok) setMessage(text || 'Training triggered');
    else setMessage('Error: ' + text);
  };

  const triggerPush = async () => {
    setMessage('Pushing artifacts...');
    const resp = await fetch('/dvc_push', { method: 'POST' });
    const text = await resp.text();
    if (resp.ok) setMessage(text || 'Push completed');
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
          {message && <p className="text-center mt-4">{message}</p>}
        </div>
      </div>
    </div>
  );
}