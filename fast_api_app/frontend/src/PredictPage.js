import React, { useState } from "react";

export default function PredictPage() {
  const [reviews, setReviews] = useState([""]); 
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('single');

  const handleAddReview = () => setReviews([...reviews, ""]);
  const handleChange = (i, val) => {
    const copy = [...reviews];
    copy[i] = val;
    setReviews(copy);
  };

  const handleSubmit = async () => {
    setLoading(true);
    const endpoint = mode === 'single' ? '/predict' : '/batch_predict';
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviews })
    });
    if (!resp.ok) {
      alert("Error: " + (await resp.text()));
      setLoading(false);
      return;
    }
    const { predictions } = await resp.json();
    setPredictions(predictions);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white p-6 rounded-2xl shadow-lg w-full max-w-xl">
        <h1 className="text-2xl font-bold mb-4 text-center">IMDB Sentiment Prediction</h1>

        <div className="flex justify-center mb-4">
          <button
            className={`px-4 py-2 mx-1 rounded-lg ${mode==='single'? 'bg-blue-500 text-white':''}`}
            onClick={()=>setMode('single')}
          >Single</button>
          <button
            className={`px-4 py-2 mx-1 rounded-lg ${mode==='batch'? 'bg-blue-500 text-white':''}`}
            onClick={()=>setMode('batch')}
          >Batch</button>
        </div>

        {reviews.map((r, i) => (
          <textarea
            key={i}
            rows={3}
            className="w-full p-2 mb-3 border rounded-lg focus:outline-blue-400"
            placeholder="Enter movie review here..."
            value={r}
            onChange={e => handleChange(i, e.target.value)}
          />
        ))}
        <button
          onClick={handleAddReview}
          className="mb-4 text-blue-600 hover:underline"
        >+ Add another review</button>

        <button
          disabled={loading}
          onClick={handleSubmit}
          className="w-full bg-green-500 text-white py-2 rounded-lg hover:bg-green-600 transition"
        >{loading ? 'Predicting...' : 'Predict'}</button>

        {predictions.length > 0 && (
          <div className="mt-6">
            <h2 className="text-xl font-semibold">Predictions:</h2>
            <ul className="list-disc ml-5 mt-2">
              {predictions.map((p, idx) => (
                <li key={idx} className="capitalize">Review {idx+1}: {p === 1 ? 'Positive' : 'Negative'}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}