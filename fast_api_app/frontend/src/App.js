import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import PredictPage from './PredictPage';
import TrainPage from './TrainPage';



export default function App() {
  return (
    <Router>
      <nav className="bg-gray-800 text-white p-4 flex space-x-4">
        <Link to="/" className="hover:underline">Predict</Link>
        <Link to="/train" className="hover:underline">Train</Link>
      </nav>
      <Routes>
        <Route path="/" element={<PredictPage />} />
        <Route path="/train" element={<TrainPage />} />
      </Routes>
    </Router>
  );
}


