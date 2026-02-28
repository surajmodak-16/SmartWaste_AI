import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import "./App.css";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [records, setRecords] = useState([]);
  const [activeTab, setActiveTab] = useState("dashboard");

  const fetchRecords = async () => {
    try {
      const res = await axios.get(`${API_BASE}/records`);
      setRecords(res.data);
    } catch (error) {
      console.error("Error fetching records:", error);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  const wasteCounts = records.reduce((acc, rec) => {
    acc[rec.waste_type] = (acc[rec.waste_type] || 0) + 1;
    return acc;
  }, {});

  const chartData = {
    labels: Object.keys(wasteCounts),
    datasets: [
      {
        data: Object.values(wasteCounts),
        backgroundColor: ["#00ff9f", "#007bff", "#ffd000", "#ff4d00", "#bc00ff"],
      },
    ],
  };

  const recyclableCount = records.filter((r) =>
    r.route?.toLowerCase().includes("recycling")
  ).length;

  const avgCarbon =
    records.length > 0
      ? Math.round(
          records.reduce((a, r) => {
            if (r.carbon_impact === "High") return a + 3;
            if (r.carbon_impact === "Medium") return a + 2;
            return a + 1;
          }, 0) / records.length
        )
      : 0;

  const mostCommonWaste =
    records.length > 0
      ? [...records.reduce((m, r) => {
          m.set(r.waste_type, (m.get(r.waste_type) || 0) + 1);
          return m;
        }, new Map())].sort((a, b) => b[1] - a[1])[0][0]
      : "-";

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2 className="logo">♻ SmartWaste</h2>

        <button
          className={`nav-btn ${activeTab === "dashboard" ? "active" : ""}`}
          onClick={() => setActiveTab("dashboard")}
        >
          📊 Dashboard
        </button>

        <button
          className={`nav-btn ${activeTab === "webcam" ? "active" : ""}`}
          onClick={() => setActiveTab("webcam")}
        >
          📷 Webcam Scanner
        </button>

        <button
          className={`nav-btn ${activeTab === "upload" ? "active" : ""}`}
          onClick={() => setActiveTab("upload")}
        >
          🖼 Upload Scanner
        </button>
      </aside>

      <main className="main">
        {activeTab === "dashboard" && (
          <div className="container">
            <h1 className="dashboard-title">Smart Waste Dashboard</h1>

            <div className="stats-container">
              <div className="stat-card">
                <h3>Total Scans</h3>
                <p>{records.length}</p>
              </div>

              <div className="stat-card">
                <h3>Recyclable %</h3>
                <p>
                  {records.length > 0
                    ? Math.round((recyclableCount / records.length) * 100) + "%"
                    : "0%"}
                </p>
              </div>

              <div className="stat-card">
                <h3>Avg Carbon Score</h3>
                <p>{avgCarbon}</p>
              </div>

              <div className="stat-card">
                <h3>Most Common Waste</h3>
                <p>{mostCommonWaste}</p>
              </div>
            </div>

            <div className="chart-container">
              <Pie data={chartData} />
            </div>

            <div className="table-container">
              <h2 className="section-title">Recent Classifications</h2>
              <table>
                <thead>
                  <tr>
                    <th>Waste</th>
                    <th>Calorific</th>
                    <th>Carbon</th>
                    <th>Route</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((rec) => (
                    <tr key={rec._id}>
                      <td>{rec.waste_type}</td>
                      <td>{rec.calorific_value}</td>
                      <td>{rec.carbon_impact}</td>
                      <td>{rec.route}</td>
                      <td>{rec.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <button onClick={fetchRecords}>🔄 Refresh</button>
          </div>
        )}

        {activeTab === "webcam" && <WebcamView onClassified={fetchRecords} />}

        {activeTab === "upload" && <UploadScanner onClassified={fetchRecords} />}
      </main>
    </div>
  );
}

/* ------------ Webcam Scanner ------------ */

function WebcamView({ onClassified }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const startCamera = async () => {
    setError("");
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
    } catch (err) {
      console.error(err);
      setError("Camera access denied!");
    }
  };

  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    videoRef.current.srcObject = null;
    setStream(null);
  };

  const captureAndClassify = async () => {
    setLoading(true);
    try {
      const canvas = canvasRef.current;
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext("2d").drawImage(videoRef.current, 0, 0);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg")
      );

      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");

      const res = await axios.post(`${API_BASE}/classify`, formData);
      setPrediction(res.data);
      onClassified();
    } catch (e) {
      setError("Classification failed!");
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1 className="dashboard-title">Webcam Scanner</h1>

      <video ref={videoRef} autoPlay className="webcam-video" />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div className="controls">
        {!stream ? (
          <button onClick={startCamera}>🎥 Start Camera</button>
        ) : (
          <button onClick={stopCamera}>🛑 Stop</button>
        )}
        <button onClick={captureAndClassify} disabled={!stream || loading}>
          {loading ? "Classifying..." : "📸 Capture & Classify"}
        </button>
      </div>

      {error && <p className="error-msg">{error}</p>}

      {prediction && (
        <div className="prediction-card">
          <h3>Prediction</h3>
          <p>Waste: {prediction.waste_type}</p>
          <p>Route: {prediction.route}</p>
          <p>Carbon: {prediction.carbon_impact}</p>
        </div>
      )}
    </div>
  );
}

/* ------------ Upload Scanner ------------ */

function UploadScanner({ onClassified }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const img = e.target.files[0];
    if (!img) return;
    setFile(img);
    setPreview(URL.createObjectURL(img));
  };

  const classifyUploaded = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post(`${API_BASE}/classify`, formData);
    setResult(res.data);
    onClassified();
  };

  return (
    <div className="container">
      <h1 className="dashboard-title">Upload Waste Image</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {preview && (
        <>
          <img src={preview} alt="Preview" className="preview-img" />
          <button onClick={classifyUploaded}>🚀 Classify</button>
        </>
      )}

      {result && (
        <div className="prediction-card">
          <h3>Prediction</h3>
          <p>Waste: {result.waste_type}</p>
          <p>Route: {result.route}</p>
          <p>Carbon: {result.carbon_impact}</p>
        </div>
      )}
    </div>
  );
}


