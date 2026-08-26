import React, { useState, useEffect, useRef } from 'react';
import { runBenchmarkTrain } from '../api';

const OPTIMIZER_COLORS = {
  SGD: '#ef4444',
  SGDMomentum: '#f97316',
  Momentum: '#f97316',
  NAG: '#eab308',
  AdaGrad: '#22c55e',
  RMSProp: '#3b82f6',
  Adam: '#a855f7',
  AdamW: '#b45309',
};

const ALL_OPTIMIZERS = ['SGD', 'SGDMomentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW'];

const LOG_LR_OPTIONS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];

export default function PartBDashboard({ appMode }) {
  const isSimple = appMode === 'Simple mode';

  // Control states
  const [selectedOpts, setSelectedOpts] = useState(['SGD', 'SGDMomentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam', 'AdamW']);
  const [lr, setLr] = useState(0.01);
  const [epochs, setEpochs] = useState(100);
  const [batchChoice, setBatchChoice] = useState('32');
  const [beta, setBeta] = useState(0.9);
  const [beta1, setBeta1] = useState(0.9);
  const [beta2, setBeta2] = useState(0.999);
  const [weightDecay, setWeightDecay] = useState(0.001);

  // Training & modal states
  const [isTraining, setIsTraining] = useState(false);
  const [resultsData, setResultsData] = useState(null);
  const [showModal, setShowModal] = useState(false);

  // Canvas refs
  const trainLossCanvasRef = useRef(null);
  const testLossCanvasRef = useRef(null);
  const testAccCanvasRef = useRef(null);
  const effLrCanvasRef = useRef(null);

  const toggleOpt = name => {
    if (selectedOpts.includes(name)) {
      setSelectedOpts(selectedOpts.filter(o => o !== name));
    } else {
      setSelectedOpts([...selectedOpts, name]);
    }
  };

  const handleTrain = async () => {
    if (selectedOpts.length === 0) return;
    setIsTraining(true);
    try {
      const bs = batchChoice === 'Full Batch' ? 455 : parseInt(batchChoice);
      const res = await runBenchmarkTrain({
        optimizers: selectedOpts,
        lr,
        epochs,
        batch_size: bs,
        beta,
        beta1,
        beta2,
        weight_decay: weightDecay,
      });
      setResultsData(res);
    } catch (err) {
      console.error("Training error:", err);
    } finally {
      setIsTraining(false);
    }
  };

  // Generic Line Chart Renderer for HTML5 Canvas
  const drawChart = (canvas, seriesData, title, yLabel, isLog = false, isPercent = false) => {
    if (!canvas || !seriesData) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#1A2029';
    ctx.fillRect(0, 0, width, height);

    // Title & Grid
    ctx.fillStyle = '#E7EAEE';
    ctx.font = '11px "Space Grotesk", sans-serif';
    ctx.fillText(title, 35, 18);

    ctx.strokeStyle = '#2E3742';
    ctx.lineWidth = 0.5;
    for (let y = 0.2; y < 1.0; y += 0.2) {
      ctx.beginPath();
      ctx.moveTo(35, y * height);
      ctx.lineTo(width - 10, y * height);
      ctx.stroke();
    }

    const maxEpoch = epochs;
    const toX = ep => 35 + ((ep - 1) / Math.max(1, maxEpoch - 1)) * (width - 45);

    // Collect all values to calculate Y min/max
    let allVals = [];
    Object.keys(seriesData).forEach(opt => {
      if (seriesData[opt]) {
        allVals = allVals.concat(seriesData[opt]);
      }
    });

    if (allVals.length === 0) return;

    let minY = Math.min(...allVals);
    let maxY = Math.max(...allVals);

    if (isPercent) {
      minY = Math.min(50, minY);
      maxY = 100;
    }

    const toY = val => {
      if (isLog) {
        const logVal = Math.log10(Math.max(1e-6, val));
        const logMin = Math.log10(Math.max(1e-6, minY));
        const logMax = Math.log10(Math.max(1e-5, maxY));
        const norm = (logVal - logMin) / (logMax - logMin || 1);
        return height - 20 - norm * (height - 40);
      } else {
        const norm = (val - minY) / (maxY - minY || 1);
        return height - 20 - norm * (height - 40);
      }
    };

    // Draw lines for each optimizer
    Object.keys(seriesData).forEach(optName => {
      const vals = seriesData[optName];
      if (!vals || vals.length === 0) return;

      const color = OPTIMIZER_COLORS[optName] || '#ffffff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();

      vals.forEach((val, idx) => {
        const x = toX(idx + 1);
        const y = toY(val);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  };

  // Redraw charts when resultsData changes
  useEffect(() => {
    if (!resultsData) return;

    const res = resultsData.results;
    const trainLossSeries = {};
    const testLossSeries = {};
    const testAccSeries = {};
    const effLrSeries = {};

    Object.keys(res).forEach(opt => {
      trainLossSeries[opt] = res[opt].train_loss;
      testLossSeries[opt] = res[opt].test_loss;
      testAccSeries[opt] = res[opt].test_acc;
      if (res[opt].eff_lr) effLrSeries[opt] = res[opt].eff_lr;
    });

    drawChart(trainLossCanvasRef.current, trainLossSeries, 'Training Loss vs Epoch', 'Loss', true);
    drawChart(testLossCanvasRef.current, testLossSeries, 'Validation / Test Loss vs Epoch', 'Loss', true);
    drawChart(testAccCanvasRef.current, testAccSeries, 'Test Accuracy (%) vs Epoch', 'Acc (%)', false, true);
    if (!isSimple) {
      drawChart(effLrCanvasRef.current, effLrSeries, 'Effective Learning Rate (\eta_{eff}) vs Epoch', '\eta_{eff}', true);
    }
  }, [resultsData, isSimple]);

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Sidebar Controls */}
      <aside className="col-span-3 bezel-card p-4 space-y-4">
        <div className="label-smallcaps">Part B Settings</div>

        <div>
          <label className="text-xs text-[#8B96A3] block mb-1">Optimizers to Benchmark</label>
          <div className="flex flex-wrap gap-1.5">
            {ALL_OPTIMIZERS.map(opt => {
              const active = selectedOpts.includes(opt);
              return (
                <button
                  key={opt}
                  onClick={() => toggleOpt(opt)}
                  style={{ backgroundColor: active ? OPTIMIZER_COLORS[opt] : '#1A2029', color: active ? '#10141A' : '#8B96A3' }}
                  className="px-2 py-0.5 text-[11px] font-bold rounded border border-[#2E3742] transition"
                >
                  {opt}
                </button>
              );
            })}
          </div>
        </div>

        <div>
          <label className="text-xs text-[#8B96A3] block mb-1">Learning Rate (\eta): {lr}</label>
          <select
            value={lr}
            onChange={e => setLr(parseFloat(e.target.value))}
            className="w-full bg-[#10141A] border border-[#2E3742] rounded px-2 py-1.5 text-xs text-white font-mono-num"
          >
            {LOG_LR_OPTIONS.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs text-[#8B96A3] block mb-1">Number of Epochs: {epochs}</label>
          <input
            type="number"
            min="10"
            max="500"
            step="10"
            value={epochs}
            onChange={e => setEpochs(parseInt(e.target.value) || 10)}
            className="w-full bg-[#10141A] border border-[#2E3742] rounded px-2 py-1.5 text-xs text-white font-mono-num"
          />
        </div>

        <div>
          <label className="text-xs text-[#8B96A3] block mb-1">Batch Size</label>
          <select
            value={batchChoice}
            onChange={e => setBatchChoice(e.target.value)}
            className="w-full bg-[#10141A] border border-[#2E3742] rounded px-2 py-1.5 text-xs text-white font-mono-num"
          >
            <option value="32">32</option>
            <option value="64">64</option>
            <option value="128">128</option>
            <option value="Full Batch">Full Batch (455)</option>
          </select>
        </div>

        {!isSimple && (
          <div className="space-y-3 pt-2 border-t border-[#2E3742]">
            <div className="label-smallcaps">Hyperparameters</div>
            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\beta (Momentum / RMSProp): {beta}</label>
              <input
                type="range"
                min="0.0"
                max="0.999"
                step="0.01"
                value={beta}
                onChange={e => setBeta(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\beta_1 (Adam / AdamW): {beta1}</label>
              <input
                type="range"
                min="0.0"
                max="0.999"
                step="0.01"
                value={beta1}
                onChange={e => setBeta1(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\beta_2 (Adam / AdamW): {beta2}</label>
              <input
                type="range"
                min="0.0"
                max="0.999"
                step="0.001"
                value={beta2}
                onChange={e => setBeta2(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-xs text-[#8B96A3] block mb-1">\lambda (AdamW Weight Decay): {weightDecay}</label>
              <input
                type="range"
                min="0.0"
                max="0.1"
                step="0.001"
                value={weightDecay}
                onChange={e => setWeightDecay(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        )}

        <button
          onClick={handleTrain}
          disabled={isTraining}
          className="btn-primary w-full py-2 text-xs flex items-center justify-center gap-2 mt-4"
        >
          {isTraining ? '⏳ Training Benchmark...' : '🚀 Train Optimizers'}
        </button>

        <button
          onClick={() => setShowModal(true)}
          className="btn-secondary w-full py-1.5 text-xs mt-2"
        >
          📖 How to Use Part B
        </button>
      </aside>

      {/* Main Content Area */}
      <section className="col-span-9 space-y-6">
        {/* Dataset Metadata Cards */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bezel-card p-3">
            <div className="label-smallcaps">Total Features</div>
            <div className="text-lg font-bold font-mono-num text-[#E3A23B]">30</div>
          </div>
          <div className="bezel-card p-3">
            <div className="label-smallcaps">Train Samples</div>
            <div className="text-lg font-bold font-mono-num text-[#E3A23B]">455</div>
          </div>
          <div className="bezel-card p-3">
            <div className="label-smallcaps">Test Samples</div>
            <div className="text-lg font-bold font-mono-num text-[#E3A23B]">114</div>
          </div>
          <div className="bezel-card p-3">
            <div className="label-smallcaps">Train / Test Split</div>
            <div className="text-lg font-bold font-mono-num text-[#E3A23B]">80% / 20%</div>
          </div>
        </div>

        {/* Live Training Charts */}
        <div className="bezel-card p-4 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center justify-between">
            <span>📈 Live Benchmark Training Charts</span>
            {isTraining && <span className="text-xs text-[#E3A23B] font-mono-num animate-pulse">Training in Progress...</span>}
          </h3>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <canvas ref={trainLossCanvasRef} width={380} height={220} className="border border-[#2E3742] rounded w-full bg-[#10141A]" />
            </div>
            <div>
              <canvas ref={testLossCanvasRef} width={380} height={220} className="border border-[#2E3742] rounded w-full bg-[#10141A]" />
            </div>
            <div>
              <canvas ref={testAccCanvasRef} width={380} height={220} className="border border-[#2E3742] rounded w-full bg-[#10141A]" />
            </div>
            {!isSimple && (
              <div>
                <canvas ref={effLrCanvasRef} width={380} height={220} className="border border-[#2E3742] rounded w-full bg-[#10141A]" />
              </div>
            )}
          </div>
        </div>

        {/* Auto-Generated Comparison Summary Table */}
        {resultsData && resultsData.summary_table && (
          <div className="bezel-card p-4 space-y-3">
            <h3 className="text-sm font-bold text-white">📊 Benchmark Summary Results Table</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono-num">
                <thead>
                  <tr className="border-b border-[#2E3742] text-[#8B96A3] font-heading uppercase text-[11px]">
                    <th className="py-2 px-3">Optimizer</th>
                    <th className="py-2 px-3">Final Train Loss</th>
                    <th className="py-2 px-3">Final Test Loss</th>
                    <th className="py-2 px-3">Train Acc (%)</th>
                    <th className="py-2 px-3">Test Acc (%)</th>
                    <th className="py-2 px-3">Convergence Epoch</th>
                  </tr>
                </thead>
                <tbody>
                  {resultsData.summary_table.map(row => (
                    <tr key={row.optimizer} className="border-b border-[#2E3742]/50 hover:bg-[#13171F]">
                      <td className="py-2.5 px-3 font-bold" style={{ color: OPTIMIZER_COLORS[row.optimizer] }}>
                        {row.optimizer}
                      </td>
                      <td className="py-2.5 px-3 text-[#E7EAEE]">{row.final_train_loss.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-[#E7EAEE]">{row.final_test_loss.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-[#E7EAEE]">{row.final_train_acc.toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-[#E3A23B] font-bold">{row.final_test_acc.toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-[#E7EAEE]">{row.convergence_epoch}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* How to Use Modal Guide */}
      {showModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#1A2029] border border-[#2E3742] rounded-md max-w-xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex justify-between items-center border-b border-[#2E3742] pb-3">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                📖 How to Use Part B (Neural Network Benchmark)
              </h2>
              <button
                onClick={() => setShowModal(false)}
                className="text-[#8B96A3] hover:text-white font-bold text-lg"
              >
                ✕
              </button>
            </div>

            <div className="text-xs text-[#E7EAEE] space-y-3 leading-relaxed">
              <p>
                <strong>Dataset:</strong> Benchmark 7 optimization algorithms (SGD, SGDMomentum, NAG, AdaGrad, RMSProp, Adam, AdamW)
                on a 3-layer Multi-Layer Perceptron trained on the <strong>Breast Cancer Wisconsin Dataset</strong> (30 input features, 569 samples).
              </p>
              <p>
                <strong>Architecture:</strong> Input(30) &rarr; Dense(16) &rarr; ReLU &rarr; Dense(8) &rarr; ReLU &rarr; Dense(1) &rarr; Sigmoid with Binary Cross-Entropy loss.
              </p>
              <p>
                <strong>Effective Learning Rate:</strong> Adaptive algorithms (AdaGrad, RMSProp, Adam, AdamW) scale step sizes per parameter based on accumulated squared gradients: &eta;<sub>eff</sub> = &eta; / (&radic;v<sub>t</sub> + &epsilon;).
              </p>
              <p>
                <strong>Convergence Epoch:</strong> Automatically identifies the first epoch where validation loss reaches within 1% of its final value and remains within 1% for all subsequent epochs.
              </p>
            </div>

            <div className="pt-2 border-t border-[#2E3742] flex justify-end">
              <button
                onClick={() => setShowModal(false)}
                className="btn-primary px-4 py-1.5 text-xs"
              >
                Got It
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
