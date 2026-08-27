/**
 * Shared number formatting utilities for consistent numerical representation across the visualizer.
 */

export function fmtNumber(val, decimals = 4) {
  if (val == null || Number.isNaN(val)) return 'NaN';
  if (!Number.isFinite(val)) return val > 0 ? '∞' : '-∞';
  
  const absVal = Math.abs(val);
  if (absVal === 0) return (0).toFixed(decimals);
  
  if (absVal >= 1e4 || absVal < 1e-4) {
    return val.toExponential(3);
  }
  
  return val.toFixed(decimals);
}

export function fmtPos(x, y, decimals = 3) {
  if (x == null || y == null) return '(—, —)';
  return `(${fmtNumber(x, decimals)}, ${fmtNumber(y, decimals)})`;
}

export function fmtLoss(loss, decimals = 4) {
  if (loss == null || Number.isNaN(loss)) return 'NaN';
  if (!Number.isFinite(loss)) return loss > 0 ? '∞' : '-∞';
  
  const absLoss = Math.abs(loss);
  if (absLoss >= 1e4 || (absLoss < 1e-4 && absLoss > 0)) {
    return loss.toExponential(3);
  }
  return loss.toFixed(decimals);
}

export function fmtDelta(dx, dy, decimals = 4) {
  if (dx == null || dy == null) return 'Δ = (—, —)';
  return `Δx = ${fmtNumber(dx, decimals)}, Δy = ${fmtNumber(dy, decimals)}`;
}
