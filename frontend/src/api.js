/**
 * API client module for FastAPI backend interaction with offline client fallback.
 */

const API_BASE_URL = 'http://localhost:8000';

export async function fetchTrajectoryData(payload) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/simulate-2d`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (response.ok) {
      return await response.json();
    }
  } catch (err) {
    console.warn('Backend API unavailable. Using resilient client-side simulation engine.', err);
  }
  return simulate2DClientFallback(payload);
}

export async function runBenchmarkTrain(payload) {
  const response = await fetch(`${API_BASE_URL}/api/train-nn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  return response.json();
}

function simulate2DClientFallback(payload) {
  const {
    surface_name = 'L2: c = 50 (default)',
    optimizers = ['SGD', 'Adam', 'AdamW'],
    x0 = 8.0,
    y0 = 8.0,
    lr = 0.01,
    beta = 0.9,
    beta1 = 0.9,
    beta2 = 0.999,
    weight_decay = 0.001,
    max_steps = 500,
  } = payload;

  let c = 50;
  if (surface_name.includes('c = 1000')) c = 1000;
  else if (surface_name.includes('c = 100')) c = 100;
  else if (surface_name.includes('c = 10')) c = 10;

  const func = (x, y) => x * x + c * y * y;
  const gradFn = (x, y) => [2 * x, 2 * c * y];

  const trajectories = {};
  const losses = {};
  const gradients = {};
  const deltas = {};
  const statuses = {};
  const final_statuses = {};
  const peaks = {};

  optimizers.forEach(name => {
    let x = x0, y = y0;
    let vx = 0, vy = 0;
    let mx = 0, my = 0;
    let Gx = 0, Gy = 0;

    const traj = [[x, y]];
    const lossList = [func(x, y)];
    const gradList = [gradFn(x, y)];
    const deltaList = [[0.0, 0.0]];
    const statusList = ['RUNNING'];
    let optStatus = 'RUNNING';
    let peakTheta = Math.max(Math.abs(x), Math.abs(y));

    for (let step = 1; step <= max_steps; step++) {
      let gx, gy;
      let dx = 0, dy = 0;

      if (name === 'NAG') {
        const peekX = x - beta * vx;
        const peekY = y - beta * vy;
        const peekGrad = gradFn(peekX, peekY);
        gx = peekGrad[0]; gy = peekGrad[1];
        vx = beta * vx + (1 - beta) * gx;
        vy = beta * vy + (1 - beta) * gy;
        dx = -lr * vx; dy = -lr * vy;
      } else {
        const g = gradFn(x, y);
        gx = g[0]; gy = g[1];

        if (name === 'SGD') {
          dx = -lr * gx; dy = -lr * gy;
        } else if (name === 'SGDMomentum' || name === 'Momentum') {
          vx = beta * vx + (1 - beta) * gx;
          vy = beta * vy + (1 - beta) * gy;
          dx = -lr * vx; dy = -lr * vy;
        } else if (name === 'AdaGrad') {
          Gx += gx * gx; Gy += gy * gy;
          dx = - (lr / (Math.sqrt(Gx) + 1e-8)) * gx;
          dy = - (lr / (Math.sqrt(Gy) + 1e-8)) * gy;
        } else if (name === 'RMSProp') {
          vx = beta * vx + (1 - beta) * gx * gx;
          vy = beta * vy + (1 - beta) * gy * gy;
          dx = - (lr / (Math.sqrt(vx) + 1e-8)) * gx;
          dy = - (lr / (Math.sqrt(vy) + 1e-8)) * gy;
        } else if (name === 'Adam') {
          mx = beta1 * mx + (1 - beta1) * gx;
          my = beta1 * my + (1 - beta1) * gy;
          vx = beta2 * vx + (1 - beta2) * gx * gx;
          vy = beta2 * vy + (1 - beta2) * gy * gy;
          const mHatX = mx / (1 - Math.pow(beta1, step));
          const mHatY = my / (1 - Math.pow(beta1, step));
          const vHatX = vx / (1 - Math.pow(beta2, step));
          const vHatY = vy / (1 - Math.pow(beta2, step));
          dx = - lr * mHatX / (Math.sqrt(vHatX) + 1e-8);
          dy = - lr * mHatY / (Math.sqrt(vHatY) + 1e-8);
        } else if (name === 'AdamW') {
          mx = beta1 * mx + (1 - beta1) * gx;
          my = beta1 * my + (1 - beta1) * gy;
          vx = beta2 * vx + (1 - beta2) * gx * gx;
          vy = beta2 * vy + (1 - beta2) * gy * gy;
          const mHatX = mx / (1 - Math.pow(beta1, step));
          const mHatY = my / (1 - Math.pow(beta1, step));
          const vHatX = vx / (1 - Math.pow(beta2, step));
          const vHatY = vy / (1 - Math.pow(beta2, step));
          dx = - lr * (mHatX / (Math.sqrt(vHatX) + 1e-8) + weight_decay * x);
          dy = - lr * (mHatY / (Math.sqrt(vHatY) + 1e-8) + weight_decay * y);
        }
      }

      x += dx; y += dy;
      const lVal = func(x, y);

      const isDiv = isNaN(x) || isNaN(y) || Math.abs(x) > 1e6 || Math.abs(y) > 1e6 || isNaN(lVal) || lVal > 1e6;

      const safeX = isNaN(x) || Math.abs(x) > 1e12 ? (x > 0 ? 1e12 : -1e12) : x;
      const safeY = isNaN(y) || Math.abs(y) > 1e12 ? (y > 0 ? 1e12 : -1e12) : y;
      const safeLoss = isNaN(lVal) || lVal > 1e12 ? 1e12 : lVal;

      traj.push([safeX, safeY]);
      lossList.push(safeLoss);
      gradList.push([gx, gy]);
      deltaList.push([dx, dy]);

      if (isDiv) {
        statusList.push('DIVERGED');
        optStatus = 'DIVERGED';
        break;
      } else if (safeLoss < 1e-6) {
        statusList.push('CONVERGED');
        optStatus = 'CONVERGED';
      } else {
        statusList.push('RUNNING');
      }

      peakTheta = Math.max(peakTheta, Math.abs(safeX), Math.abs(safeY));
    }

    trajectories[name] = traj;
    losses[name] = lossList;
    gradients[name] = gradList;
    deltas[name] = deltaList;
    statuses[name] = statusList;
    final_statuses[name] = optStatus;
    peaks[name] = peakTheta;
  });

  return { trajectories, losses, gradients, deltas, statuses, final_statuses, peaks };
}
