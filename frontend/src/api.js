/**
 * API client module for FastAPI backend interaction.
 */

const API_BASE_URL = 'http://localhost:8000';

export async function fetchTrajectoryData(payload) {
  const response = await fetch(`${API_BASE_URL}/api/simulate-2d`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  return response.json();
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
