import { CLASS_NAMES } from './diseases';

export const MAX_FILE_BYTES = 10 * 1024 * 1024;

export function validateFile(file) {
  if (!file || !['image/jpeg', 'image/png'].includes(file.type) || !/\.(jpe?g|png)$/i.test(file.name)) {
    throw new Error('Choose a JPG or PNG image. HEIC and other formats are not supported.');
  }
  if (file.size > MAX_FILE_BYTES) throw new Error('This image exceeds 10 MB. Please choose a smaller image.');
  if (!file.size) throw new Error('This file is empty. Please choose another image.');
}

export function validatePrediction(data) {
  const probabilities = data?.probabilities;
  if (!probabilities || Object.keys(probabilities).length !== 4 ||
      !CLASS_NAMES.every(name => Number.isFinite(probabilities[name]) && probabilities[name] >= 0 && probabilities[name] <= 1) ||
      !CLASS_NAMES.includes(data.predicted_class) || !Number.isFinite(data.confidence) ||
      Math.abs(Object.values(probabilities).reduce((a, b) => a + b, 0) - 1) > 0.001 ||
      Math.abs(data.confidence - probabilities[data.predicted_class]) > 0.00001 ||
      data.confidence < Math.max(...Object.values(probabilities)) - 0.00001 ||
      typeof data.model_identifier !== 'string' || !data.model_identifier) {
    throw new Error('The server returned an invalid prediction. Please try again.');
  }
  return data;
}

export async function predict(file, signal) {
  const base = (import.meta.env.VITE_API_URL || '').trim().replace(/\/$/, '');
  if (!base) throw new Error('The prediction service is not configured. Set VITE_API_URL and rebuild the frontend.');
  if (!navigator.onLine) throw new Error('An Internet connection is required for AI analysis. Reconnect and try again.');
  const form = new FormData();
  form.append('file', file);
  let response;
  try { response = await fetch(`${base}/predict`, { method: 'POST', body: form, signal, cache: 'no-store' }); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Could not reach the prediction service. A free server may be waking up; please try again shortly.');
  }
  let data;
  try { data = await response.json(); }
  catch {
    if ([502, 503, 504].includes(response.status)) throw new Error('The prediction service may be starting or temporarily unavailable. Please wait a moment and try again.');
    throw new Error('The prediction service returned an unreadable response. Please try again.');
  }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Analysis failed. Please try again.');
  return validatePrediction(data);
}
