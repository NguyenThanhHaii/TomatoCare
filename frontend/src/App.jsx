import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, BookOpen, Camera, Check, CirclePlus, Image as ImageIcon, Info, Menu, Share2, X } from 'lucide-react';
import { predict, validateFile } from './api';
import { CLASS_NAMES, DISEASES } from './diseases';
import { LeafIllustration, LeafMark } from './Leaves';

export default function App() {
  const [screen, setScreen] = useState('upload');
  const [file, setFile] = useState(null);
  const [photo, setPhoto] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [dialog, setDialog] = useState(null);
  const upload = useRef(null);
  const camera = useRef(null);
  const controller = useRef(null);
  const requestId = useRef(0);
  const main = useRef(null);

  useEffect(() => {
    if (!file) { setPhoto(''); return; }
    const url = URL.createObjectURL(file);
    setPhoto(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => { main.current?.focus(); window.scrollTo(0, 0); }, [screen]);
  useEffect(() => () => controller.current?.abort(), []);

  function cancel() { requestId.current += 1; controller.current?.abort(); }
  function reset() { cancel(); setFile(null); setResult(null); setError(''); setScreen('upload'); }
  function select(event) {
    const chosen = event.target.files?.[0];
    event.target.value = '';
    if (!chosen) return;
    try { validateFile(chosen); cancel(); setFile(chosen); setResult(null); setError(''); setScreen('preview'); }
    catch (err) { setError(err.message); }
  }
  async function analyze() {
    if (!file) return;
    cancel();
    const id = requestId.current;
    const active = new AbortController();
    controller.current = active;
    setError(''); setScreen('analyzing');
    const timeout = setTimeout(() => active.abort('timeout'), 120000);
    try {
      const prediction = await predict(file, active.signal);
      if (id === requestId.current) { setResult(prediction); setScreen('result'); }
    } catch (err) {
      if (id === requestId.current) setError(active.signal.aborted ? 'The request timed out. The free server may be starting; please try again.' : err.message);
    } finally { clearTimeout(timeout); }
  }
  async function share() {
    if (!navigator.share || !result) return;
    try { await navigator.share({ title: 'TomatoCare', text: `${result.predicted_class}: ${(result.confidence * 100).toFixed(1)}% model confidence. Educational support, not a confirmed diagnosis.` }); }
    catch (err) { if (err.name !== 'AbortError') setError('Sharing was unavailable. Please try again.'); }
  }
  const disease = result ? DISEASES[result.predicted_class] : null;

  return <div className={`app-shell screen-${screen}`}>
    <input ref={upload} type="file" accept="image/jpeg,image/png" aria-label="Upload leaf image" onChange={select} hidden />
    <input ref={camera} type="file" accept="image/jpeg,image/png" capture="environment" aria-label="Take leaf photo" onChange={select} hidden />
    {screen !== 'analyzing' && <header className="app-header">
      {screen === 'upload' ? <><div className="brand"><span className="brand-icon"><LeafMark /></span>TomatoCare</div><button className="icon-button" aria-label="About TomatoCare" onClick={() => setDialog('about')}><Menu size={21} /></button></> :
        <><button className="icon-button" aria-label="Back" onClick={() => screen === 'result' ? setScreen('preview') : reset()}><ArrowLeft size={23} /></button><h1>{screen === 'preview' ? 'Preview Image' : 'Result'}</h1>{screen === 'result' && navigator.share ? <button className="icon-button" aria-label="Share result" onClick={share}><Share2 size={20}/></button> : <span className="header-spacer"/>}</>}
    </header>}
    <main ref={main} tabIndex={-1}>
      {screen === 'upload' && <section className="upload-screen">
        <h1>Tomato Leaf<br/>Disease Recognition</h1>
        <p className="intro">Upload or capture a photo of a tomato<br className="wide-break"/> leaf to identify the disease and get<br className="wide-break"/> care recommendations.</p>
        <LeafIllustration />
        <div className="actions"><button className="button primary" onClick={() => upload.current.click()}><ImageIcon size={24}/>Upload Image</button><button className="button secondary" onClick={() => camera.current.click()}><Camera size={25}/>Take Photo</button></div>
        <div className="support-card"><Info size={24}/><div><p>Supported format</p><small>JPG, PNG (max 10 MB)</small></div></div>
        <LeafIllustration decorative />
      </section>}
      {screen === 'preview' && <section className="preview-screen">
        <img className="preview-image" src={photo} alt="Your selected tomato leaf"/>
        <div className="quality-card"><h2>Image looks good?</h2><p className="quality-note">Before analyzing, please check:</p><ul>{['Leaf is clear and in focus', 'Main part of the leaf is visible', 'Good lighting condition'].map(text => <li key={text}><span className="check-icon"><Check size={14}/></span>{text}</li>)}</ul></div>
        <div className="actions preview-actions"><button className="button primary" onClick={analyze}>Analyze Image<ArrowRight size={18}/></button><button className="button secondary" onClick={() => upload.current.click()}>Choose Another Image</button></div>
      </section>}
      {screen === 'analyzing' && <section className="analyzing-screen">
        <LeafMark className="analyzing-leaf"/><h1>Analyzing Image</h1>
        <div className={`scan-frame ${error ? 'scan-paused' : ''}`}><img src={photo} alt="Leaf being analyzed"/><div className="scan-grid"/>{!error && <div className="scan-line"/>}<i className="corner tl"/><i className="corner tr"/><i className="corner bl"/><i className="corner br"/></div>
        <h2>{error ? 'Analysis interrupted' : 'Analyzing...'}</h2><p className="intro">Our AI model is identifying the disease<br/>from your tomato leaf image.</p>
        {!error && <div className="processing" role="status" aria-live="polite"><div className="indeterminate"/><p>Waiting for the prediction service</p><ul className="process-steps"><li className="active-step">Request in progress</li><li>Model analysis and results pending</li></ul><small>Free hosting may need a moment to wake up.</small></div>}
        <div className="actions"><button className="button secondary" onClick={() => {cancel(); setError(''); setScreen('preview');}}>Back to Preview</button></div>
      </section>}
      {screen === 'result' && result && <section className="result-screen">
        <div className={`detection-card ${result.predicted_class === 'Healthy' ? 'healthy-card' : ''}`}><img src={photo} alt="Analyzed tomato leaf"/><div><span className="disease-label">{result.predicted_class === 'Healthy' ? 'Model Prediction' : 'Detected Disease'}</span><h2>{result.predicted_class}</h2><p>Confidence: {(result.confidence * 100).toFixed(1)}%</p></div></div>
        <div className="predictions-card"><h2>All Predictions</h2><ul>{[...CLASS_NAMES].sort((a,b) => result.probabilities[b] - result.probabilities[a]).map((name, i) => <li key={name}><span>{name}</span><div className="probability-track" role="meter" aria-label={`${name} probability`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={result.probabilities[name] * 100}><span className={`probability-fill color-${i}`} style={{width: `${result.probabilities[name] * 100}%`}}/></div><span className="probability-value">{(result.probabilities[name] * 100).toFixed(1)}%</span></li>)}</ul></div>
        <div className="recommendations-card"><h2><LeafMark/>Recommendations</h2><ul>{disease.recommendations.map(text => <li key={text}>{text}</li>)}</ul></div>
        <p className="result-note">Model guidance, not a confirmed diagnosis.</p>
        <div className="actions"><button className="button primary" onClick={reset}><CirclePlus size={21}/>Analyze Another Image</button><button className="button secondary info-button" onClick={() => setDialog('disease')}><BookOpen size={22}/>View Disease Information</button></div>
      </section>}
      {error && <div className="error-card" role="alert"><p>{error}</p>{screen === 'analyzing' && <button className="button primary" onClick={analyze}>Try Again</button>}<button className="dismiss-error" onClick={() => setError('')} aria-label="Dismiss error" hidden={screen === 'analyzing'}><X size={16}/></button></div>}
    </main>
    {dialog && <InfoDialog title={dialog === 'disease' ? disease.title : 'About TomatoCare'} onClose={() => setDialog(null)}>
      {dialog === 'disease' ? <><p>{disease.description}</p><ul>{disease.recommendations.map(text => <li key={text}>{text}</li>)}</ul><p>Supported classes: Healthy, Early Blight, Late Blight and Leaf Mold. Other diseases are outside this model’s scope.</p></> : <><p>TomatoCare uses the UniRepLKNet-A baseline to support tomato leaf care. DINOv3 integration is planned for a later phase.</p><p>Photos are sent to the prediction service for processing and are not stored permanently. AI analysis requires Internet; installing the app does not enable offline inference.</p><p>Confidence is a model probability, not diagnostic certainty. Consult a local plant specialist when needed.</p></>}
    </InfoDialog>}
  </div>;
}

function InfoDialog({title, children, onClose}) {
  const ref = useRef(null);
  useEffect(() => { ref.current.showModal(); return () => ref.current?.close(); }, []);
  return <dialog ref={ref} className="info-dialog" aria-labelledby="information-title" onCancel={onClose}><div className="dialog-title"><h2 id="information-title">{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close information"><X/></button></div>{children}<button className="button primary" onClick={onClose}>Close</button></dialog>;
}
