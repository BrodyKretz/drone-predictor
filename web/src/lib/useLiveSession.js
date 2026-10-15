import { useCallback, useEffect, useRef, useState } from "react";
import { identify } from "../api.js";
import { MIN_WINDOW_SAMPLES, decodeFile, decodeUrl, playBuffer, startMic } from "./audioSession.js";

const HISTORY_LENGTH = 40;

function knownSpec(known) {
  const spec = {};
  for (const [key, value] of Object.entries(known)) if (value !== "") spec[key] = Number(value);
  return Object.keys(spec).length ? spec : null;
}

function appendHistory(history, report) {
  const next = { ...history };
  for (const [key, v] of Object.entries(report.variables)) {
    const point = { median: v.median, low: v.interval_low, high: v.interval_high };
    next[key] = [...(history[key] || []), point].slice(-HISTORY_LENGTH);
  }
  return next;
}

export function useLiveSession(known) {
  const [mode, setMode] = useState("idle");
  const [source, setSource] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState({});
  const [error, setError] = useState(null);
  const session = useRef(null);
  const sessionId = useRef(0);
  const inflight = useRef(false);
  const pendingFinal = useRef(null);
  const knownRef = useRef(known);
  knownRef.current = known;

  const analyze = useCallback(async function run(wav, id, final = false) {
    if (!wav || id !== sessionId.current) return;
    if (inflight.current) {
      if (final) pendingFinal.current = { wav, id };
      return;
    }
    inflight.current = true;
    try {
      const report = await identify(wav, knownSpec(knownRef.current));
      if (id !== sessionId.current) return;
      setResult(report);
      setError(null);
      setHistory((h) => appendHistory(h, report));
    } catch (err) {
      if (id === sessionId.current) setError(`Analysis failed: ${err.message}`);
    } finally {
      inflight.current = false;
      const next = pendingFinal.current;
      pendingFinal.current = null;
      if (next) run(next.wav, next.id, true);
    }
  }, []);

  const begin = useCallback((nextMode, nextSource) => {
    const old = session.current;
    session.current = null;
    sessionId.current += 1;
    old?.stop();
    setMode(nextMode);
    setSource(nextSource);
    setResult(null);
    setHistory({});
    setError(null);
    return sessionId.current;
  }, []);

  const play = useCallback((buffer, label, sampleId = null) => {
    const id = begin("playing", { label, sampleId, duration: buffer.duration });
    session.current = playBuffer(buffer, {
      onWindow: (wav) => analyze(wav, id),
      onEnd: (wav) => {
        if (id !== sessionId.current) return;
        setMode("done");
        if (wav) {
          analyze(wav, id, true);
        } else {
          const seconds = (MIN_WINDOW_SAMPLES / buffer.sampleRate).toFixed(1);
          setError(`Augur needs at least ${seconds} s of audio to identify a drone.`);
        }
      },
    });
  }, [analyze, begin]);

  const playSample = useCallback(async (sample) => {
    try {
      play(await decodeUrl(`/templates/${sample.id}.wav`), `${sample.label} sample (simulated)`, sample.id);
    } catch (err) {
      setError(`Couldn't load the ${sample.label} sample: ${err.message}`);
    }
  }, [play]);

  const playFile = useCallback(async (file) => {
    try {
      play(await decodeFile(file), file.name);
    } catch {
      setError(`Couldn't decode ${file.name}. Try a WAV, MP3 or OGG file.`);
    }
  }, [play]);

  const record = useCallback(async () => {
    const id = begin("recording", { label: "Microphone", sampleId: null, duration: 0 });
    let mic;
    try {
      mic = await startMic({ onWindow: (wav) => analyze(wav, id) });
    } catch (err) {
      if (id === sessionId.current) {
        setMode("idle");
        setError(`Microphone unavailable: ${err.message}`);
      }
      return;
    }
    if (id !== sessionId.current) mic.stop();
    else session.current = mic;
  }, [analyze, begin]);

  const stop = useCallback(() => {
    if (!session.current) return;
    session.current.stop();
    setMode("done");
  }, []);

  const clear = useCallback(() => {
    begin("idle", null);
  }, [begin]);

  const elapsed = useCallback(() => (session.current ? session.current.elapsed() : null), []);

  useEffect(() => () => session.current?.stop(), []);

  return { mode, source, result, history, error, playSample, playFile, record, stop, clear, elapsed };
}
