import { useCallback, useEffect, useRef, useState } from "react";
import { identify } from "../api.js";
import {
  MIN_WINDOW_SAMPLES, decodeFile, decodeUrl, playBuffer, resumeAudio, startMic,
} from "./audioSession.js";

const HISTORY_LENGTH = 40;
const AUDIO_BLOCKED =
  "Your browser blocked audio until you interact with the page. Click a sample or Choose file to start.";
const NO_FACTS = Object.freeze({});

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
  const [analysis, setAnalysis] = useState(null);
  const [history, setHistory] = useState({});
  const [error, setError] = useState(null);
  const session = useRef(null);
  const sessionId = useRef(0);
  const inputRequest = useRef(0);
  const inflightSession = useRef(null);
  const pendingFinal = useRef(null);
  const finalWindow = useRef(null);
  const knownRef = useRef(known);
  knownRef.current = known;
  const factsKey = JSON.stringify(knownSpec(known));

  const analyze = useCallback(async function run(wav, id, final = false) {
    if (!wav || id !== sessionId.current) return;
    if (inflightSession.current === id) {
      if (final) pendingFinal.current = { wav, id };
      return;
    }
    inflightSession.current = id;
    const spec = knownSpec(knownRef.current);
    try {
      const report = await identify(wav, spec);
      if (id !== sessionId.current) return;
      setAnalysis({ report, facts: spec ?? NO_FACTS });
      setError(null);
      setHistory((h) => appendHistory(h, report));
    } catch (err) {
      if (id === sessionId.current) setError(`Analysis failed: ${err.message}`);
    } finally {
      if (inflightSession.current === id) {
        inflightSession.current = null;
        const next = pendingFinal.current;
        pendingFinal.current = null;
        if (next) run(next.wav, next.id, true);
      }
    }
  }, []);

  const begin = useCallback((nextMode, nextSource) => {
    const old = session.current;
    session.current = null;
    sessionId.current += 1;
    finalWindow.current = null;
    old?.stop();
    setMode(nextMode);
    setSource(nextSource);
    setAnalysis(null);
    setHistory({});
    setError(null);
    return sessionId.current;
  }, []);

  const endHandler = useCallback((id) => (wav, sampleRate) => {
    if (id !== sessionId.current) return;
    setMode("done");
    if (wav) {
      finalWindow.current = { wav, id };
      analyze(wav, id, true);
    } else {
      const seconds = (MIN_WINDOW_SAMPLES / sampleRate).toFixed(1);
      setError(`Augur needs at least ${seconds} s of audio to identify a drone.`);
    }
  }, [analyze]);

  const play = useCallback((buffer, label, sampleId) => {
    const id = begin("playing", { label, sampleId, duration: buffer.duration });
    session.current = playBuffer(buffer, {
      onWindow: (wav) => analyze(wav, id),
      onEnd: endHandler(id),
    });
  }, [analyze, begin, endHandler]);

  const playDecoded = useCallback(async ({ decode, decodeError, label, sampleId = null }) => {
    const request = ++inputRequest.current;
    const audioReady = resumeAudio();
    let buffer;
    try {
      buffer = await decode();
    } catch (err) {
      if (request === inputRequest.current) setError(decodeError(err));
      return;
    }
    if (request !== inputRequest.current) return;
    const running = await audioReady;
    if (request !== inputRequest.current) return;
    if (running) play(buffer, label, sampleId);
    else setError(AUDIO_BLOCKED);
  }, [play]);

  const playSample = useCallback((sample) => playDecoded({
    decode: () => decodeUrl(`/templates/${sample.id}.wav`),
    decodeError: (err) => `Couldn't load the ${sample.label} sample: ${err.message}`,
    label: `${sample.label} sample (simulated)`,
    sampleId: sample.id,
  }), [playDecoded]);

  const playFile = useCallback((file) => playDecoded({
    decode: () => decodeFile(file),
    decodeError: () => `Couldn't decode ${file.name}. Try a WAV, MP3 or OGG file.`,
    label: file.name,
  }), [playDecoded]);

  const record = useCallback(async () => {
    const request = ++inputRequest.current;
    const running = await resumeAudio();
    if (request !== inputRequest.current) return;
    if (!running) {
      setError(AUDIO_BLOCKED);
      return;
    }
    const id = begin("recording", { label: "Microphone", sampleId: null, duration: 0 });
    let mic;
    try {
      mic = await startMic({ onWindow: (wav) => analyze(wav, id), onEnd: endHandler(id) });
    } catch (err) {
      if (id === sessionId.current) {
        setMode("idle");
        setError(`Microphone unavailable: ${err.message}`);
      }
      return;
    }
    if (id !== sessionId.current) mic.stop();
    else session.current = mic;
  }, [analyze, begin, endHandler]);

  const stop = useCallback(() => {
    inputRequest.current += 1;
    if (session.current) {
      session.current.stop();
      setMode("done");
    } else {
      sessionId.current += 1;
      setMode("idle");
    }
  }, []);

  const elapsed = useCallback(() => (session.current ? session.current.elapsed() : null), []);

  // a live session picks new facts up on its next tick; only a finished one needs a re-run
  useEffect(() => {
    const last = finalWindow.current;
    if (last && last.id === sessionId.current) analyze(last.wav, last.id, true);
  }, [factsKey, analyze]);

  useEffect(() => () => {
    inputRequest.current += 1;
    sessionId.current += 1;
    session.current?.stop();
  }, []);

  return {
    mode,
    source,
    result: analysis ? analysis.report : null,
    resultFacts: analysis ? analysis.facts : NO_FACTS,
    history,
    error,
    playSample,
    playFile,
    record,
    stop,
    elapsed,
  };
}
