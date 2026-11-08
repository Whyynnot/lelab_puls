import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { getInferenceStatus, InferenceStatus, stopInferenceAndWait } from "@/lib/inferenceApi";

/** Poll only while configuring a run. Paused runs still own the hardware. */
export function useInferenceAvailability(open: boolean) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [checking, setChecking] = useState(true);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState("");
  const revision = useRef(0);
  const stoppingRef = useRef(false);

  useEffect(() => {
    ++revision.current;
    setStatus(null);
    setChecking(open);
    setError("");
    if (!open) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      if (!stoppingRef.current) {
        const requestRevision = revision.current;
        try {
          const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
          if (!cancelled && requestRevision === revision.current) {
            setStatus(next);
            setError("");
            setChecking(false);
          }
        } catch (e) {
          if (!cancelled && requestRevision === revision.current) {
            setError(e instanceof Error ? e.message : String(e));
            setChecking(false);
          }
        }
      }
      if (!cancelled) timer = setTimeout(poll, 1000);
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      ++revision.current;
    };
  }, [open, baseUrl, fetchWithHeaders]);

  const stopPrevious = useCallback(async () => {
    if (stoppingRef.current) return;
    stoppingRef.current = true;
    const requestRevision = ++revision.current;
    setStopping(true);
    setError("");
    try {
      const next = await stopInferenceAndWait(baseUrl, fetchWithHeaders);
      if (requestRevision === revision.current) {
        setStatus(next);
        setChecking(false);
      }
    } catch (e) {
      if (requestRevision === revision.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      stoppingRef.current = false;
      setStopping(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  return { status, checking, stopping, error, stopPrevious,
    blocked: !open || checking || stopping || !!error || status?.inference_active !== false };
}
