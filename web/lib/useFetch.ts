"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

export interface Loadable<T> {
  data: T | undefined;
  error: ApiError | undefined;
  loading: boolean;
  /** Fetch again, keeping the current data on screen while it loads. */
  reload: () => void;
}

/**
 * Load something from the gateway when the component mounts and whenever `key`
 * changes. A slower earlier request can never overwrite a newer one.
 */
export function useFetch<T>(load: (signal: AbortSignal) => Promise<T>, key: string = ""): Loadable<T> {
  const [state, setState] = useState<{ data?: T; error?: ApiError; loading: boolean; key: string }>({ loading: true, key });
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    const ctl = new AbortController();
    setState((s) => (s.key === key ? { ...s, loading: true } : { loading: true, key }));
    loadRef
      .current(ctl.signal)
      .then((data) => setState({ data, loading: false, key }))
      .catch((err: unknown) => {
        if (ctl.signal.aborted) return;
        const error = err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err));
        setState({ error, loading: false, key });
      });
    return () => ctl.abort();
  }, [key, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  const fresh = state.key === key;
  return { data: fresh ? state.data : undefined, error: fresh ? state.error : undefined, loading: state.loading || !fresh, reload };
}
