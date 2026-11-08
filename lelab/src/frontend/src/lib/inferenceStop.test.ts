import { expect, test, vi } from "vitest";
import { stopInferenceAndWait } from "./inferenceApi";

function fetchSequence(...responses: { active?: boolean; error?: boolean }[]) {
  return vi.fn(async () => {
    const item = responses.shift();
    if (!item) throw new Error("Unexpected extra request");
    return { ok: !item.error, status: item.error ? 500 : 200,
      json: async () => item.error ? { detail: "stop failed" } : { inference_active: item.active } } as Response;
  });
}

test("idle stop is idempotent and does not terminate another process", async () => {
  const fetcher = fetchSequence({ active: false });
  expect((await stopInferenceAndWait("", fetcher)).inference_active).toBe(false);
  expect(fetcher).toHaveBeenCalledTimes(1);
});

test("waits for inactive confirmation after the stop response", async () => {
  vi.useFakeTimers();
  try {
    const fetcher = fetchSequence({ active: true }, {}, { active: true }, { active: false });
    let done = false;
    const pending = stopInferenceAndWait("", fetcher).then(() => { done = true; });
    await vi.advanceTimersByTimeAsync(0);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(250);
    await pending;
    expect(fetcher.mock.calls.map(call => (call as unknown[])[0])).toEqual([
      "/inference-status", "/stop-inference", "/inference-status", "/inference-status",
    ]);
  } finally { vi.useRealTimers(); }
});

test("failed stop keeps the device gate closed; natural exit race is harmless", async () => {
  await expect(stopInferenceAndWait("", fetchSequence({ active: true }, { error: true }, { active: true }))).rejects.toThrow("stop failed");
  expect((await stopInferenceAndWait("", fetchSequence({ active: true }, { error: true }, { active: false }))).inference_active).toBe(false);
});

test("does not claim success if the backend keeps reporting a live session", async () => {
  vi.useFakeTimers();
  try {
    const fetcher = vi.fn(async () => ({ ok: true, json: async () => ({ inference_active: true }) }) as Response);
    const result = stopInferenceAndWait("", fetcher).catch(e => e);
    await vi.advanceTimersByTimeAsync(10000);
    expect((await result).message).toContain("旧推理尚未退出");
  } finally { vi.useRealTimers(); }
});
