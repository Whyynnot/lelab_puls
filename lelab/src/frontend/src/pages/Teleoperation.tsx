import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";

const TeleoperationPage = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [fault, setFault] = useState<{ message: string; detail: string } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let pending = false;
    const checkStatus = async () => {
      if (pending || controller.signal.aborted) return;
      pending = true;
      try {
        const response = await fetchWithHeaders(`${baseUrl}/teleoperation-status`, {
          signal: controller.signal,
        });
        if (!response.ok) return;
        const status = await response.json();
        if (!controller.signal.aborted && status.last_error) {
          setFault({ message: status.fault_message || "遥操作异常停止，请先支撑好从臂再检查连接。", detail: status.last_error });
        }
      } catch {
        // Keep an existing fault visible if the connection is interrupted.
      } finally {
        pending = false;
      }
    };
    void checkStatus();
    const timer = window.setInterval(checkStatus, 1000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [baseUrl, fetchWithHeaders]);

  // Stop teleoperation exactly once, however the user leaves, so the back
  // button, an in-app link, and the unmount safety net can't double-stop or
  // double-toast.
  const stoppedRef = useRef(false);
  const stopTeleoperation = useCallback(async () => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
      });
      const data = await res.json();
      if (data?.success) {
        toast({
          title: "Teleoperation stopped",
          description: "The arm was disconnected cleanly.",
        });
      }
    } catch {
      /* best-effort */
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  // Cover every exit path so a session can't keep running and block the next
  // start with "already active":
  //   - the back button awaits stopTeleoperation() then navigates (below);
  //   - any other in-app navigation unmounts this component → stop via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload
  //     and stashes a flag the next page reads to confirm the clean disconnect.
  //     It uses a bare fetch (no JSON Content-Type) so the request stays a CORS
  //     "simple request" and isn't dropped to a preflight mid-unload.
  useEffect(() => {
    const handlePageHide = () => {
      try {
        sessionStorage.setItem("lelab:teleop-stopped", "1");
      } catch {
        /* sessionStorage may be unavailable; the stop below still runs */
      }
      fetch(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    };
    window.addEventListener("pagehide", handlePageHide);

    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      stopTeleoperation();
    };
  }, [baseUrl, stopTeleoperation]);

  const handleGoBack = async () => {
    await stopTeleoperation();
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-2 sm:p-4">
      <div className="w-full h-[95vh] flex flex-col gap-3">
        {fault && (
          <div role="alert" className="shrink-0 rounded-lg border border-red-500 bg-red-950 px-4 py-3 text-white">
            <p className="font-semibold">遥操作已中断</p>
            <p className="mt-1 text-sm">{fault.message}</p>
            <details className="mt-2 text-xs text-red-100">
              <summary className="cursor-pointer">通信错误详情</summary>
              <pre className="mt-1 whitespace-pre-wrap break-words">{fault.detail}</pre>
            </details>
          </div>
        )}
        <div className="flex min-h-0 flex-1">
        <VisualizerPanel
          onGoBack={handleGoBack}
          className="lg:w-full"
          rightSlot={<TeleopCameraPanel />}
        />
        </div>
      </div>
    </div>
  );
};

export default TeleoperationPage;
