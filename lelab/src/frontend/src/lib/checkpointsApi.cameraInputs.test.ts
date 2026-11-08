import { expect, test } from "vitest";
import { getCheckpointPolicyConfig, physicalImageFeatures } from "./checkpointsApi";

test.each(["pi05", "pi0", "smolvla", "future_policy"])(
  "%s: generated padding never appears as a physical camera, including legacy API responses",
  async (policy_type) => {
    const raw = { policy_type, requires_task: true,
      image_features: { head: { width: 640, height: 480 }, wrist: { width: 640, height: 480 },
        empty_camera_0: { width: 224, height: 224 }, empty_camera_1: { width: 224, height: 224 } } };
    const summary = await getCheckpointPolicyConfig("http://test", async () => new Response(JSON.stringify(raw)), "job", 1000);
    expect(Object.keys(summary.image_features)).toEqual(["head", "wrist"]);
    expect(Object.keys(raw.image_features)).toHaveLength(4);
  },
);

test("real camera names remain intact", () => {
  const features = { camera1: { width: 640, height: 480 }, camera2: { width: 640, height: 480 },
    camera3: { width: 640, height: 480 }, empty_camera_front: { width: 640, height: 480 } };
  expect(physicalImageFeatures(features)).toEqual(features);
});
