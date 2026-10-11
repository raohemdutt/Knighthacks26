import { Composition } from "remotion";
import { Atlas } from "./Atlas";
import timeline from "./timeline.json";

/**
 * One composition, frame-exact: 9:58 at 30 fps is 17,940 frames, ten slots of 1,794. YouTube's number keys jump to
 * tenths of the video, so key k lands on slot k. The timeline comes from contexts/demo/video.py (`npm run timeline`);
 * never edit timeline.json by hand.
 */
export const Root: React.FC = () => (
  <Composition
    id="Atlas"
    component={Atlas}
    durationInFrames={timeline.duration_frames}
    fps={timeline.fps}
    width={1920}
    height={1080}
  />
);
