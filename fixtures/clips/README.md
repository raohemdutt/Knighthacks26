# Desk-object camera clips (L22)

Recorded on 2026-10-10 about 7:50 PM ET through the Jetson gateway's blurred `snapshot.jpg` with `jetson/record_clip.py`
(command and method in `lab/README.md`, section "Camera clips"). Each is 10 s at 1280x720, H.264, played at the rate
the gateway really delivered (5.4 to 5.6 frames a second that evening, not the 15 to 30 the plan asked for). The
reader inside the gateway itself reported only 9.8 to 10.9 fps, and the snapshot endpoint served about half of those;
the webcam is capable of 30 (see `lab/README.md`).

- `package-at-door.mp4` (3.2 MB, sha256 `8df906b5f121776df0d91860b2f89a5bcbf68342d070882ad53f68ffe95c2d69`): a close
  view of stacked flat corrugated cardboard with a taped box on top and a blurred room behind it; nothing moves. Stands
  in for a package at the door.
- `keys-on-desk.mp4` (3.2 MB, sha256 `ec1f2d832049f9d2c06c6a184d5be3e4fbbbb13bbadc1e953fe08d3aedf98200`): placeholder.
  There were no keys in front of the camera, so this is the same cardboard scene as the clip above.
- `person-walks-through.mp4` (2.1 MB, sha256 `f91bbd704968e519b0a16ae3a3094046d4705a1008779e17f63603a124c617cf`): NOT in
  this repo. Someone picked up the camera and put it close to their face; the face blur did not catch a face that close,
  so the frames show eyes, nose and mouth. It stays on the box at `~/atlas/fixtures/clips/person-walks-through.mp4`
  (with its frames and `clip.json` beside it). It is a person in front of the camera, not a person walking through.

`../learning/desk-clips-fixture/` holds 12 frames (6 from each clip above, no labels yet) in the export layout that
`lab/dataset.py` reads. Its manifest carries the action id `fixture:desk-clips`: nobody approved an export; this is
fixture data. Both scenes are the same, so it is a format fixture for the labeling pipeline, not a varied set; record
real keys and a real package when the box is at a desk.
