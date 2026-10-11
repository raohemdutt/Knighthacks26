# The Atlas enclosure (for the 3D model in world.js)

From Dom's CAD notes (Sat 2026-10-10). Millimetres. The CAD package (atlas_geometry.json, SVGs) is NOT on
disk; every number the film needs is here. Do not look for those files.

## Coordinates

Origin (0, 0, 0) = the lower-front-right OUTSIDE corner of the cardboard box. +X toward the device's LEFT,
+Y toward the BACK, +Z UP. Facing the honeycomb front, the device's right is the viewer's left, so X = 0 is
on screen-left and X = 125 on screen-right.

Three.js mapping (1 world unit = 100 mm, the box centred on x/z, the front facing +z, sitting on the desk at
y = -1.0): `x = (X - 62.5) / 100`, `y = -1.0 + Z / 100`, `z = (99 - Y) / 100`. Write it once in world.js as
`mm(X, Y, Z)` and a box helper that takes a size and a min corner, and use it for every part. Check one
snapshot from the front: the cable gap (device right) must be on screen-left.

## Shell

| Part | Size (X × Y × Z) | Notes |
| --- | --- | --- |
| Cardboard box, outside | 125 × 198 × 80 | bottom, left, right, back walls are separate cardboard panels, 1.3 thick |
| Inside | about 122.4 × 196.7 | |
| Top lid (3D printed) | 130.5 × 206.5 × 3 | overhangs the box a little on every side; a cooling vent over the Jetson; the camera socket |
| Front panel (3D printed) | 125 × 79.5 × 4 | honeycomb grille, raised "Atlas" logo |
| Floor rail (3D printed) | 125 wide | a thin rail under the front edge |

Look: cardboard walls read as a warm matte kraft-grey in the soft palette (keep it close to the surface
colour, slightly warmer); printed parts (lid, front, rail) read as the soft white surface.

## Inside (bounding boxes, min corner)

| Part | Size (X × Y × Z) | Min corner | Responsible for (the film's words) |
| --- | --- | --- | --- |
| onn speaker | 121 × 65 × 70 | (2, 0.5, 2.9) | answering out loud (voice) |
| NVIDIA Jetson Orin Nano dev kit | 103 × 90.5 × 34.77 (official) | (11, 76.2, 2.9) | the vision model, the memory, the agent: everything live |
| Webcam clip (in the lid socket) | 38 × 47 × 17 | (43.5, 8.65, ~63.6) | holds the camera |
| Webcam housing (on top of the lid) | 96.5 × 30.5 × 35.6 | (14.25, 17.65, 88.6) | seeing: the camera |

Socket opening in the lid: 39.5 × 48.5 with a 22 × 22 cable gap on the device's right.

Known conflict: the 70 mm speaker overlaps the downward camera socket. For the film, shorten the socket so it
stops above the speaker. Do not try to resolve the CAD.

Jetson detail worth drawing: a green-black carrier board (100 × 79), the module with a finned heat sink and a
fan on top, a few ports on the back edge (USB, Ethernet, power). The Ethernet port is where the network cable
plugs in (back wall, behind the Jetson).
