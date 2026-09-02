"""Isolated cuRobo v0.7.8 planning backend."""

PINNED_VERSION = '0.7.8'
PINNED_COMMIT = 'd64c4b005459db10c5dd867d8b30a87d5bda9bdb'
PINNED_WARP_VERSION = '1.11.1'
# Keep ordinary MotionGen solutions away from the raw PiPER execution limits.
# Joint 2's configured home is exactly its lower limit and joint 3's configured
# home is exactly its upper limit, so those two intentional boundaries remain
# available and are protected by strict output validation instead of clipping.
# The remaining 0.005 rad insets are deliberately smaller than Tesseract's IK
# search margin while still dwarfing float32 boundary noise.
POSITION_LIMIT_CLIP_RAD = (0.005, 0.0, 0.0, 0.005, 0.005, 0.005)
