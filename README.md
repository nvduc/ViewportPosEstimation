# ViewportPosEstimation

visible_mask_8x8_FoV_90.txt
This file contains the visible tiles at each viewport positions when the number of tiles is 64 and the viewport size is 90x90 (degrees).
Each line is format as follows.
0,-90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24654,24795,24654,24795,24654,24795,24654,24795,74676,75048,74850,75048,74857,75041,74850,75048,15457,15596,15456,15597,15562,15491,15484,15569

The first number is the longitude in degrees (0 -> 359)
The second number is the latitude in degrees (-90 -> 90)
The following 64 numbers is the numbers of pixels in the viewport of each tile. A value larger than zero indicates that the tile is a visible tile.
