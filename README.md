# mosquito-data
This is the Python files for generating synthesized mosquito videos based on existing lab footages and track information.

## Note on logic 2:
It doesn't work well as there is no way to detect the exact "pasting position", nor could we get the precise "centroid" from the mask, hence when extracting out only the local patch, the patch size could be very different from frame to frame, hence pasting using the manually-calculated position leads to fuzzy result.