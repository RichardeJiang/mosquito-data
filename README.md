# mosquito-data
This is the Python files for generating synthesized mosquito videos based on existing lab footages and track information.

## Libraries:
The following libraries are used in the code, namely:
- vidstab; for video stabilization.
- OpenCV;
- SciPy; for importing matlab files.
- numpy.

## Files:
All listed files are used somewhere along the way during the development of the project. Important ones and the features tested by them are listed as below:
- stab.py: the stabilization code for background videos;
- logic1.py: <br>
Initial logic, where the whole frame from the lab video is manipulated and pasted onto the background video.
- logic2.py: <br>
Finalized logic, where the local patch of the mosquito (defined by the centroid + width / height) is cut off, and then pasted onto the background video.
- logic2all.py: <br>
The code to generate the one-mosquito videos using logic 2; using STAIR dataset as the background. At any time, there will be only one mosquito in the synthesized result.
- multiplelogic2.py: <br>
The **main code** to generate the final output: multiple mosquitoes, starting from random positions, and possibly fly out and re-enter into the frame at some time.

## Main logic for multiplelogic2.py:
### Important Utility Functions Used inside the Code:
- getObjFromMaskWhitish(): <br>
Retrieve the pixel coordinates from the mask, based on the pixel value and the pre-defined threshold;
- pasteWithRescaling(): <br>
Paste the extracted mosquitoes onto the bg video; rescale the RGB values first, check the validity of pasting positions, and do the pasting;

### Main Function Logic:
The pseudocode for the main logic can be found as below:
```
retrieve all information (bg video list, mosquito video list, track list);
for bgVideo in bgVideoList:
    randomly choose the number of mosquitoes to be pasted into this video;
    for i in numOfMosquitoes:
        randomly choose the starting position;
        randomly choose the mosquito video / tracks to be used;
        randomly choose the down-scale factor of each frame (not completely random, choose based on the previous value so there is no sudden change);
        calculate the pasting position of each frame based on the actual movement and down-scale factor of the frame;
        smooth the transition between tracks;
        for each frame:
            retrieve the mosquitoes based on lab video, tracks, masks;
            check if pasting position is valid (inside the boundary), if true:
                do the pasting and append the result image to the list;
                append the position on the final result frame to the label list;
            else:
                use a random number of bg video frames as the result frame, and choose a random edge for re-entering, adjust all subsequent pasting positions;
                append [-1, -1] as the position label to the label list;

    write video and labels to file;
```
Other details can be found inside the code; important steps are written with comments of the logic / reason, so you can check out the specific implementation there.

## To run the main code:
Modify the path to all files (tracks, mosquito raw video, mosquito mask video, your own background video) located from **line 236** to **line 254**, and simply
```
python multiplelogic2.py
```

stab.py can be used in similar ways: modify the paths of your bg video to the correct value, and ``` python stab.py ```.

## Note on the implementation of logic 2:
Previously, logic 2 also works by cutting out the local patch and pasting it onto the background video, but the result is bad; the main reason is due to the pasting positions: the width/height provided by the track information varies in each frame, hence if pasting using the centroid as "start", it results in minor shift of the real centroid position and the final effect is the fuzzy flying pattern. Hence, the correct logic is using the centroid as the centroid, and when pasting, using ``` pastingPosX - width / 2 ``` as the starting X, and similarly ``` pastingPosY - height / 2 ``` as the starting Y.

## Sample output:
We provide one sample output synthesized video for you to get a sense on the effect of our composition logic, available from this [link](https://drive.google.com/open?id=1z1EKYxx-msV6xDclsRCun7FLgQvWHjF8); the binary and position labels can also be found there.
