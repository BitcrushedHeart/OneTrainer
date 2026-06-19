# DPO Pairing Tool

## Issue 1
The images still do not correctly scale to the right resolution - the images should be grown to fill the box dependent on the UI's display resolution (while maintaining it's aspect ratio), for images smaller, scale them up to fit.

## Issue 2
Let the user see the prompt in full with an expandable box in the top, if they're using DPO, it's likely they want to know what the prompt actually says to help steer the model to better prompt adherence, this means they'll need to see the detected prompt in full rather than just a few words of it.