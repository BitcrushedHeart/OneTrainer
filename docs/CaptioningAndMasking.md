# Captioning and Masking

OneTrainer includes a UI for captioning and masking of your dataset. To access it, click on `Dataset Tools` within the `tools`
tab. Once the UI is open, click the `Open` button at the top left, and select the directory of your dataset.

### Navigating the UI

To switch between the images, either click on the filename in the list on the left side, or use the up and down arrow
keys to go to the next or previous image.

### Manual captioning

In the input box at the bottom you can input your caption. To save the caption, press enter.

### Automatic captioning

Clicking on the `Generate Captions` button opens the batch captioning modal. Here you can choose which model to use for
captioning. For the initial caption, you can choose a text that should be used to start the new caption. To generate the
captions, press the `Create Captions` button. When you use this for the first time, it has to download the model.
Depending on the model you chose, this can take a while.

Entries in the `Prefix` field will be added at the start of the caption. `Postfix` will be added at the end of the caption.

### Manual masking

Check the `Enable Mask Editing` checkbox at the top. Now you can draw a mask onto the image. Left-click adds to the
masked region, right click removes parts from the mask. With the mouse wheel you can increase or decrease the brush
size. Use `Ctrl + M` to only show the mask. To save the mask, click into the caption input field, then press enter.

### Automatic masking

Clicking on the `Generate Masks` button opens the batch masking modal. Here you can choose which model to use for
masking. Some models like ClipSeg support masking based on a prompt. Play around with Threshold, Smooth and Expand values to find what works best for your dataset.

To generate the masks, press the "Create Masks" button at the bottom. When you use this for the first time, it has to download the model. Depending on the model you chose, this can take a while.

### Masked DPO

When masked training is enabled on an RLHF DPO run, each pair's preference margin is computed inside the mask. The
chosen image uses its own `-masklabel.png`. The rejected image uses its own `-masklabel.png` when one exists next to it
and otherwise shares the chosen image's mask, which is the right choice for aligned pairs (a rejected image fixed in
place to produce the chosen one). Pairs whose compositions differ need a mask on both images.

Each half's log-probability is the per-pixel average error inside its weighted region, so the margin does not scale
with the mask's area and beta means the same thing for every pair. The `Unmasked Weight` still applies: with a weight
above zero the background keeps contributing to the margin, so region-focused DPO usually wants it at or near zero.

Two per-concept overrides in the concept window apply to masked SFT and DPO concepts alike. `Unmasked Weight Override`
replaces the run's unmasked weight for that concept. `Unmasked Probability Override` trains a sample on the whole image
with that chance, drawn per sample each epoch; a DPO pair is unmasked together. A negative value inherits the run's
setting. Inpainting models apply only the run-wide `Unmasked Probability`, because their unmasking step also blanks
the conditioning image.
