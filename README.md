### Crop and align talking videos according to lip-movement

This is a part of the multimodal word learning project. The original talking videos contain the whole face of the speaker. To prepare video stimuli for the fMRI experiment, the original videos are cropped so that only the lip area remains, and are aligned based on the lip position. Specifically, [the facial landmarks](https://ibug.doc.ic.ac.uk/resources/300-W/) are used for estimating the Center of Mass (CoM) of the lip area. The videos are cropped and aligned using the corresponding CoMs.

[!Example](https://i.imgur.com/vZNngGT.png)
