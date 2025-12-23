# Project_Documentation-

##segmentation model
### Model Tested: Mask R-CNN (ResNet-50 FPN)

**What I Tried:**  
I used the Mask R-CNN model with a ResNet-50 backbone to detect and segment cricket players in images. The idea was to get clean masks for each player, so I could later extract features for identification.

**How It Worked:**  
The model was able to find people in the images and create masks around them. I set a high confidence threshold to reduce false positives, and saved each detected player as a separate image.

**What Didn’t Work Well:**  
- **Slow Processing:** Running the model on multiple images took too long. It wasn’t practical for a larger dataset.  
- **Missed Important Features:** Sometimes it didn’t capture all the player’s details, especially helmets. This was a problem because missing a helmet or other key part could hurt identification accuracy.  
- **Feature vs. Noise Trade-off:** In this project, I’d rather have a little extra noise than miss important features entirely. Unfortunately, Mask R-CNN occasionally skipped over crucial parts.

**Decision:**  
In the end, I **decided not to use Mask R-CNN**. While it’s great for general segmentation, it was **too slow** and **didn’t consistently capture all the features** we need for identifying players.
