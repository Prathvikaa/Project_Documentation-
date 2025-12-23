# Project_Documentation-

## SEGMENTATION MODELS
### Model Tested: Mask R-CNN (ResNet-50 FPN)

**What I Tried:**  
I used the Mask R-CNN model with a ResNet-50 backbone to detect and segment cricket players in images. The idea was to get clean masks for each player, so I could later extract features for identification.

**How It Worked:**  
The model could detect people and generate masks for them.Confidence was set high (0.8) to reduce false positives.Each detected player was saved as a separate image for further processing.T

**What Didn’t Work Well:**  
- **Slow Processing:** Running the model on multiple images took too long. It wasn’t practical for a larger dataset.  
- **Missed Important Features:** Sometimes it didn’t capture important parts of the player, especially helmets. This is a problem because even missing small details can hurt the accuracy of identifying players later.
- **Feature vs. Noise Trade-off:** In this project, I’d rather have a little extra noise than miss important features entirely. Unfortunately, Mask R-CNN occasionally skipped over crucial parts.

**Decision:**  
In the end, I **decided not to use Mask R-CNN**. While it’s great for general segmentation, it was **too slow** and **didn’t consistently capture all the features** we need for identifying players.

**Output:**


<img width="364" height="742" alt="image" src="https://github.com/user-attachments/assets/24f2a73a-402d-4882-8b50-7ad5768ebf4c" />
<img width="364" height="742" alt="image" src="https://github.com/user-attachments/assets/2ad49169-ed6d-4520-8db2-f6d6f076b7ef" />
<img width="364" height="742" alt="image" src="https://github.com/user-attachments/assets/95ff9d39-e39f-454d-81c4-29bcd4c59761" />
<img width="364" height="690" alt="image" src="https://github.com/user-attachments/assets/1719343e-0ae3-41cc-aacd-215c3273e8cf" />
<img width="356" height="840" alt="image" src="https://github.com/user-attachments/assets/2e60ddc3-1229-4cf4-80fd-c2217b0cd273" />
<img width="357" height="791" alt="image" src="https://github.com/user-attachments/assets/c6482fc5-d6a9-4c9c-8e18-96330256cff9" />
<img width="356" height="813" alt="image" src="https://github.com/user-attachments/assets/8b81516d-5875-425f-bd24-394f32c863ae" />



