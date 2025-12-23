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


### Model Tested: YOLOv8-Seg for Player Segmentation

**What I Tried:**  
After testing Mask R-CNN, I decided to try YOLOv8-seg for detecting and segmenting cricket players. The goal remained the same: isolate each player in an image so I could later extract features for identification. I wanted a model that was **faster and more reliable at capturing all player features**, even when the player was partially occluded or wearing equipment like helmets.

**How It Worked:**  
YOLOv8-seg was able to detect players quickly and generate segmentation masks for each person in the image. It handled multiple players per frame, preserved their visual features, and saved each segmented player as a separate image. Even when there was some background noise, the model ensured that **no important parts of the player were lost**.  

Some key technical points:  
- Used a confidence threshold of 0.25 and a mask threshold of 0.5.  
- Converted the predicted masks to the original image size and applied them directly on the original image.  
- Supported batch processing of multiple images with consistent results.  

**Why It Worked Better Than Mask R-CNN:**  
- **Speed:** YOLOv8-seg processed images much faster than Mask R-CNN, which is important for practical use and real-time applications.  
- **Feature Preservation:** Unlike Mask R-CNN, it did not remove key features of the player. Even helmets and parts of the uniform were preserved, which is crucial for accurate player identification.  
- **Balance Between Accuracy and Noise:** For this project, it’s better to include a little extra noise than to miss important features. YOLOv8-seg successfully struck this balance, capturing all necessary visual details while still being fast.  

**Decision:**  
I decided to **proceed with YOLOv8-seg** for the segmentation stage in the final workflow. It provides **a good combination of speed, accuracy, and feature preservation**, making it suitable for both real-time detection and downstream player identification tasks.

**Output**
<img width="363" height="737" alt="image" src="https://github.com/user-attachments/assets/c149b8fa-53ad-4802-9a67-42ad9f83dc73" />
<img width="366" height="725" alt="image" src="https://github.com/user-attachments/assets/7bab9203-d40f-4035-9570-7469f5b3fcb7" />
<img width="363" height="735" alt="image" src="https://github.com/user-attachments/assets/11192512-40c7-4c44-940f-d38ec613bb8e" />
<img width="365" height="676" alt="image" src="https://github.com/user-attachments/assets/beb31b73-e89d-426e-947e-794d94cb800b" />
<img width="365" height="722" alt="image" src="https://github.com/user-attachments/assets/848e6610-79ce-43d9-abff-fef0e5728da1" />
<img width="366" height="589" alt="image" src="https://github.com/user-attachments/assets/4848f229-ebeb-40c2-9403-26c6635082ca" />
<img width="363" height="687" alt="image" src="https://github.com/user-attachments/assets/c4151a35-3dc6-4cec-a1ed-152273c73d87" />
<img width="363" height="687" alt="image" src="https://github.com/user-attachments/assets/7eae9b8e-900d-4e32-b9f2-c09b6dd959a0" />
<img width="360" height="806" alt="image" src="https://github.com/user-attachments/assets/aa9c82de-ff86-47a4-88d4-8cb549fca6b6" />
<img width="360" height="806" alt="image" src="https://github.com/user-attachments/assets/30147e93-5563-4746-a8b6-6d5cd67c1e10" />






