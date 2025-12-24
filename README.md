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

---

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

---

## Why Other Instance Segmentation Models Were Not Used

While working on this project, I didn’t start by coding every available instance segmentation model. Before implementing anything, I spent time reading research papers and understanding how different segmentation models behave in real-world pipelines.

From this process, I realized that many models perform well on benchmarks but show practical limitations when used in an end-to-end system.


### Segment Anything Model (SAM)

When SAM was released, it looked very promising because of the quality of its segmentation masks. I did try to work with SAM, but I quickly ran into practical issues.

The model is very heavy in terms of checkpoint size and memory usage. Running it repeatedly on multiple images was not feasible on my setup. In addition, SAM is not designed for fully automated pipelines and often requires prompts or extra logic to consistently segment only the player.

Because of these reasons, even though the segmentation quality is high, SAM was not practical for this project.


### Cascade Mask R-CNN

From reading research papers, I learned that Cascade Mask R-CNN improves accuracy by refining predictions in multiple stages. However, this also makes the model even slower than standard Mask R-CNN.

Since I had already observed that Mask R-CNN was slow and sometimes removed important features like helmets, it was clear that Cascade Mask R-CNN would only make these issues worse. Therefore, it was not implemented.


### YOLACT / YOLACT++

YOLACT models are often described as real-time instance segmentation approaches. From the literature, I found that they achieve speed by simplifying the mask generation process.

The downside of this trade-off is that the resulting masks can be coarse. Small but important details, such as helmets or player equipment, are sometimes lost. For this project, losing even a small feature can affect player identification accuracy, which made YOLACT unsuitable.


### SOLO / SOLOv2

SOLO-based models remove the concept of bounding boxes and directly predict instance masks. While this approach is interesting, papers show that these models struggle when objects overlap or when there are large variations in scale.

Since cricket images often contain overlapping players and varying camera angles, SOLO-based models were not reliable for this use case.


## Final Takeaway

Instead of implementing many models and encountering the same limitations repeatedly, I focused on understanding their drawbacks first. By studying research papers and testing models where possible, it became clear which approaches would not fit the project requirements.

YOLOv8-seg was ultimately chosen not because it is popular, but because it offers the best balance between speed, feature preservation, and practical usability for this task.

---

## **Embedding Model**

After completing the segmentation stage, the next step in the pipeline was to generate embeddings that could be used to identify players using similarity comparison.


### **ArcFace (Face-Based Embedding Model)**

**What I Tried**  
I first experimented with ArcFace, a popular and widely used face recognition embedding model. I used the pretrained `buffalo_l` model from the InsightFace library to extract embeddings from player images and compared them using cosine similarity.

ArcFace produces a **512-dimensional embedding** for each detected face, which is commonly used in face recognition systems.


**What Worked**  
- The model generated stable embeddings when a clear face was visible.
- Cosine similarity behaved correctly in controlled cases with frontal face images.
- For clean face images, similarity scores between the same person were consistent.


**What Didn’t Work Well**  
While ArcFace works extremely well for face recognition, it showed clear limitations for this project:

- **Strong dependency on face detection:** If a face was not detected, the embedding could not be generated.
- **Helmet and occlusion issue:** In many cricket images, players wear helmets or face away from the camera, causing face detection to fail.
- **Limited information:** ArcFace only uses facial features and ignores other important cues such as jersey color, body structure, posture, and equipment.
- **Unreliable in real scenarios:** Since face visibility cannot be guaranteed in sports images, the pipeline became unstable.


**Decision**  
I decided **not to proceed with ArcFace** for the final system. Although it is a strong face recognition model, it is **not suitable for cricket player identification**, where faces are often occluded or not visible at all.

This experiment made it clear that a **full-body or part-based embedding approach** is required instead of face-only embeddings.

**Output**

<img width="456" height="82" alt="image" src="https://github.com/user-attachments/assets/01bf2cda-b1ef-4cbd-91c7-61b861566dae" />
<img width="390" height="88" alt="image" src="https://github.com/user-attachments/assets/42ed4013-1c23-4cc4-9fd1-52c6c335a5cc" />


---

After observing the limitations of face-based embeddings, I moved on to models designed specifically for **person re-identification**, where the focus is on full-body appearance rather than facial features.


### **OSNet (Omni-Scale Network)**

**Why I Tried It**  
OSNet is a popular person re-identification model that captures visual features at multiple spatial scales. Since cricket player identification depends on clothing, posture, and overall body structure, OSNet appeared to be a strong candidate after rejecting face-based models.

I used a pretrained `osnet_x1_0` model from TorchReID and evaluated embeddings using cosine similarity.


**What Worked Well**  
- The model does not depend on face visibility.
- It captures global appearance features such as jersey color, body shape, and stance.
- Embedding extraction is stable and consistent across images.


**Where It Fell Short**

During practical testing, several limitations became clear:

- **Weak separation between players**  
  Similarity scores for the same player and different players were often very close, making it difficult to define a reliable threshold.

- **Sensitive to pose and viewpoint changes**  
  Changes in camera angle, posture, or movement caused noticeable variation in embeddings.

- **No understanding of body parts**  
  OSNet looks at the person as a whole. If the upper body, legs, or head were partially hidden or cropped, the embedding quality dropped.

- **Sports-specific challenges were not handled well**  
  Helmets, bats, pads, and fast motion introduced variations that the model could not consistently deal with.


**Final Decision**  
I decided **not to continue with OSNet**.

Even though OSNet works well for standard person re-identification problems, it was not reliable enough for cricket player identification. The lack of part-level understanding and the overlap in similarity scores made it unsuitable for achieving high accuracy without retraining or heavy customization.

**Output**

<img width="390" height="88" alt="image" src="https://github.com/user-attachments/assets/b8c294b3-0ccd-4c13-842e-89e2e2d0f655" />
<img width="390" height="88" alt="image" src="https://github.com/user-attachments/assets/bd1f2ad1-1e67-4945-9a1d-59a437ee5b96" />

---

Before moving to specialized identity models, I initially started with a more generic image embedding approach to understand how far I could get without task-specific assumptions.


### **OpenCLIP (ViT-B/32)**

**Why I Tried It**  
OpenCLIP is widely known for producing strong image embeddings that capture overall visual meaning. Since it works directly on full images and does not depend on face detection, it felt like a good starting point to test whether general-purpose embeddings could separate different players.


**What Looked Promising at First**  
- The model was easy to integrate and ran without any failures.
- Every image produced a valid embedding.
- Visually similar images resulted in high cosine similarity, which initially seemed encouraging.


**What Went Wrong**

Once I started comparing players more carefully, a clear issue emerged:

- **Different players looked too similar to the model**  
  Even when the images were of completely different players, the cosine similarity scores were still high. This made it hard to trust the similarity values.

- **The model cares about “what” not “who”**  
  OpenCLIP focuses on recognizing the general concept of an image (for example, “a cricket player in uniform”) rather than identifying a specific individual. Because of this, players wearing similar jerseys or standing in similar poses were treated as almost the same person.

- **No clear separation boundary**  
  The similarity scores for same-player pairs and different-player pairs overlapped heavily, which made threshold-based decisions unreliable.


**Final Decision**  
I decided **not to continue with OpenCLIP**.

Although OpenCLIP is powerful for general image understanding, it is not suitable for player identification. The experiment made it clear that this problem requires identity-aware embeddings, not just visually or semantically similar ones.

This test helped narrow down the direction and confirmed that more specialized embedding models were necessary.

**Output**

<img width="390" height="88" alt="image" src="https://github.com/user-attachments/assets/58129406-efd0-4712-ac77-3a02d6be772f" />
<img width="350" height="86" alt="image" src="https://github.com/user-attachments/assets/4298fa03-ac5e-4274-95d9-5993a03f4dac" />

---

After experimenting with multiple embedding models and understanding their limitations, I wanted to explore an approach that could handle the real challenges of cricket player identification more intelligently.

This led me to experiment with a **part-based embedding model**.


### **PCB-P6 (Part-based Convolutional Baseline)**

**Why This Model Caught My Attention**  
One major issue with earlier models was that they treated the entire person as a single entity. In sports images, this becomes a problem because different body regions contribute differently to identity.

PCB-P6 addresses this by **dividing the person into six horizontal parts** and learning embeddings for each part separately. This idea immediately felt relevant for cricket, where features like the helmet, jersey, and leg posture all carry identity information.

Because of this part-based design, PCB-P6 stood out conceptually, even before looking at numerical results.


**What Looks Promising So Far**

- **Part-level understanding**  
  By splitting the person into six regions, the model does not rely on a single global feature. Even if one part is noisy or occluded, other parts can still contribute useful information.

- **More robust to occlusion**  
  Helmets, bats, or partial occlusions affect only certain body parts. PCB-P6 reduces the impact of such occlusions by distributing identity information across multiple regions.

- **Better alignment with sports data**  
  Cricket images often vary in pose, camera angle, and movement. A part-based representation is more suitable for these variations than a single global embedding.

- **Conceptually strong for identity matching**  
  Even when cosine similarity scores were not dramatically better than previous models, the **structure of the representation made more sense** for this problem.
  

**Current Observations**

At this stage, PCB-P6 shows **moderate performance** in terms of cosine similarity and distance. The results are neither exceptionally high nor poor, but they are consistent.

More importantly, the model provides a **strong foundation for improvement**, such as:
- Weighted part similarity
- Ignoring noisy parts
- Combining part-level scores instead of relying on a single embedding


**Why I Decided to Continue with PCB-P6**

Unlike earlier models, PCB-P6 does not fail due to missing faces or excessive background influence. Instead, it offers a **clear direction for refinement and experimentation**.

Even though the model is **not finalized yet**, its part-based design aligns closely with the requirements of cricket player identification. Because of this, I chose to continue experimenting with PCB-P6 and build further improvements on top of it.


**Current Status**

PCB-P6 is the **active model under experimentation**. The focus is now on improving similarity aggregation, reducing noise from less informative parts, and integrating it effectively with the segmentation pipeline.

**Output**

<img width="350" height="86" alt="image" src="https://github.com/user-attachments/assets/ff7ab98a-d6a0-438c-9f88-5c39d64b4e00" />
<img width="350" height="86" alt="image" src="https://github.com/user-attachments/assets/c7030687-5f6f-4fa4-8809-a2f538931148" />



















