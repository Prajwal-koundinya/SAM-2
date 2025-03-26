# Advanced Object Masking with SAM 2 



## 📌 Overview

This repository demonstrates the use of **Segment Anything Model 2 (SAM 2)** for advanced image segmentation by predicting object masks using various input prompts such as points and bounding boxes. SAM 2 efficiently generates high-quality masks by first converting the image into an embedding and then predicting the masks based on user-defined prompts.

## 🚀 Features

- **High-Quality Object Masking**: Uses SAM 2’s powerful model to generate object masks with precision.
- **Multi-Prompt Support**: Accepts **points, bounding boxes, and previous masks** as inputs.
- **CUDA Acceleration**: Optimized for GPU to enhance performance.
- **Batched Processing**: Supports multi-object and multi-image inference.
- **Refined Mask Selection**: Generates multiple masks and selects the best one based on model confidence scores.

## 🛠️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Prajwal-koundinya/SAM-2.git
cd SAM-2

# Install dependencies
pip install torch torchvision segment-anything
```

## 📷 Image Segmentation Workflow

1. **Load the SAM 2 Model & Predictor**

   - Ensure you have the correct model checkpoint.
   - Run SAM 2 on **CUDA** for optimal performance.

2. **Process the Image**

   - Use `SAM2ImagePredictor.set_image(image_path)` to generate an embedding.

3. **Apply Object Selection Methods**

   - **Single Point Selection**: Select an object with a foreground (1) or background (0) point.
   - **Bounding Boxes**: Provide rectangular prompts for object detection.
   - **Combined Prompts**: Use both **points and boxes** for refined selection.
   - **Multiple Points for Precision**: Enhance accuracy by selecting multiple points to specify objects more precisely.
   - **Iterative Refinement**: Use masks from previous predictions to improve segmentation.

4. **Predict & Generate Masks**

   ```python
   masks, scores, logits = predictor.predict(points=[(x, y, 1)], multimask_output=True)
   ```

   - If using `multimask_output=True`, SAM 2 generates **three masks**, selecting the best based on confidence scores.
   - When `multimask_output=False`, the model returns a single refined mask.

5. **Batched Inference for Multiple Objects**

   - Use multiple input prompts in one batch for efficient object segmentation.
   - Apply end-to-end batched processing for large-scale image segmentation tasks.

## 🖼️ Example Usage

```python
from segment_anything import SAM2ImagePredictor

# Initialize predictor and set image
predictor = SAM2ImagePredictor(model_path='sam2_checkpoint.pth')
predictor.set_image('example.jpg')

# Predict mask using a single point prompt
masks, scores, logits = predictor.predict(points=[(150, 200, 1)], multimask_output=True)

# Visualize the mask
predictor.show_mask(masks[0])
```

## 🎯 Results

- **Multi-mask output** enables selection of the most accurate segmentation.
- **Batched inference** allows processing multiple objects efficiently.
- **Combined prompts** refine object selection further.

## 📌 Future Enhancements

- Add support for **real-time segmentation** in video streams.
- Integrate **interactive UI** for better user experience.
- Extend **model fine-tuning** on custom datasets.

## 🤝 **Acknowledgments**
Special thanks to the medical and AI communities for their valuable datasets and research.  
Inspirational guidance from **Dr. Victor Ikechukwu**. Explore their work: [Dr. Victor Ikechukwu](https://github.com/Victor-Ikechukwu). 

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

Made by [Prajwal Koundinya](https://github.com/Prajwal-koundinya)



