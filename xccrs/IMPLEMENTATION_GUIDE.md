# XCCRS Implementation Guide & Technical Deep Dive

## System Architecture Explained
The system is built on a modular "perception-reasoning-action" loop, enforcing a strict separation between neural probabilistic outputs and symbolic deterministic logic.

### 1. Neural Perception Layer
- **Scene Understanding (BLIP)**: Acts as the "eyes" for context. We use  because it handles diverse industrial scenes better than smaller variants. It outputs a caption which we then match to keywords.
- **Object Detection (YOLOv8)**: The "locator". It finds entities. We normalize outputs to ensure downstream modules receive consistent labels (e.g., all 0-class detections become 'person').
- **Attribute Verification (CLIP)**: The "inspector". It takes crops of detected objects and verifies specific properties (PPE). We use a contrastive prompt engineering approach ("wearing helmet" vs "without helmet") to maximize accuracy.

### 2. Symbolic Reasoning Layer
- **Knowledge Base**: A JSON file acting as the immutable set of laws.
- **Reasoning Engine**: A pure Python logical module that takes the purely descriptive state (Person A has Helmet, Person B has no Helmet) and applies the laws (Factory requires Helmet) to derive a normative judgment (Violation).

## Critical Implementation Patterns

### The Separation Pattern
Never let a neural model decide "compliance". 
* **Bad**: Ask LLM "Is this worker safe?" (Unpredictable, Hallucinations)
* **Good**: 
    1. Neural: "Person Detected." (99% conf)
    2. Neural: "Helmet Detected." (95% conf)
    3. Symbolic: RULE(Person AND NoHelmet) -> Violation.

### CLIP Cropping Strategy
To get accurate attribute verification, we must crop the object first. Passing the whole image to CLIP with the prompt "person with helmet" often fails if there are multiple people or background clutter.
```python
crop = image.crop(box)
# Verify ONLY the crop
clip.score(crop, ["helmet", "no helmet"])
```

## Common Pitfalls & Solutions

1. **BLIP Hallucination**: BLIP might describe a warehouse as a factory.
   * *Fix*: Use strict keyword matching with a fallback to "unknown".
2. **YOLO Class Limitation**: YOLOv8n doesn't know "safety vest".
   * *Fix*: Use YOLO for 'person' detection, then use CLIP to check for the vest on the person crop.
3. **CLIP Thresholding**: Setting a fixed cosine similarity threshold is tricky.
   * *Fix*: Use "argmax" between positive and negative prompts (e.g., score(helmet) > score(no_helmet)).
4. **Small Object Detection**: Missing small hazardous items.
   * *Fix*: Increase image resolution or use sliding window (not implemented in v1).
5. **Video FPS Overhead**: Processing every frame is too slow.
   * *Fix*: Sample at 1 FPS (implemented in ).

## Customization
To add a new scene type:
1. Edit .
2. Add keywords to .
3. No retraining needed.

## Performance Optimization
- Run CLIP and BLIP on GPU ( or ).
- Batch CLIP queries if processing many people (currently sequential for clarity).
