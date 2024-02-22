# Define the transformation pipeline
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to convert bounding box from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    x_min = x_c - 0.5 * w
    y_min = y_c - 0.5 * h
    x_max = x_c + 0.5 * w
    y_max = y_c + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

# Function to rescale bounding boxes according to image size
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# Function to perform object detection on an image
def detect(im, model, transform):
    # Apply transformation
    img = transform(im).unsqueeze(0)
    # Ensure image size is within limits
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    # Forward pass through the model
    outputs = model(img)
    # Extract class probabilities and bounding boxes
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # Filter out detections with confidence above threshold
    keep = probas.max(-1).values > 0.7
    # Rescale bounding boxes
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

# Function to plot detection results
def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

# Load image
image_path = '/content/OIP.jpg'
im = Image.open(image_path)

# Perform object detection
scores, boxes = detect(im, detr, transform)

# Visualize detection results
plot_results(im, scores, boxes)
