import cv2
import layoutparser as lp
from paddleocr import PaddleOCR, draw_ocr
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re

def get_next_filename(output_folder, prefix='Attendance', extension='csv'):
    """
    Get the next available filename in the form of 'Attendance[i].csv'.
    The index 'i' is the next available number based on the files already in the output folder.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the output folder that match the pattern 'Attendance*.csv'
    existing_files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith(f'.{extension}')]

    if existing_files:
        # Extract the numeric part of the filenames
        numbers = [int(f.replace(prefix, '').split('.')[0]) for f in existing_files if f.replace(prefix, '').split('.')[0].isdigit()]
        next_number = max(numbers) + 1 if numbers else 1
    else:
        next_number = 1  # Start from 1 if no files exist

    # Return the next available filename
    return f'{prefix}{next_number}.{extension}'

# Directory to save the output files
OUTPUT_FOLDER = './outputs'

def getImageFile():
    max_index = -1
    max_file = None
    directory = "./uploads/"
    pattern = re.compile(r"Attendance(\d+)\.(png|jpg)")

    for name in os.listdir(directory):
        match = pattern.match(name)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
                max_file = name
          
    if max_file:
        return os.path.join(directory, max_file)
    return None


print(getImageFile())

image = cv2.imread(getImageFile())
# image = image[..., ::-1]


# load model
model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                                threshold=0.5,
                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                enforce_cpu=False,
                                enable_mkldnn=True)
# detect
layout = model.detect(image)

for l in layout:
    if l.type == 'Table':
        x_1 = int(l.block.x_1)
        y_1 = int(l.block.y_1)
        x_2 = int(l.block.x_2)
        y_2 = int(l.block.y_2)
    break

im = cv2.imread(getImageFile())

cv2.imwrite('ext_im.jpg', im[y_1:y_2,x_1:x_2])

ocr = PaddleOCR(lang='en', use_angle_cls=True)
image_path = 'ext_im.jpg'
image_cv = cv2.imread(image_path)
image_height = image_cv.shape[0]
image_width = image_cv.shape[1]
output = ocr.ocr(image_path)[0]

# print(output)

boxes = [line[0] for line in output]
texts = [line[1][0] for line in output]
probabilities = [line[1][1] for line in output]

image_boxes = image_cv.copy()

for box,text in zip(boxes,texts):
  cv2.rectangle(image_boxes, (int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1])),(0,0,255),1)
  cv2.putText(image_boxes, text, (int(box[0][0]),int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)

cv2.imwrite('detections.jpg', image_boxes)

im = image_cv.copy()

horiz_boxes = []
vert_boxes = []

for box in boxes:
  x_h, x_v = 0, int(box[0][0])
  y_h, y_v = int(box[0][1]), 0
  width_h, width_v = image_width, int(box[2][0] - box[0][0])
  height_h, height_v = int(box[2][1] - box[0][1]), image_height

  horiz_boxes.append([x_h, y_h, x_h+width_h, y_h+height_h])
  vert_boxes.append([x_v, y_v, x_v+width_v, y_v+height_v])

  cv2.rectangle(im, (x_h, y_h), (x_h+width_h, y_h+height_h), (255,0,0), 1)
  cv2.rectangle(im, (x_v, y_v),(x_v+width_v, y_v+height_v), (0,255,0), 1)

cv2.imwrite('horiz_vert.jpg', im)

#Using tensorflow for lines suppression
horiz_out = tf.image.non_max_suppression(
  horiz_boxes,
  probabilities,
  max_output_size = 1000,
  iou_threshold = 0.1,
  score_threshold = float('-inf'),
  name = None
)

horiz_lines = np.sort(np.array(horiz_out))

# print(horiz_lines)

im_nms = image_cv.copy() #Normalized img

for val in horiz_lines:
  cv2.rectangle(im_nms, (int(horiz_boxes[val][0]), int(horiz_boxes[val][1]), int(horiz_boxes[val][2]), int(horiz_boxes[val][3])), (0,0,255), 1)

vert_out = tf.image.non_max_suppression(
  vert_boxes,
  probabilities,
  max_output_size = 1000,
  iou_threshold = 0.1,
  score_threshold = float('-inf'),
  name = None
)

vert_lines = np.sort(np.array(vert_out))

# print(vert_lines)

for val in vert_lines:
  cv2.rectangle(im_nms, (int(vert_boxes[val][0]), int(vert_boxes[val][1]), int(vert_boxes[val][2]), int(vert_boxes[val][3])), (255,0,0), 1)

cv2.imwrite("im_nms.jpg", im_nms)

#Convert to CSV
out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

# print(np.array(out_array).shape)

unordered_boxes = []

for i in vert_lines:
  unordered_boxes.append(vert_boxes[i][0])

ordered_boxes = np.argsort(unordered_boxes)
# print(ordered_boxes)

def intersection(box_1, box_2):
  return [box_2[0], box_1[1], box_2[2], box_1[3]]

def iou(box_1, box_2):
  x_1 = max(box_1[0], box_2[0])
  y_1 = max(box_1[1], box_2[1])
  x_2 = min(box_1[2], box_2[2])
  y_2 = min(box_1[3], box_2[3])

  inter = abs(max((x_2 - x_1), 0) * max((y_2 - y_1), 0))
  if inter == 0:
    return 0
  
  box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
  box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

  # print("Union is:", float(box_1_area + box_2_area - inter))

  return inter / float(box_1_area + box_2_area - inter)


for i in range(len(horiz_lines)):
  for j in range(len(vert_lines)):
    resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
    # print(resultant)

    for b in range(len(boxes)):
      the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
      if(iou(resultant, the_box) > 0.1):
        out_array[i][j] = texts[b]

# print("Output is:", out_array)

if not os.path.exists("./outputs"):
    os.makedirs("outputs")

print("Core file running")

 # Generate the next filename for saving the output
output_filename = get_next_filename(OUTPUT_FOLDER)

# Convert the array to a DataFrame and save it as a CSV
output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)
df = pd.DataFrame(out_array)

print("DataFrame columns:", df.columns.tolist())

# Calculate the Net Attendance, starting from row 7
net_attendance = []

# Iterate through rows starting from index 6 (7th row)
for index, row in df.iterrows():
    if index < 6:  # Skip rows before the 7th one
        net_attendance.append(None)  # Keep it None or 0 for these rows
        continue

    total_percents = []

    for value in row:
        try:
            # Convert to float
            numeric_value = float(value)
            # Count values greater than 20 as attendance percentages
            if numeric_value > 20:  
                total_percents.append(numeric_value)
        except ValueError:
            # Ignore non-numeric values
            continue

    # Calculate the average percentage for Net Attendance
    if total_percents:  # Ensure there are valid percentages
        net_attendance.append(sum(total_percents) / len(total_percents))
    else:
        net_attendance.append(0)  # Default to 0 if no attendance data

# Add the Net Attendance as a new column to the DataFrame
df['Net Attendance'] = net_attendance

# Save it as a CSV
output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)
df.to_csv(output_file_path, index=False)

print(f"Output saved to {output_file_path}")  # Optional confirmation message