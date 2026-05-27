from yolov5 import train, detect  # pseudo-code for running

# Run detection on image
!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/

# Output saved to runs/detect
