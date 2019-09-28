#!/usr/bin/env python
import os
import face_recognition
import time
import argparse
import psutil
from PIL import Image,ImageFont,ImageDraw

if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Facial Recognition Script"
    )
    # Add the positional parameter
    parser.add_argument('GroupPhotoName', help="Name of the Group Photo in which to identify faces with jpg extension.")

    # Add the positional parameter
    parser.add_argument('MatchingTolerance', help="Specify a face matching tolerance .3 to.9 (smaller number is more conservative). ex.  0.5")

    # Add the positional parameter
    parser.add_argument('MatchingModel', help="Specify a Model, ex. hog (fast) or cnn (slow)")

    # Add the positional parameter
    parser.add_argument('FaceSize', help="Specify how small of face to recognize (0-3), 3 is a smaller face")
    
    # Parse the arguments
    arguments = parser.parse_args()
else:
    arguments.GroupPhotoName = "test1.jpg"
    arguments.MatchingTolerance = 0.6
    arguments.MatchingModel="hog"
    arguments.FaceSize=1

script_dir = os.path.dirname(__file__) #<-- absolute dir of script execution
rel_SourcePath = "training-data"
rel_IDImgPath = "test-data"
rel_LabelPath = "labelled-data"
readPath = os.path.join(script_dir, rel_SourcePath)
imgIDPath = os.path.join(script_dir, rel_IDImgPath)
labPath = os.path.join(script_dir, rel_LabelPath)
print("readPath="+readPath)
print("imgIDPath=",imgIDPath)
print("Loading Reference Faces")
# Create arrays of known face encodings and their names
start = time.time()
known_face_names = []
known_face_encodings = []
i=0
for filename in os.listdir(readPath):
    print("filename="+filename)
    #work_image = face_recognition.load_image_file(readPath +"\\" + filename) #windows
    work_image = face_recognition.load_image_file(readPath +"/" + filename)  #linux, ubuntu
    ref_face_encoding = face_recognition.face_encodings(work_image)
    known_face_encodings.append(ref_face_encoding)
    known_face_names.append(filename)
    i=i+1
end = time.time()
print("average reference face load time=" + str((end - start)/i))

# Find all the faces and face encodings in the unknown image, need to parameterize the input of the test image
print("Loading Group Photo")
unkImage = face_recognition.load_image_file(os.path.join(imgIDPath, arguments.GroupPhotoName))
print("Mapping faces in the Group Photo")
unk_face_locations = face_recognition.face_locations(unkImage, number_of_times_to_upsample=int(arguments.FaceSize), model=arguments.MatchingModel) #model="hog"/"cnn"- hog is faster but need to tighten up tolerance in compare faces
unk_face_coord = [],[],[],[]
unk_face_coords = []
idx=0
for idx, face_location in enumerate(unk_face_locations):
    top, right, bottom, left = face_location
  
print("Extracting encodings from mapped faces in Group Photo")
unknown_face_encodings = face_recognition.face_encodings(unkImage, unk_face_locations)
#benchmark memory used
process = psutil.Process(os.getpid())
print("Memory in use="+ str(process.memory_info().rss))  # in bytes 
#find the dimensions of the test image for scaling to rgba image mapping for correct box placement
m,n,a = unkImage.shape
source_img = Image.open(os.path.join(imgIDPath,arguments.GroupPhotoName)).convert("RGBA")
width, height = source_img.size

print("Total faces found in Group Photo =" + str(len(unknown_face_encodings)))
index=0
index1=0
source_img = Image.open(os.path.join(imgIDPath,arguments.GroupPhotoName)).convert("RGBA")
for index, unknown_face_encoding in enumerate(unknown_face_encodings):
    match = False
    # search unknown faces for match to references
    for index1, known_face_encoding in enumerate(known_face_encodings):
          matches = face_recognition.compare_faces(known_face_encoding, unknown_face_encoding, tolerance=float(arguments.MatchingTolerance)) #smaller tolerance means more strict matching
          # create image with correct size and black background
          label_img = Image.new('RGBA', source_img.size,  (255,255,255,0))
          #only 1 to 1 match test so index is 0
          if matches[0] == True:
              print("Known Face Match name=" + known_face_names[index1] + " for unknown face index=" + str(index))
              font = ImageFont.truetype("VeraMono.ttf", 35) #linux, ubuntu
              #font = ImageFont.truetype("arial", 35) #windows
              text = known_face_names[index1]
              label_draw = ImageDraw.Draw(label_img, "RGBA")
              for idx, face_location in enumerate(unk_face_locations):
                  if idx==index:
                      top, right, bottom, left = face_location
              label_draw.text((left, top), text, fill=(177,255,47,255), font=font) #fill of 255,0,0 is dark red, 50,205,50 is lime green,  177,255,47 is green-yellow
              combined_img = Image.alpha_composite(source_img, label_img) 
              source_img = combined_img
              match=True            
    if match == False:
        font = ImageFont.truetype("VeraMono.ttf", 20) #linux, ubuntu
        #font = ImageFont.truetype("arial", 35) #windows
        text = "Unknown " + str(index)
        for idx, face_location in enumerate(unk_face_locations):
            if idx==index:
                top, right, bottom, left = face_location
        label_draw = ImageDraw.Draw(label_img, "RGBA")
        label_draw.text((left,top), text, fill=(255,0,0,255), font=font) #fill is dark red
        combined_img = Image.alpha_composite(source_img, label_img)
        source_img = combined_img
              
source_img.show()
time.sleep(6)

              



