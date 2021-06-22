from PIL import Image 
import sys


def printImageAttributes(imageObject, imagePath):

	# Retrieve the attributes of the image

	fileFormat      = imageObject.format        # Format of the image

	imageMode       = imageObject.mode          # Mode of the image

	imageSize       = imageObject.size          # Size of the image - tupe of (width, height)

	colorPalette    = imageObject.palette       # Palette used in the image
	# Print the attributes of the image

	print("%dx%d"%imageSize)

# 	print("Attributes of image:%s"%imagePath)
# 
# 
# 
# 	print("The file format of the image is:%s"%fileFormat)
# 
# 	print("The mode of the image is:%s"%imageMode)
# 
# 	print("The size of the image is:width %d pixels,height %d pixels"%imageSize)
# 
# 	print("Color palette used in image:%s"%colorPalette)
# 
# 
# 
# 	print("Keys from image.info dictionary:%s")
# 
# 	for key, value in imageObject.info.items() :
# 		print(key)



if len(sys.argv) < 3:
	quit()

	
input =sys.argv[1]
output=sys.argv[2]


img = Image.open(input)
printImageAttributes(img,input)
img.save(output)
