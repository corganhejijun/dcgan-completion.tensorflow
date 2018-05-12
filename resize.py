import os, sys
import Image

size = 128

root = sys.argv[1]
for infile in os.listdir(root):
    outfile = os.path.join(sys.argv[2], infile)
    print outfile
    im = Image.open(os.path.join(root, infile))
    img = im.resize((size, size),Image.ANTIALIAS)  
    img.save(outfile, "JPEG")
