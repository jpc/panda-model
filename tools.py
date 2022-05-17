from fastai.vision.all import *
import tifffile

imgdir = Path('/scratch/train_images')

def get_crops(x):
    tile_size = 250
    if type(x) == PILImage:
        img = np.array(x)
    else:
        tiff_file = imgdir/f'{x["image_id"]}.tiff'
        img = tifffile.imread(tiff_file, key=1)
    crop = np.array(img.shape) // tile_size * tile_size; crop
    imgc = img[:crop[0],:crop[1]]
    imgc = imgc.reshape(imgc.shape[0] // tile_size, tile_size, imgc.shape[1] // tile_size, tile_size, 3)
    xs, ys = (imgc.mean(axis=1).mean(axis=2).mean(axis=-1) < 252).nonzero()
    if len(xs) == 0:
        xs, ys = (imgc.mean(axis=1).mean(axis=2).mean(axis=-1)).nonzero()
#     if len(xs) < 2: print("no data in image:", x)
    pidxs = random.choices(list(range(len(xs))), k=36)
    return PILImage.create(imgc[xs[pidxs],:,ys[pidxs],:].reshape(6,6,tile_size,tile_size,3).transpose(0,2,1,3,4).reshape(6*tile_size,6*tile_size,3))
#     return imgc.mean(axis=1).mean(axis=2).mean(axis=-1)

def get_labels(x):
    return np.arange(5) <= x['isup_grade']
