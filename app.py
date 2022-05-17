import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('panda-model-1')

labels = learn.dls.vocab

def get_crops(img):
    tile_size = 250
    img = np.array(img)
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

def predict(img):
    img = get_crops(PILImage.create(img))
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Prostate cANcer graDe Assessment model"
description = "A model to predict the ISUP grade for prostate cancer based on whole-slide images of digitized H&E-stained biopsies."
# article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
examples = ['test.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(224, 224)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch()
