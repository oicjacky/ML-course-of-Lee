## [Deploying PyTorch in Python via a REST API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html#api-definition)

In this tutorial, we will deploy a PyTorch model using Flask and expose a REST API for model inference. In particular, we will deploy a pretrained DenseNet-121 model which detects the image.

- DenseNet-121 model from [ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/).
- [torchvision.datasets.ImageNet](https://pytorch.org/vision/0.8/datasets.html#imagenet)

## Usage

```bash
set FLASK_ENV=development
set FLASK_APP=app.py
flask run
```

Then use a python terminal to send a POST request to our app with following:
```python
import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(r'C:\Users\user\Pyvirtualenv\course_CNN\images\hi-res-9021ebe2b61de5b19dbe52f63b3e651c_crop_north.jpg','rb')})
print(resp.json())
```

## Next steps

- The endpoint `/predict` assumes that always there will be a image file in the request. This may not hold true for all requests.

- The user may send non-image type files.

- The model may not be able to recognize all images.


## Reference
- Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>