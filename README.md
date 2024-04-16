# FacePointDetectoin


This repo contain training-testing pipeline for custom neural network architecture.
The goal is to compare custom trained model with trained dlib face-landmark detector -  http://dlib.net/face_landmark_detection.py.html

## Data
* Menpo
* 300W ( dlib was trained on that dataset )
* Face extraction was done with dlib detector ( http://dlib.net/face_detector.py.html )
* Also : cleaning datasets function can be fount in `ūtils/clean_data.py` (drop duplicates, extend coped dlib images)
* Augmentations : 
    ```commandline
          transform = A.Compose([
                      A.RandomBrightnessContrast(p=0.4),
                      A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.3, p=0.1),
                      A.RandomSunFlare(flare_roi=(0, 0, 0.8, 0.1), angle_upper=0.1, p=0.1),
                      A.HorizontalFlip(p=0.5),
                      A.Rotate(p=0.5),
            
                  ], keypoint_params= A.KeypointParams(format='xy', remove_invisible=False))
    ```
## How reproduce:
1. ```git clone```
2. define your custom architecture in `models.py` (optional)
3. setup path to your train / test data in `def train()` in `train_test.py`
4. run ``train()`` in `train_test.py`
4. test your model by calculating Cumulative Error Distribution (`def test_CED()`) - that will save plot like that in desired directory:

  ![ced.png](https://github.com/IrDIE/FacePointDetectoin/blob/main/readme_utils/ced_dlib.png)


