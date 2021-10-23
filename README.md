# bob-plda-docker
A docker container for bob (bob.learn.em) signal processing, machine learning, and biometrics toolkit in python 3.5

### docker

`docker build -t nabil/bob-plda-docker .`

`docker run -i -t nabil/bob-plda-docker /bin/bash`

### conda

`conda activate bob`

### python

```python
import bob.learn.em
```

```python
python plda_train.py
```
