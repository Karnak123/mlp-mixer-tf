# mlp-mixer-tf


Simple tensorflow implementation of MLP-Mixer.

## Example usage
```python
from mlp_mixer import MLPMixer
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
model = MLPMixer(input_shape=x_train.shape[1:],
	num_classes=len(np.unique(y_train)), 
	num_blocks=4, 
	patch_size=8,
	hdim=32, 
	tokens_mlp_dim=64,
	channels_mlp_dim=128)
model.compile(loss='sparse_categorical_crossentropy', metrics='acc')
history = model.fit(x_train, y_train, validation_data=(x_test, y_test))
model.summary()
```
