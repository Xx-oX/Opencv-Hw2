import random
import numpy as np  
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.python.keras.preprocessing import image

model = load_model('./model-resnet50.h5')
path = './test_set/'
cls_list = ['cats', 'dogs']


while True:
	k = input()
	if k == 'exit':
		break
		
	if k == '1':
		img = plt.imread('./log.png')
		plt.axis('off')
		plt.imshow(img)
		plt.show()
	
	if k == '2':
		img = plt.imread('./tensorBoardResult.png')
		plt.axis('off')
		plt.imshow(img)
		plt.show()
	
	if k == '3':
		file = ''
		type = random.randint(0, 1)
		num = random.randint(4001, 5000)
		print(type, num)
		if type == 0:
			file = 'cat.' + str(num) + '.jpg'
		else:
			file = 'dog.' + str(num) + '.jpg'
			
			
		img = image.load_img(path + file, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis = 0)
		pred = model.predict(x)[0]
		top_inds = pred.argsort()[::-1][:5]
		
		img_show = plt.imread(path + file)
		# plt.title(cls_list[top_inds[0]])
		plt.title(cls_list[type])
		plt.imshow(img_show)
		plt.show()
		
	if k == '4':
		img = plt.imread('./resize.png')
		plt.axis('off')
		plt.imshow(img)
		plt.show()
		
		
		
		
