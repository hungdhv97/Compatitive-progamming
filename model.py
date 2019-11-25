import numpy as np 
import tensorflow as tf 
import time
from tqdm import tqdm 
from PIL import Image
from threading import Lock

lock = Lock()

def get(name):
	return tf.get_default_graph().get_tensor_by_name('import/' + name + ':0')

def tensorflow_session():
	config = tf.ConfigProto()
	config.gpu_options.visible_device_list = str(0)
	sess = tf.Session(config=config)
	return sess

graph_path = 'graph_unoptimized.pb'
inputs = {
	'dec_eps_0': 'Placeholder',
	'dec_eps_1': 'Placeholder_1',
	'dec_eps_2': 'Placeholder_2',
	'dec_eps_3': 'Placeholder_3',
	'dec_eps_4': 'Placeholder_4',
	'dec_eps_5': 'Placeholder_5',
	'enc_x': 'input/image',
	'enc_x_d': 'input/downsampled_image',
	'enc_y': 'input/label'
}
outputs = {
	'dec_x': 'model_1/Cast_1',
	'enc_eps_0': 'model/pool0/truediv_1',
	'enc_eps_1': 'model/pool1/truediv_1',
	'enc_eps_2': 'model/pool2/truediv_1',
	'enc_eps_3': 'model/pool3/truediv_1',
	'enc_eps_4': 'model/pool4/truediv_1',
	'enc_eps_5': 'model/truediv_4'
}

def update_feed(feed_dict, bs):
	x_d = 128 * np.ones([bs, 128, 128, 3], dtype=np.uint8)
	y = np.zeros([bs], dtype=np.int32)
	feed_dict[enc_x_d] = x_d
	feed_dict[enc_y] = y
	return feed_dict

with tf.gfile.GFile(graph_path, 'rb') as f:
	graph_def_optimized = tf.GraphDef()
	graph_def_optimized.ParseFromString(f.read())

sess = tensorflow_session()
tf.import_graph_def(graph_def_optimized)

print("Loaded model")


# Encoder
n_eps = 6
enc_x = get(inputs['enc_x'])
enc_eps = [get(outputs['enc_eps_' + str(i)]) for i in range(n_eps)]

enc_x_d = get(inputs['enc_x_d'])
enc_y = get(inputs['enc_y'])

# Decoder
dec_x = get(outputs['dec_x'])
dec_eps = [get(inputs['dec_eps_' + str(i)]) for i in range(n_eps)]


eps_shapes = [(128, 128, 6), (64, 64, 12), (32, 32, 24),
              (16, 16, 48), (8, 8, 96), (4, 4, 384)]
eps_sizes = [np.prod(e) for e in eps_shapes]
eps_size = 256 * 256 * 3
z_manipulate = np.load('z_manipulate.npy')

_TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
_TAGS = _TAGS.split()

flip_tags = ['No_Beard', 'Young']
for tag in flip_tags:
	i = _TAGS.index(tag)
	z_manipulate[i] = -z_manipulate[i]

scale_tags = ['Narrow_Eyes']
for tag in scale_tags:
	i = _TAGS.index(tag)
	z_manipulate[i] = 1.2 * z_manipulate[i]

z_sq_norms = np.sum(z_manipulate**2, axis=-1, keepdims=True)
z_proj = (z_manipulate / z_sq_norms).T 


def run(sess, fetches, feed_dict):
	with lock:
		# Khóa tensorflow giảm thời gian phản hồi của máy chủ
		result = sess.run(fetches, feed_dict)
	return result

def flatten_eps(eps):
	# [BS, eps_size]
	return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)

def unflatten_eps(feps):
	index = 0
	eps = []
	bs = feps.shape[0] # feps_size // eps_size
	for shape in eps_shapes:
		eps.append(np.reshape(feps[:, index: index+np.prod(shape)], (bs, *shape)))
		index += np.prod(shape)
	return eps		


def encode(img):
	if len(img.shape) == 3:
		img = np.expand_dims(img, 0)
	bs = img.shape[0]
	assert img.shape[1:] == (256, 256, 3)
	feed_dict = {enc_x: img}

	update_feed(feed_dict, bs) # Dành cho unoptimized model
	return flatten_eps(run(sess, enc_eps, feed_dict))


def decode(feps):
	if len(feps.shape) == 1:
		feps = np.enpand_dims(feps, 0)
	bs = feps.shape[0]
	eps = unflatten_eps(feps)

	feed_dict = {}
	for i in range(n_eps):
		feed_dict[dec_eps[i]] = eps[i]

	update_feed(feed_dict, bs) # Dành cho unoptimized model
	return run(sess, dec_x, feed_dict)

def project(z):
	return np.dot(z, z_proj)


def _manipulate(z, dz, alpha):
	z = z + alpha * dz
	return decode(z), z

def _manipulate_range(z, dz, points, scale):
	z_range = np.concatenate([z + scale*(pt/(points - 1)) * dz for pt in range(0, points)], axis=0)
	return decode(z_range), z_range


# alpha từ [0,1]
def mix(z1, z2, alpha):
	dz = (z2 - z1)
	return _manipulate(z1, dz, alpha)

def mix_range(z1, z2, points=5):
	dz = (z2 - z1)
	return _manipulate_range(z1, dz, points, 1.)

# alpha từ [-1,1]
def manipulate(z, typ, alpha):
	dz = z_manipulate[typ]
	return _manipulate(z, dz, alpha)

def manipulate_all(z, typs, alphas):
	dz = 0.0
	for i in range(len(typs)):
		dz += alphas[i] * z_manipulate[typs[i]]
	return _manipulate(z, dz, 1.0)

def manipulate_range(z, typ, points=5, scale=1):
	dz = z_manipulate[typ]
	return _manipulate_range(z - dz, 2*dz, points, scale)

def random(bs=1, eps_std=0.7):
	feps = np.random.normal(scale=eps_std, size=[bs, eps_size])
	return decode(feps), feps

# _img, _z = random(1)
# _z = encode(_img)
print("Warn started tf model")