from PIL import Image
import model
import argparse
import numpy as np
import time
import align_face
import os


_TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
_TAGS = _TAGS.split()

def get_glow(input_image, tag, alpha):
	if input_image == "all":
		path = 'Input_test'
		for file_name in os.listdir(path):
			file_input = "Input_test/" + file_name
			get_tag(file_input, 7, tag, alpha)
	else:
		file_input = "Input_test/" + input_image
		get_tag(file_input, 3, tag, alpha)


def get_tag(file_input, number, tag, alpha):
	eps = encode(file_input)
	if tag == "all":
		for tag in _TAGS:
			get_alpha(file_input, eps, number, tag, alpha)
	else:
		number -= 2
		get_alpha(file_input, eps, number, tag, alpha)


def get_alpha(file_input, eps, number, tag, alpha):
	if alpha > 1:
		for x in range(int(alpha)):
			ralpha = round(-1 + x * 2 / (alpha - 1), 2)
			file_output = "Output_test" + str(number) + "/" + os.path.splitext(os.path.split(file_input)[1])[0] + '_' + tag + '_' + str(x) + '.png'
			manipulate(eps, file_output, tag, ralpha)
	else:
		number -= 1
		file_output = "Output_test" + str(number) + "/" + os.path.splitext(os.path.split(file_input)[1])[0] + '_' + tag + '_' + str(alpha) + '.png'
		manipulate(eps, file_output, tag, alpha)

def align_image(input_path, output_path):
	img = align_face.align(input_path)
	img = Image.fromarray(img).convert('RGB')
	img.save(output_path)

def encode(input_path):
	img = Image.open(input_path)
	img = align_face.align(input_path)
	img = np.reshape(img, [1,256,256,3])

	t = time.time()
	eps = model.encode(img)
	print("Encoding latency {} sec/img".format(time.time() - t))
	return eps

def decode(eps, tag, alpha):
	t = time.time()
	dec, _ = model.manipulate(eps, _TAGS.index(tag), alpha)
	print("Manipulating latency {} sec/img".format(time.time() - t))
	img = Image.fromarray(dec[0])
	return img

def manipulate(eps, output_path, tag, alpha):
	img = decode(eps, tag, alpha)
	img.save(output_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "GLOW")

	parser.add_argument('--input_image', help = 'path image input', type = str, default = "rashida.png")
	parser.add_argument('--output_image', help = 'path image output', type = str, default = "smiling.png")
	parser.add_argument('--tag', help = 'attribute', type = str, default = "Smiling")
	parser.add_argument('--alpha', help = 'number manipulate', type = float, default = 0.66)

	args = parser.parse_args()

	input_image = args.input_image
	tag = args.tag
	alpha = args.alpha

	get_glow(input_image, tag, alpha)
	