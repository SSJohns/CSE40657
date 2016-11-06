from language_model import *
from language_model2 import *


def main(args):
	lm = Language_Model(args)

	lmc = Language_Model2('chinese')

if __name__ == '__main__':
	# python main.py language path/train
	main('english')
