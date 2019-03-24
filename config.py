import os
import logging
import logging.handlers
import tarfile
import glob

### Logging
import logging
import logging.handlers
from datetime import datetime

from vision.utils import run_tensorboard

DATASET_DIR = '/raid/datasets/'
ROOT_LOGGER_NAME = 'SSD300'


def initialize_logger(args):
	
	if args.exp_time is None:
	    args.exp_time        = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
	exp_name        = '_' + args.exp_name if args.exp_name is not None else ''
	args.jobs_dir        = os.path.join( 'jobs', args.exp_time + '_' + args.net + exp_name )

	snapshot_dir    = os.path.join( args.jobs_dir, 'snapshots' )
	tensorboard_dir    = os.path.join( args.jobs_dir, 'tensorboardX' )
	if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)
	if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)
	run_tensorboard( tensorboard_dir, args.port )


	### Backup current source codes
	tar = tarfile.open( os.path.join(args.jobs_dir, 'sources.tar'), 'w' )
	tar.add( 'vision' )
	for file in glob.glob('*.py'):
		tar.add( file )
	tar.close()


	### Logger
	# logging.basicConfig(format='[%(levelname)s][%(asctime)s][%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	fmt = logging.Formatter('[%(levelname)s][%(asctime)s][%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

	logger = logging.getLogger(ROOT_LOGGER_NAME)
	logger.setLevel(logging.INFO)

	h = logging.StreamHandler()
	h.setFormatter(fmt)
	logger.addHandler(h)

	h = logging.FileHandler(os.path.join(args.jobs_dir, 'log_{:s}.txt'.format(args.exp_time)))
	h.setFormatter(fmt)
	logger.addHandler(h)
	
	### Exp settings
	settings = vars(args)
	for key, value in settings.items():
		settings[key] = value   

	logger.info('Exp time: {}'.format(settings['exp_time']))
	for key, value in settings.items():
		if key == 'exp_time':
			continue
		logger.info('\t{}: {}'.format(key, value))

	return logger, args
