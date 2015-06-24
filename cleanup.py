import pymongo
import shutil
import json
import os
from spearmint.utils.parsing import parse_db_address

def main():

	dev_path = os.environ['DNN_PATH']
	try:
		shutil.rmtree( '%s/output' % dev_path )
		print 'deleted: %s/output' % dev_path
	except:
		pass

	try:
		shutil.rmtree( '%s/log' % dev_path )
		print 'deleted: %s/log' % dev_path
	except:
		pass

	drop_db(dev_path)


def drop_db(path):

    if not os.path.isdir(path):
        raise Exception("%s is not a valid directory" % path)

    with open(os.path.join(path, 'config.json'), 'r') as f:
        cfg = json.load(f)

    db_address = parse_db_address(cfg)
    print 'Cleaning up experiment %s in database at %s' % (cfg["experiment-name"], db_address)

    client = pymongo.MongoClient(db_address)

    db = client.spearmint
    db[cfg["experiment-name"]]['jobs'].drop()
    db[cfg["experiment-name"]]['hypers'].drop()
    del client





if __name__ == '__main__':
	main()