from pymongo import MongoClient
import shutil
import os

def main():

    dev_path = os.environ['DEV_PATH']
    try:
        shutil.rmtree( '%s/output' % dev_path )
        print 'deleted: %s/output' % dev_path
    except:
        pass

    # try:
    #   shutil.rmtree( '%s/model' % dev_path )
    #   print 'deleted: %s/model' % dev_path
    # except:
    #   pass

    try:
        shutil.rmtree( '%s/log' % dev_path )
        print 'deleted: %s/log' % dev_path
    except:
        pass
    # try:
    #   shutil.rmtree( '%s/hist' % dev_path )
    #   print 'deleted: %s/hist' % dev_path
    # except:
    #   pass
    try:

        db = MongoClient()    
        db.drop_database('spearmint')
        print 'dropped database'
    except:
        pass



if __name__ == '__main__':
    main()