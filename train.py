import numpy as np
import tensorflow as tf
import os.path as osp
import cv2, random
import cPickle
import model, data
import IPython, sys, os

flags = tf.app.flags

model_dir = "model"

# input argument
flags.DEFINE_integer('batch_size', 64, 'Value of batch size')
flags.DEFINE_float('lr', 0.00001, 'learing rate')
flags.DEFINE_boolean('test', False, 'Whether test')
flags.DEFINE_string('model', '', 'Model path')

FLAGS = flags.FLAGS
# Use appropiate GPU memory instead of full
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# cifar 10
CLASS_NUMBER = 10  # the class number of dataset 
# oxford flower 17
#CLASS_NUMBER = 17  # the class number of dataset 

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def train(dataset_train, dataset_val, finetune='',caffemodel=''):
    """ main train function for training and save the model of min-valiation loss"""
    batch_size = FLAGS.batch_size
    with tf.Graph().as_default():
        # Graph of AlexNet
        images = tf.placeholder('float32', shape=(None, 227, 227, 3))
        labels = tf.placeholder('int64', shape=(None))
        keep_prob_ = tf.placeholder('float32') # dropout rate

        logits = model.inference(images, keep_prob_, CLASS_NUMBER)

        loss = model.loss(logits, labels)

        optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.lr).minimize(loss)


        # initialize all variables in graph
        init_op = tf.initialize_all_variables()


        step = 0
        epo = 1

        # store 5 latest model
        saver = tf.train.Saver(max_to_keep=5)
        min_loss = sys.maxint


        # Session of training prcoess
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # load pretrained alexnet model training by imagenet
            
            if caffemodel:
                model.load_alexnet(sess, caffemodel)
                print('loaded pretrained caffemodel: {}'.format(caffemodel))
                # load pretrained model ourselves
            elif finetune:
                saver.restore(sess, finetune)
                print('loaded finetune model: {}'.format(caffemodel))
            

            # training epo
            while epo <= 100:
                # get batch data from dataset     
                batch_x, batch_y, isnextepoch = dataset_train.sample(batch_size)
                feed_dict = {images: batch_x, labels: batch_y, keep_prob_: 0.5}

                # run graph and backpropagation
                _, loss_value= sess.run([optimizer, loss], feed_dict=feed_dict)
                step += len(batch_y)

                #print step:
                if step/batch_size %10 ==0:
                    print('epo{}: {}/{}, loss = {}'.format(epo, step, len(dataset_train), loss_value))
                

                # epo end
                if isnextepoch:
                    val_loss_sum = []
                    isnextepoch = False # set for validation 

                    # Validation process
                    while not isnextepoch:
                        val_x, val_y, isnextepoch = dataset_val.sample(batch_size)
                        feed_dict = {images: batch_x, labels: batch_y, keep_prob_: 1.}
                        logit, val_loss = sess.run([logits, loss], feed_dict=feed_dict)
                        val_loss_sum.append(val_loss)
                        
                    val_loss = np.mean(val_loss_sum)
                    print('validation loss: {}'.format(val_loss))

                    # if validation is good, save model
                    if min_loss > val_loss:
                        print('Save model...')
                        saver.save(sess, osp.join(model_dir, 'model_best'))
                        saver.save(sess, osp.join(model_dir, 'val_{}_{}'.format(epo, val_loss)))
                        min_loss = val_loss

                    epo += 1
                    step = 0

                    # shuffle dataset to prevent overfitting
                    dataset_train.reset_sample()




def test(dataset_test, model_path):
    batch_size = FLAGS.batch_size
    # Foward graph
    with tf.Graph().as_default():
        images  = tf.placeholder('float32', shape=(None, 227, 227, 3))
        labels = tf.placeholder('int64', shape=(None))

        logits = model.inference(images, 1., CLASS_NUMBER)
        loss = model.loss(logits, labels)
        prediction = model.classify(logits)


        saver = tf.train.Saver()

        #session for testing
        with tf.Session(config=config) as sess:
       
            if not model_path:
                saver.restore(sess, osp.join(model_dir ,'model_best'))
            else:
                saver.restore(sess, model_path)
            
            accuracy_sum = []
            isnextepoch = False
            step = 0
            loss_value_sum = []
            while not isnextepoch:
                batch_x, batch_y, isnextepoch = dataset_test.sample(batch_size)
                step += len(batch_x)
                feed_dict = {images: batch_x, labels: batch_y} 
                l, pred, loss_value= sess.run([logits, prediction, loss], feed_dict=feed_dict)
                loss_value_sum.append(loss_value)
                #IPython.embed()
                accuracy = 0.
                for index in xrange(len(pred)):
                    if pred[index] == batch_y[index]:
                        accuracy += 1
                accuracy /= len(pred)
                accuracy_sum.append(accuracy)
                sys.stdout.write("\r{:7d}/{}".format(step, len(dataset_test)))
                sys.stdout.flush()
            print('\nTest loss: {},Accuarcy: {}'.format(np.mean(loss_value_sum), np.mean(accuracy_sum)) )


def unpickle(filename):
    with open(filename, 'rb') as f:
        dic = cPickle.load(f)
    return dic

def create_training():
    """create training dataset"""
    dataset = []
    """
    # for oxford floswer 17
    labels = [0]

    img_dir = "jpg"
    with open('labels.txt','rb') as f:
        for line in f:
            labels.append(int(line.strip())-1)
    for img in os.listdir(img_dir):
        if img.find('.jpg') ==-1: #.txt file
            continue 
        index = int(img.replace('image_','').replace('.jpg',''))


        dataset.append(data.ImageClassData(cv2.imread(osp.join(img_dir, img), cv2.IMREAD_COLOR), labels[index], img))

        sys.stdout.write("\r{:7d}".format(len(dataset)))
        sys.stdout.flush()

    """
    # for cifar10 
    print('Loading cifar 10 training data...')
    for batch_index in xrange(1, 6):
        batch_dic = unpickle('cifar-10-batches-py/data_batch_{}'.format(batch_index))
        for index in xrange(len(batch_dic['data'])):
            dataset.append(data.ImageClassData(batch_dic['data'][index], batch_dic['labels'][index]))
            sys.stdout.write("\r{:7d}".format(len(dataset)))
            sys.stdout.flush()
    
    val_ratio = 1./5 # split for validation
    split_index = int(len(dataset) *val_ratio)
    random.shuffle(dataset)

    return data.Dataset(dataset[split_index:]), data.Dataset(dataset[:split_index])

def create_testing():
    """create testing dataset"""
    dataset = []
    """
    # for oxford floswer 17
    labels = [0]
    img_dir = "jpg"
    with open('labels.txt','rb') as f:
        for line in f:
            labels.append(int(line.strip())-1)
    for img in os.listdir(img_dir):
        if img.find('.jpg') ==-1: #.txt file
            continue 
        index = int(img.replace('image_','').replace('.jpg',''))
        dataset.append(data.ImageClassData(
            cv2.imread(osp.join(img_dir, img), cv2.IMREAD_COLOR), labels[index], img))

        sys.stdout.write("\r{:7d}".format(len(dataset)))
        sys.stdout.flush()

    """
    # for cifar10 
    print('Loading cifar 10 testing data...')
    batch_dic = unpickle('cifar-10-batches-py/test_batch')
    for index in xrange(len(batch_dic['data'])):
        dataset.append(data.ImageClassData(batch_dic['data'][index], batch_dic['labels'][index]))
        sys.stdout.write("\r{:7d}".format(len(dataset)))
        sys.stdout.flush()
    
    return data.Dataset(dataset)


def main(argv):
    # if test
    if not FLAGS.test:
        train_data, val_data = create_training()
        # Fine-tune on own dataset
        if FLAGS.model:
            train(train_data, val_data, finetune=FLAGS.model)
        else:
            train(train_data, val_data, caffemodel='alexnet_imagenet.npy')
    # if training
    else: 
        test_data = create_testing()
        test(test_data, FLAGS.model)


if __name__ == '__main__':
    main(sys.argv)