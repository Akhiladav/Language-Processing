import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
lemmatizer = WordNetLemmatizer()



# here we aredefining how many nodes each layer should have..
# input layer has already 28 X 28 nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10

# We also can compile whole data set in one go...but it is good to go 
# with 100 data ate once
batch_size = 100

# PlaceHolders for some values
x = tf.placeholder('float', [None, 423])
y = tf.placeholder('float')





def nural_network_model(data,w1,b1,w2,b2,w3,b3,wo,bo):
  '''
  hidden_1_layer ={'weights' : tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                   'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
  hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                   'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
  hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                   'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
  output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                   'biases' : tf.Variable(tf.random_normal([n_classes]))}
  '''
  l1= tf.add(tf.matmul(([data]),w1),b1)
  l1=tf.nn.relu(l1)
  
  l2= tf.add(tf.matmul(l1,w2),b2)
  l2=tf.nn.relu(l2)
  
  l3= tf.add(tf.matmul(l2,w3),b3)
  l3=tf.nn.relu(l3)
  
  output= tf.add(tf.matmul(l3,wo),bo)
 
  return output



    

def use_neural_network(input_data):
  
  
    #prediction = nural_network_model(x)
    tf.reset_default_graph()  
    saver2 = tf.train.import_meta_graph('drive/My Drive/ml4/mymodel.meta')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver2.restore(sess, 'drive/My Drive/ml4/mymodel')
        graph = tf.get_default_graph()
        
        with open('drive/My Drive/lexicon.pickle','rb') as f:
            lexicon = pickle.load(f)
        print("lexion",len(lexicon))
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                print(index_value)
                # OR DO +=1, test both
                features[index_value] += 1

        #features = np.array(list(features))
        
        #features=np.array(((features)))
        features=np.array(features)
        features=tf.convert_to_tensor(features, np.float32)
        print("features",sess.run(features))
        w1 = graph.get_tensor_by_name("wh1:0")
        b1 = graph.get_tensor_by_name("bh1:0")
        w2 = graph.get_tensor_by_name("wh2:0")         
        b2 = graph.get_tensor_by_name("bh2:0")
        w3 = graph.get_tensor_by_name("wh3:0")
        b3 = graph.get_tensor_by_name("bh3:0")
        wo = graph.get_tensor_by_name("wo:0")         
        bo = graph.get_tensor_by_name("bo:0")
        result=nural_network_model(features,w1,b1,w2,b2,w3,b3,wo,bo)
        result=tf.reshape(result, [2])
        print(sess.run(result))
        print(sess.run(tf.argmax(result)))
        '''
        wh1 = graph.get_tensor_by_name("wh1:0")
        print(sess.run(wh1))
        
    
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[input_data]}),1)))
        print(result[0])
            
            
        '''
use_neural_network("everyone blames her")
