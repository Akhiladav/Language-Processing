import tensorflow as tf
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100

print(len(train_x[0]))

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')



accuracy=tf.Variable([0],name="accuracy")
W = {
        'h1': tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1]), name='wh1'),
        'h2': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]), name='wh2'),
        'h3': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]), name='wh3'),
        'wo0': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes]), name='wo')
    }
b = {
        'b1': tf.Variable(tf.random_normal([n_nodes_hl1]), name='bh1'),
        'b2': tf.Variable(tf.random_normal([n_nodes_hl2]), name='bh2'),
        'b3': tf.Variable(tf.random_normal([n_nodes_hl3]), name='bh3'),
        'bo0': tf.Variable(tf.random_normal([n_classes]), name='bo')
    }  
    
saver = tf.train.Saver(save_relative_paths=True)

def neural_network_model(data):
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
  l1= tf.add(tf.matmul(data,W['h1']),b['b1'])
  l1=tf.nn.relu(l1)
  
  l2= tf.add(tf.matmul(l1,W['h2']),b['b2'])
  l2=tf.nn.relu(l2)
  
  l3= tf.add(tf.matmul(l2,W['h3']),b['b3'])
  l3=tf.nn.relu(l3)
  
  output= tf.add(tf.matmul(l3,W['wo0']),b['bo0'])
 
  return output

def train_neural_network(input_data):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
              start = i
              end = i+batch_size
              batch_x = np.array(train_x[start:end])
              batch_y = np.array(train_y[start:end])
              _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
              epoch_loss += c
              i+=batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        saver.save(sess, "drive/My Drive/ml4/mymodel")
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)
