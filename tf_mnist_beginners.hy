(import [tensorflow.examples.tutorials.mnist [input_data]])
(setv mnist (input_data.read_data_sets "MNIST_data/" :one_hot True))

(import [tensorflow :as tf])

(setv x (tf.placeholder tf.float32 [None 784]))

(setv W (tf.Variable (tf.zeros [784 10])))
(setv b (tf.Variable (tf.zeros 10)))

(setv y (tf.nn.softmax(+ (tf.matmul x W) b)))

(setv y_ (tf.placeholder tf.float32 [None 10]))

(setv cross_entropy (tf.reduce_mean (- (tf.reduce_sum (* y_ (tf.log y)) :reduction_indices 1))))

(setv train_step (.minimize (tf.train.GradientDescentOptimizer 0.5) cross_entropy))

(setv sess (tf.InteractiveSession))

(.run (tf.global_variables_initializer))

(for [_ (range 1000)]
    (setv [batch_xs batch_ys] (mnist.train.next_batch 100))
    (sess.run train_step :feed_dict {x batch_xs y_ batch_ys}))

(setv correct_prediction (tf.equal (tf.argmax y 1) (tf.argmax y_ 1)))

(setv accuracy (tf.reduce_mean (tf.cast correct_prediction tf.float32)))

(print (sess.run accuracy :feed_dict {x mnist.test.images y_ mnist.test.labels}))

