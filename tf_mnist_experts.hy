(import [tensorflow.examples.tutorials.mnist [input_data]])
(setv mnist (input_data.read_data_sets "MNIST_data/" :one_hot True))

(import [tensorflow :as tf])
(setv sess (tf.InteractiveSession))

(setv x (tf.placeholder tf.float32 [None 784]))
(setv y_ (tf.placeholder tf.float32 [None 10]))

(setv W (tf.Variable (tf.zeros [784 10])))
(setv b (tf.Variable (tf.zeros 10)))

(sess.run (tf.global_variables_initializer))
(setv y (+ (tf.matmul x W) b))
(setv cross_entropy (tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits :labels y_ :logits y)))
(setv train_step (.minimize (tf.train.GradientDescentOptimizer 0.5) cross_entropy))

(for [_ (range 1000)]
     (setv batch (mnist.train.next_batch 100))
     (sess.run train_step :feed_dict {x (get batch 0) y_ (get batch 1)}))

(setv correct_prediction (tf.equal (tf.argmax y 1) (tf.argmax y_ 1)))
(setv accuracy (tf.reduce_mean (tf.cast correct_prediction tf.float32)))
(print (accuracy.eval :feed_dict {x mnist.test.images y_ mnist.test.labels}))




(defn weight_variable [shape]
     (setv initial (tf.truncated_normal shape :stddev 0.1))
     (tf.Variable initial))

(defn bias_variable [shape]
     (setv initial (tf.constant 0.1 :shape shape))
     (tf.Variable initial))

(defn conv2d [x W]
     (tf.nn.conv2d x W :strides [1 1 1 1] :padding "SAME"))

(defn max_pool_2x2 [x]
     (tf.nn.max_pool x :ksize [1 2 2 1] :strides [1 2 2 1] :padding "SAME"))

(setv W_conv1 (weight_variable [5 5 1 32]))
(setv b_conv1 (bias_variable [32]))

(setv x_image (tf.reshape x [-1 28 28 1]))
(setv h_conv1 (tf.nn.relu(+ (conv2d x_image W_conv1) b_conv1))) 
(setv h_pool1 (max_pool_2x2 h_conv1))

(setv W_conv2 (weight_variable [5 5 32 64]))
(setv b_conv2 (bias_variable [64]))

(setv h_conv2 (tf.nn.relu(+ (conv2d h_pool1 W_conv2) b_conv2))) 
(setv h_pool2 (max_pool_2x2 h_conv2))

(setv W_fc1 (weight_variable [(* 7 7 64) 1024]))
(setv b_fc1 (bias_variable [1024]))
(setv h_pool2_flat (tf.reshape h_pool2 [-1 (* 7 7 64)]))
(setv h_fc1 (tf.nn.relu (+ (tf.matmul h_pool2_flat W_fc1) b_fc1)))

(setv keep_prob (tf.placeholder tf.float32))
(setv h_fc1_drop (tf.nn.dropout h_fc1 keep_prob))

(setv W_fc2 (weight_variable [1024 10]))
(setv b_fc2 (bias_variable [10]))
(setv y_conv (+ (tf.matmul h_fc1_drop W_fc2) b_fc2))

(setv cross_entropy (tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits :labels y_ :logits y_conv)))
(setv train_step (.minimize (tf.train.AdamOptimizer 1e-4) cross_entropy))
(setv correct_prediction (tf.equal (tf.argmax y_conv 1) (tf.argmax y_ 1)))
(setv accuracy (tf.reduce_mean (tf.cast correct_prediction tf.float32)))

(with (sess (tf.Session))
     (sess.run (tf.global_variables_initializer))
     (for [i (range 20000)]
         (setv batch (mnist.train.next_batch 50))
         (when (= (% i 100) 0)
             (setv train_accuracy (accuracy.eval :feed_dict {x (get batch 0) y_ (get batch 1) keep_prob 1.0}))
             (print (.format "step {0}, training accuracy {1:.2f}" i train_accuracy)))
         (train_step.run :feed_dict {x (get batch 0) y_ (get batch 1) keep_prob 0.5}))
     (print (.format "test accuracy {:.3f}"  (accuracy.eval :feed_dict {x mnist.test.images y_ mnist.test.labels keep_prob 1.0}))))

