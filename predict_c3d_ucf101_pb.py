
import input_data
import numpy as np
import tensorflow as tf
import time

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('max_steps', 10, 'Max steps.')
FLAGS = flags.FLAGS

def model_test(model_file,
               input_tensor_name,
               output_tensor_name,
               test_list_file):
  num_test_videos = len(list(open(test_list_file,'r')))
  print("Info: Number of test videos is {}".format(num_test_videos))
  
  with tf.Graph().as_default():
    graph_def = tf.GraphDef()
    model_f = open(model_file, "rb")
    graph_def.ParseFromString(model_f.read())
    _ = tf.import_graph_def(graph_def, name='')
    
    # Limit GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.250)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      init = tf.global_variables_initializer()
      sess.run(init)

      tensor_input = sess.graph.get_tensor_by_name(input_tensor_name);
      tensor_output = sess.graph.get_tensor_by_name(output_tensor_name);

      images_placeholder = tensor_input
      
      # Session run
      logits = []
      #logit = sess.run(tensor_output, feed_dict={tensor_input: test_images})
      logit = tensor_output
      logits.append(logit)
      logits = tf.concat(logits, 0)
      norm_score = tf.nn.softmax(logits)

      if FLAGS.max_steps is None:
        max_steps = int((num_test_videos - 1) / (FLAGS.batch_size) + 1)
      else:
        max_steps = FLAGS.max_steps
      print("Info: Max steps is %d" % max_steps)
      
      true_count = 0
      all_count = 0
      next_start_pos = 0
      all_steps = max_steps
      for step in range(all_steps):
        start_time = time.time()

        test_images, test_labels, next_start_pos, _, valid_len = \
                input_data.read_clip_and_label(
                        test_list_file,
                        FLAGS.batch_size,
                        start_pos=next_start_pos
                        )
        predict_score = norm_score.eval(
                session=sess,
                feed_dict={images_placeholder: test_images}
                )
        for i in range(0, valid_len):
          true_label = test_labels[i],
          top1_predicted_label = np.argmax(predict_score[i], axis=0)
          if (true_label == top1_predicted_label):
            true_count = true_count + 1
          all_count = all_count + 1
        
        duration = time.time() - start_time
        print('Info: Step %d: %.3f sec' % (step, duration))
      
      acc = float(true_count) / all_count
      print("Info: Accuracy: " + "{:.5f}".format(acc))

if __name__ == "__main__":
  # Set pb model path
  model_file         = "saved_model/c3d_10000.pb"
  input_tensor_name  = "input:0"
  output_tensor_name = "out:0"
  test_list_file     = "list/test.list"
  model_test(model_file,
             input_tensor_name,
             output_tensor_name,
             test_list_file)
