import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('pre_learning_rate', 0.001, 'Initial pre learning rate')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('pre_epochs', 51, 'Number of epochs to pre-train')
flags.DEFINE_integer('sub_epochs', 0, 'Number of epochs to sub-train')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.(sc)')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.(fc)')
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.(lstm)')
flags.DEFINE_integer('hidden4', 8, 'Number of units in hidden layer 4.(dis)')
flags.DEFINE_integer('hidden5', 32, 'Number of units in hidden layer 5.(dis_fc)')
flags.DEFINE_integer('hidden6', 16, 'Number of units in hidden layer 6.(dis_fc)')
flags.DEFINE_integer('hidden7', 32, 'Number of units in hidden layer 7.(dis_lstm)')
flags.DEFINE_float('lamda', 1, 'weight of lamda')
flags.DEFINE_float('gama', 1, 'weight of gama')
flags.DEFINE_float('beta', 1, 'weight of beta')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('windows_size', 40, 'Length of time series windows')
flags.DEFINE_integer('pred_size', 40, 'Length of time series prediction windows')
flags.DEFINE_integer('window_length', 80, 'Length of windows')
flags.DEFINE_integer('remove_length', 1, 'Length of window remove stride')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('test_sp', 5, 'test step')


flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')