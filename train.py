# coding=utf-8
import os
os.environ["TF_USE_CUSOLVER"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import collections
import time
from absl import app, flags
import dataset_loader
import losses
import model
import util

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'master', 'local', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_integer(
    'ps_tasks', 0, 'Number of tasks in the ps job. If 0 no ps job is used.')
flags.DEFINE_string(
    'checkpoint_dir', '', 'The directory to save checkpoints and summaries.')
flags.DEFINE_string(
    'data_dir', '', 'The training data directory.')
flags.DEFINE_string(
    'model', '9D',
    '9D (rotation), 6D (rotation), T (translation), Single (no derotation)')
flags.DEFINE_integer('batch', 2, 'The size of mini-batches.')
flags.DEFINE_integer('n_epoch', 1, 'Number of training epochs.')
flags.DEFINE_integer(
    'distribution_height', 64, 'The height dimension of output distributions.')
flags.DEFINE_integer(
    'distribution_width', 64, 'The width dimension of output distributions.')
flags.DEFINE_integer(
    'transformed_height', 344,
    'The height dimension of input images after derotation transformation.')
flags.DEFINE_integer(
    'transformed_width', 344,
    'The width dimension of input images after derotation transformation.')
flags.DEFINE_float('lr', 1e-3, 'The learning rate.')
flags.DEFINE_float('alpha', 1e2,
                   'The weight of the distribution loss.')
flags.DEFINE_float('beta', 0.1,
                   'The weight of the spread loss.')
flags.DEFINE_float('kappa', 10.,
                   'A coefficient multiplied by the concentration loss.')
flags.DEFINE_float(
    'transformed_fov', 105.,
    'The field of view of input images after derotation transformation.')
flags.DEFINE_bool('derotate_both', True,
                  'Derotate both input images when training DirectionNet-T')

# -----------------------------
# Computation container
# -----------------------------
Computation = collections.namedtuple(
    'Computation',
    ['train_op', 'loss', 'global_step', 'summaries']
)

# -----------------------------
# Rotation model
# -----------------------------
def direction_net_rotation(src_img, trt_img, rotation_gt, n_output_distributions):
    net = model.DirectionNet(n_output_distributions)
    global_step = tf.train.get_or_create_global_step()

    directions_gt = rotation_gt[:, :n_output_distributions]
    distribution_gt = util.spherical_normalization(
        util.von_mises_fisher(
            directions_gt,
            tf.constant(FLAGS.kappa, tf.float32),
            [FLAGS.distribution_height, FLAGS.distribution_width]
        ),
        rectify=False
    )

    pred = net(src_img, trt_img, training=True)
    directions, expectation, distribution_pred = util.distributions_to_directions(pred)

    rotation_est = (
        util.svd_orthogonalize(directions)
        if n_output_distributions == 3
        else util.gram_schmidt(directions)
    )

    direction_loss = losses.direction_loss(directions, directions_gt)
    distribution_loss = FLAGS.alpha * losses.distribution_loss(distribution_pred, distribution_gt)
    spread_loss = FLAGS.beta * losses.spread_loss(expectation)

    rotation_error = tf.reduce_mean(
        util.rotation_geodesic(rotation_est, rotation_gt)
    )
    direction_error = tf.reduce_mean(tf.acos(tf.clip_by_value(
        tf.reduce_sum(directions * directions_gt, -1), -1., 1.)))

    loss = direction_loss + distribution_loss + spread_loss

    # ---- Summaries ----
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('rotation_error', util.radians_to_degrees(rotation_error))
    tf.summary.scalar('direction_error', util.radians_to_degrees(direction_error))
    tf.summary.scalar('distribution_loss', distribution_loss)
    tf.summary.scalar('spread_loss', spread_loss)

    tf.summary.image('source_image', src_img, max_outputs=4)
    tf.summary.image('target_image', trt_img, max_outputs=4)

    summaries = tf.summary.merge_all()

    optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
    train_op = optimizer.minimize(loss, global_step=global_step)
    train_op = tf.group(train_op, net.updates)

    return Computation(train_op, loss, global_step, summaries)

# -----------------------------
# Timing hook
# -----------------------------
class TimingHook(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.start)

# -----------------------------
# Main
# -----------------------------
def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many arguments.')

    ds = dataset_loader.data_loader(
        data_path=FLAGS.data_dir,
        epochs=FLAGS.n_epoch,
        batch_size=FLAGS.batch,
        training=True
    )

    it = tf.data.make_one_shot_iterator(ds).get_next()
    src_img, trt_img = it.src_image, it.trt_image
    rotation_gt = it.rotation

    computation = direction_net_rotation(src_img, trt_img, rotation_gt, 3)

    timing_hook = TimingHook()

    summary_hook = tf.train.SummarySaverHook(
        save_steps=10,
        output_dir=FLAGS.checkpoint_dir,
        summary_op=computation.summaries
    )

    with tf.train.MonitoredTrainingSession(
        hooks=[
            timing_hook,
            summary_hook,
            tf.train.StepCounterHook(),
            tf.train.NanTensorHook(computation.loss)
        ],
        checkpoint_dir=FLAGS.checkpoint_dir,
        save_checkpoint_steps=2000
    ) as sess:

        while not sess.should_stop():
            _, loss, step = sess.run([
                computation.train_op,
                computation.loss,
                computation.global_step
            ])

            if step % 100 == 0:
                print(f"[step {step}] loss={loss}", flush=True)

if __name__ == '__main__':
    app.run(main)
