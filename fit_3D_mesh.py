'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''


import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fit_3D_mesh(target_3d_mesh_fname, template_fname, tf_model_fname, weights, show_fitting=True):
    '''
    Fit FLAME to 3D mesh in correspondence to the FLAME mesh (i.e. same number of vertices, same mesh topology)
    :param target_3d_mesh_fname:    target 3D mesh filename
    :param template_fname:          template mesh in FLAME topology (only the face information are used)
    :param tf_model_fname:          saved Tensorflow FLAME model

    :param weights:             weights of the individual objective functions
    :return: a mesh with the fitting results
    '''

    target_mesh = Mesh(filename=target_3d_mesh_fname)
    template_mesh = Mesh(filename=template_fname)

    if target_mesh.v.shape[0] != template_mesh.v.shape[0]:
        print('Target mesh does not have the same number of vertices')
        return

    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')

    graph = tf.get_default_graph()
    tf_model = graph.get_tensor_by_name(u'vertices:0')

    with tf.Session() as session:
        saver.restore(session, tf_model_fname)

        # Workaround as existing tf.Variable cannot be retrieved back with tf.get_variable
        # tf_v_template = [x for x in tf.trainable_variables() if 'v_template' in x.name][0]
        tf_trans = [x for x in tf.trainable_variables() if 'trans' in x.name][0]
        tf_rot = [x for x in tf.trainable_variables() if 'rot' in x.name][0]
        tf_pose = [x for x in tf.trainable_variables() if 'pose' in x.name][0]
        tf_shape = [x for x in tf.trainable_variables() if 'shape' in x.name][0]
        tf_exp = [x for x in tf.trainable_variables() if 'exp' in x.name][0]

        mesh_dist = tf.reduce_sum(tf.square(tf.subtract(tf_model, target_mesh.v)))
        neck_pose_reg = tf.reduce_sum(tf.square(tf_pose[:3]))
        jaw_pose_reg = tf.reduce_sum(tf.square(tf_pose[3:6]))
        eyeballs_pose_reg = tf.reduce_sum(tf.square(tf_pose[6:]))
        shape_reg = tf.reduce_sum(tf.square(tf_shape))
        exp_reg = tf.reduce_sum(tf.square(tf_exp))

        # Optimize global transformation first
        vars = [tf_trans, tf_rot]
        loss = mesh_dist
        optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 1})
        print('Optimize rigid transformation')
        optimizer.minimize(session)

        # Optimize for the model parameters
        vars = [tf_trans, tf_rot, tf_pose, tf_shape, tf_exp]
        loss = mesh_dist + weights['shape'] * shape_reg + weights['expr'] * exp_reg + \
               weights['neck_pose'] * neck_pose_reg + weights['jaw_pose'] * jaw_pose_reg + weights['eyeballs_pose'] * eyeballs_pose_reg

        optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 1})
        print('Optimize model parameters')
        optimizer.minimize(session)

        print('Fitting done')

        if show_fitting:
            # Visualize fitting
            mv = MeshViewer()
            fitting_mesh = Mesh(session.run(tf_model), template_mesh.f)
            fitting_mesh.set_vertex_colors('light sky blue')

            mv.set_static_meshes([target_mesh, fitting_mesh])
            raw_input('Press key to continue')

        return Mesh(session.run(tf_model), template_mesh.f)


def fit_sources(
    dir_tup_list,
    tf_model_fname,
    weight_reg_shape,
    weight_reg_expr,
    weight_reg_neck_pos,
    weight_reg_jaw_pos,
    weight_reg_eye_pos,
):
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')

    graph = tf.get_default_graph()
    tf_model = graph.get_tensor_by_name(u'vertices:0')

    with tf.Session() as session:
        saver.restore(session, tf_model_fname)

        # get all params
        tf_trans = [x for x in tf.trainable_variables() if 'trans' in x.name][0]
        tf_rot   = [x for x in tf.trainable_variables() if 'rot'   in x.name][0]
        tf_pose  = [x for x in tf.trainable_variables() if 'pose'  in x.name][0]
        tf_shape = [x for x in tf.trainable_variables() if 'shape' in x.name][0]
        tf_exp   = [x for x in tf.trainable_variables() if 'exp'   in x.name][0]

        # mesh_dist     = tf.reduce_sum(tf.square(tf.subtract(tf_model, target_mesh.v)))
        neck_pose_reg = tf.reduce_sum(tf.square(tf_pose[:3]))
        jaw_pose_reg  = tf.reduce_sum(tf.square(tf_pose[3:6]))
        eye_pose_reg  = tf.reduce_sum(tf.square(tf_pose[6:]))
        shape_reg     = tf.reduce_sum(tf.square(tf_shape))
        exp_reg       = tf.reduce_sum(tf.square(tf_exp))
        reg_term = (
            weight_reg_neck_pos * neck_pose_reg +
            weight_reg_jaw_pos  * jaw_pose_reg  +
            weight_reg_eye_pos  * eye_pose_reg  +
            weight_reg_shape    * shape_reg     +
            weight_reg_expr     * exp_reg
        )

        def _fit_sentence(src_dir, dst_dir, prm_dir):
            if not os.path.exists(src_dir):
                print("Failed to find", src_dir)
                return
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            if not os.path.exists(prm_dir):
                os.makedirs(prm_dir)

            ply_files = []
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if os.path.splitext(f)[1] == ".ply":
                        ply_files.append(os.path.join(root, f))
            ply_files = sorted(ply_files)

            # get shared
            src_mesh = Mesh(filename=ply_files[0])
            mesh_dist = tf.reduce_sum(tf.square(tf.subtract(tf_model, src_mesh.v)))

            optimizer = scipy_pt(loss=mesh_dist, var_list=[tf_trans, tf_rot], method='L-BFGS-B', options={'disp': 1})
            optimizer.minimize(session)

            optimizer = scipy_pt(
                loss=mesh_dist+reg_term,
                var_list=[tf_trans, tf_rot, tf_pose, tf_shape, tf_exp],
                method='L-BFGS-B',
                options={'disp': 1}
            )
            optimizer.minimize(session)

            # save iden
            np.save(os.path.join(prm_dir, "iden.npy"), tf_shape.eval(), allow_pickle=False)

            for src_fname in tqdm(ply_files):
                dst_fname = os.path.join(dst_dir, os.path.basename(f))
                # param filename
                prm_fname = os.path.join(prm_dir, os.path.basename(f))
                prm_fname = os.path.splitext(prm_fname)[0] + '_exp.npy'

                src_mesh = Mesh(filename=src_fname)
                mesh_dist = tf.reduce_sum(tf.square(tf.subtract(tf_model, src_mesh.v)))

                optimizer = scipy_pt(
                    loss=mesh_dist+reg_term,
                    var_list=[tf_shape, tf_exp],
                    method='L-BFGS-B', options={'disp': 0, 'maxiter': 20}
                )
                optimizer.minimize(session)

                # save expr
                np.save(prm_fname, tf_exp.eval())

                # save mesh
                fitting_mesh = Mesh(session.run(tf_model), src_mesh.f)
                fitting_mesh.write_ply(dst_fname)


def run_corresponding_mesh_fitting():
    # Path of the Tensorflow FLAME model
    tf_model_fname = './models/generic_model'
    # tf_model_fname = './models/female_model'
    # tf_model_fname = './models/male_model'

    # Path of a tempalte mesh in FLAME topology
    template_fname = './data/template.ply'

    # target 3D mesh in dense vertex-correspondence to the model
    target_mesh_path = './data/registered_mesh.ply'

    # Output filename
    out_mesh_fname = './results/mesh_fitting.ply'

    weights = {}
    # Weight of the shape regularizer
    weights['shape'] = 1e-4
    # Weight of the expression regularizer
    weights['expr']  = 1e-4
    # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer
    weights['neck_pose'] = 1e-4
    # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer
    weights['jaw_pose'] = 1e-4
    # Weight of the eyeball pose (i.e. eyeball rotations) regularizer
    weights['eyeballs_pose'] = 1e-4
    # Show landmark fitting (default: red = target landmarks, blue = fitting landmarks)
    show_fitting = True

    result_mesh = fit_3D_mesh(target_mesh_path, template_fname, tf_model_fname, weights, show_fitting=show_fitting)

    if not os.path.exists(os.path.dirname(out_mesh_fname)):
        os.makedirs(os.path.dirname(out_mesh_fname))

    result_mesh.write_ply(out_mesh_fname)


if __name__ == '__main__':
    # run_corresponding_mesh_fitting()
    fit_sources(
        [("./vocaset/FaceTalk_170915_00223_TA/sentence01/",
          "./vocaset/FaceTalk_170915_00223_TA_clean/sentence01/",
          "./vocaset/FaceTalk_170915_00223_TA_param/sentence01/")],
        tf_model_fname='./models/generic_model',
        weight_reg_shape=1e-4,
        weight_reg_expr=1e-7,
        weight_reg_neck_pos=1e-4,
        weight_reg_jaw_pos=1e-4,
        weight_reg_eye_pos=1e-4,
    )
