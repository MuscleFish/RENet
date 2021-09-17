from tensorflow.keras import layers, metrics, models, optimizers
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

stdTechniqueIds = ['T1001', 'T1003', 'T1005', 'T1007', 'T1008', 'T1010', 'T1011', 'T1012', 'T1014', 'T1016', 'T1018', 'T1020', 'T1021', 'T1025', 'T1027', 'T1029', 'T1030', 'T1033', 'T1036', 'T1037', 'T1039', 'T1040', 'T1041', 'T1046', 'T1047', 'T1048', 'T1049', 'T1052', 'T1053', 'T1055', 'T1056', 'T1057', 'T1059', 'T1068', 'T1069', 'T1070', 'T1071', 'T1072', 'T1074', 'T1078', 'T1080', 'T1082', 'T1083', 'T1087', 'T1090', 'T1091', 'T1092', 'T1095', 'T1098', 'T1102', 'T1104', 'T1105', 'T1106', 'T1110', 'T1111', 'T1112', 'T1113', 'T1114', 'T1115', 'T1119', 'T1120', 'T1123', 'T1124', 'T1125', 'T1127', 'T1129', 'T1132', 'T1133', 'T1134', 'T1135', 'T1136', 'T1137', 'T1140', 'T1176', 'T1185', 'T1187', 'T1189', 'T1190', 'T1195', 'T1197', 'T1199', 'T1200', 'T1201', 'T1202', 'T1203', 'T1204', 'T1205', 'T1207', 'T1210', 'T1211', 'T1213', 'T1216', 'T1217', 'T1218', 'T1219', 'T1220', 'T1221', 'T1222', 'T1480', 'T1482', 'T1484', 'T1485', 'T1486', 'T1489', 'T1490', 'T1491', 'T1496', 'T1497', 'T1498', 'T1499', 'T1505', 'T1518', 'T1528', 'T1529', 'T1531', 'T1534', 'T1539', 'T1542', 'T1543', 'T1546', 'T1547', 'T1548', 'T1550', 'T1552', 'T1553', 'T1554', 'T1555', 'T1556', 'T1557', 'T1558', 'T1559', 'T1560', 'T1561', 'T1562', 'T1563', 'T1564', 'T1565', 'T1566', 'T1567', 'T1568', 'T1569', 'T1570', 'T1571', 'T1572', 'T1573', 'T1574', 'T1583', 'T1584', 'T1585', 'T1587', 'T1588', 'T1601']
stdTacticIds = ['TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009', 'TA0010', 'TA0011', 'TA0040', 'TA0042', 'TA0043']

def get_rel_weights(from_ids=None, to_ids=None, relation_file='datas/attck_tactic_tech_relation.json', default_fill=0):
    tact_ids = from_ids if from_ids else stdTacticIds
    tech_ids = to_ids if to_ids else stdTechniqueIds
    theJson = json.load(open(relation_file, encoding='utf-8-sig'))
    relWeights = np.zeros((len(tact_ids), len(tech_ids)), dtype='float32')
    if default_fill != 0:
        relWeights.fill(-0.1)
    for tac in theJson:
        for tec in tac['techs']:
            try:
                tacindex = tact_ids.index(tac['id'])
                tecindex = tech_ids.index(tec['id'])
                if tacindex in range(len(tact_ids)) and tecindex in range(len(tech_ids)):
                    relWeights[tacindex, tecindex]=0.1
            except Exception as e:
                print(e)
                continue
    return relWeights

def relevance_enhancement(tact_output_layer, tech_output_layer, type='n', weights=None):
    if weights.shape[0] == len(stdTacticIds) and weights.shape[1] == len(stdTechniqueIds) and not weights:
        weights = get_rel_weights()
    if type in ['0', 'zero']:
        relevanceTransLayer = layers.Dense(tech_output_layer.shape[1], use_bias=False, name='relevance-transform-layer',
                                           trainable=True,
                                           kernel_regularizer=tf.keras.regularizers.l2())(tact_output_layer)
    elif type in ['a', 'artificial']:
        if weights.shape[0] == tact_output_layer.shape[-1] and weights.shape[1] == tech_output_layer.shape[-1]:
            relevanceTransLayer = layers.Dense(tech_output_layer.shape[1], use_bias=False,
                                               name='relevance-transform-layer',
                                               kernel_initializer=tf.keras.initializers.Constant(weights),
                                               trainable=True,
                                               kernel_regularizer=tf.keras.regularizers.l2())(tact_output_layer)
        else:
            relevanceTransLayer = layers.Dense(tech_output_layer.shape[1], use_bias=False,
                                               name='relevance-transform-layer',
                                               trainable=True,
                                               kernel_regularizer=tf.keras.regularizers.l2())(tact_output_layer)
    elif type in ['la', 'lock-artificial']:
        if weights.shape[0] == tact_output_layer.shape[-1] and weights.shape[1] == tech_output_layer.shape[-1]:
            relevanceTransLayer = layers.Dense(tech_output_layer.shape[1], use_bias=False,
                                               name='relevance-transform-layer',
                                               kernel_initializer=tf.keras.initializers.Constant(weights),
                                               trainable=False,
                                               kernel_regularizer=tf.keras.regularizers.l2())(tact_output_layer)
        else:
            relevanceTransLayer = layers.Dense(tech_output_layer.shape[1], use_bias=False,
                                               name='relevance-transform-layer',
                                               trainable=True,
                                               kernel_regularizer=tf.keras.regularizers.l2())(tact_output_layer)
    else:
        return tech_output_layer
    minGateLayer = layers.Minimum(name='min')([relevanceTransLayer, tech_output_layer])
    return minGateLayer

def single_classifier(input_layer, output_shape, activation='sigmoid', withRCNN=False,
                      dense_units=[512,256,128], lstm_dim=256, dropout=0.5, cnn_dim=[1,2],
                      cnn_count=256, output_name=None):
    if lstm_dim > 0:
        birnn_layer = layers.Bidirectional(layers.GRU(lstm_dim, return_sequences=True))(input_layer)
        if withRCNN:
            #             reshape_layer = layers.Dense(lstm_dim, activation='relu')(input_layer)
            reshape_layer = input_layer
            birnn_layer = layers.Concatenate(axis=2)([birnn_layer, reshape_layer])
        x = layers.Dropout(dropout)(birnn_layer)
    else:
        x = input_layer
    cnn_layers = []
    for cd in cnn_dim:
        cx = layers.Conv1D(cnn_count, kernel_size=cd, padding='same')(x)
        cx = layers.LayerNormalization()(cx)
        cx = layers.ReLU()(cx)

        # 新 归一化
        cx  = layers.LayerNormalization()(cx)

        cx = layers.GlobalMaxPooling1D()(cx)
        cx = layers.Reshape((1,cx.shape[1]))(cx)
        cnn_layers.append(cx)
    if len(cnn_layers) > 1:
        x = layers.Concatenate(axis=1)(cnn_layers)
        #         x = layers.Conv1D(cnn_count,len(cnn_layers))(x)
        #         x = layers.GlobalMaxPool1D()(x)
        x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(cnn_layers[0])
    for units in dense_units:
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(output_shape, activation=activation)(x) if not output_name else \
        layers.Dense(output_shape, activation=activation, name=output_name)(x)
    return x

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def RENet(input_shape, output_shapes, activation='sigmoid', withRCNN=False,
          dense_units=[512, 256, 128], lstm_dim=256, dropout=0.5, cnn_dim=[1, 2],
          cnn_count=256, output_names=None, renet_pairs=None):
    input_layer = layers.Input(input_shape)
    output_layers = []
    for i, otsh in enumerate(output_shapes):
        output_name = output_names[i] if output_names and output_names[i] else None
        output_layers.append(single_classifier(input_layer=input_layer, output_shape=otsh,
                                               activation=activation, withRCNN=withRCNN,
                                               dense_units=dense_units, lstm_dim=lstm_dim,
                                               dropout=dropout, cnn_dim=cnn_dim, cnn_count=cnn_count,
                                               output_name=output_name))

    if len(output_layers) > 1:
        new_outs = output_layers
        if not renet_pairs or type(renet_pairs) != list or len(renet_pairs) == 0:
            for i in range(len(output_layers)-1):
                new_outs[i+1] = relevance_enhancement(output_layers[i], output_layers[i+1], type='0')
        else:
            for pair in renet_pairs:
                if type(pair) != dict:
                    continue
                from_index = pair['from'] if 'from' in pair.keys() else None
                to_index = pair['to'] if 'to' in pair.keys() else None
                net_type = pair['type'] if 'type' in pair.keys() else None
                weights = pair['weights'] if 'weights' in pair.keys() else None
                if from_index and to_index:
                    new_outs[to_index] = relevance_enhancement(output_layers[from_index],
                                                               output_layers[to_index],
                                                               type=net_type, weights=weights)
        output_layers = new_outs
    elif output_layers == 0:
        return None

    model = models.Model(input_layer, output_layers)
    model.compile(optimizer=optimizers.Adam(), loss=focal_loss(),
                  metrics=[metrics.Precision(), metrics.Recall()])
    model.summary()
    return model