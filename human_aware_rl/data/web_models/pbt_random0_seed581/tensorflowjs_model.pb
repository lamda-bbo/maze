
A
agent0/ppo2_model/pi/bConst*
value
B*
dtype0
E
agent0/ppo2_model/pi/wConst*
valueB@*
dtype0
L
!agent0/ppo2_model/pi/dense_2/biasConst*
value
B@*
dtype0
R
#agent0/ppo2_model/pi/dense_2/kernelConst*
valueB@@*
dtype0
L
!agent0/ppo2_model/pi/dense_1/biasConst*
value
B@*
dtype0
R
#agent0/ppo2_model/pi/dense_1/kernelConst*
valueB@@*
dtype0
J
agent0/ppo2_model/pi/dense/biasConst*
value
B@*
dtype0
Q
!agent0/ppo2_model/pi/dense/kernelConst*
valueB	�@*
dtype0
U
*agent0/ppo2_model/pi/flatten/Reshape/shapeConst*
value
B*
dtype0
K
 agent0/ppo2_model/pi/conv_1/biasConst*
dtype0*
value
B
Y
"agent0/ppo2_model/pi/conv_1/kernelConst*
dtype0*
valueB
K
 agent0/ppo2_model/pi/conv_0/biasConst*
value
B*
dtype0
Y
"agent0/ppo2_model/pi/conv_0/kernelConst*
valueB*
dtype0
Q
&agent0/ppo2_model/pi/conv_initial/biasConst*
dtype0*
value
B
_
(agent0/ppo2_model/pi/conv_initial/kernelConst*
valueB*
dtype0
M
agent0/ppo2_model/ObPlaceholder*
dtype0*
shape:
�
(agent0/ppo2_model/pi/conv_initial/Conv2DConv2Dagent0/ppo2_model/Ob(agent0/ppo2_model/pi/conv_initial/kernel*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
)agent0/ppo2_model/pi/conv_initial/BiasAddBiasAdd(agent0/ppo2_model/pi/conv_initial/Conv2D&agent0/ppo2_model/pi/conv_initial/bias*
T0*
data_formatNHWC
|
+agent0/ppo2_model/pi/conv_initial/LeakyRelu	LeakyRelu)agent0/ppo2_model/pi/conv_initial/BiasAdd*
T0*
alpha%��L>
�
"agent0/ppo2_model/pi/conv_0/Conv2DConv2D+agent0/ppo2_model/pi/conv_initial/LeakyRelu"agent0/ppo2_model/pi/conv_0/kernel*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides

�
#agent0/ppo2_model/pi/conv_0/BiasAddBiasAdd"agent0/ppo2_model/pi/conv_0/Conv2D agent0/ppo2_model/pi/conv_0/bias*
data_formatNHWC*
T0
p
%agent0/ppo2_model/pi/conv_0/LeakyRelu	LeakyRelu#agent0/ppo2_model/pi/conv_0/BiasAdd*
alpha%��L>*
T0
�
"agent0/ppo2_model/pi/conv_1/Conv2DConv2D%agent0/ppo2_model/pi/conv_0/LeakyRelu"agent0/ppo2_model/pi/conv_1/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
#agent0/ppo2_model/pi/conv_1/BiasAddBiasAdd"agent0/ppo2_model/pi/conv_1/Conv2D agent0/ppo2_model/pi/conv_1/bias*
data_formatNHWC*
T0
p
%agent0/ppo2_model/pi/conv_1/LeakyRelu	LeakyRelu#agent0/ppo2_model/pi/conv_1/BiasAdd*
alpha%��L>*
T0
�
$agent0/ppo2_model/pi/flatten/ReshapeReshape%agent0/ppo2_model/pi/conv_1/LeakyRelu*agent0/ppo2_model/pi/flatten/Reshape/shape*
T0*
Tshape0
�
!agent0/ppo2_model/pi/dense/MatMulMatMul$agent0/ppo2_model/pi/flatten/Reshape!agent0/ppo2_model/pi/dense/kernel*
T0*
transpose_a( *
transpose_b( 
�
"agent0/ppo2_model/pi/dense/BiasAddBiasAdd!agent0/ppo2_model/pi/dense/MatMulagent0/ppo2_model/pi/dense/bias*
T0*
data_formatNHWC
n
$agent0/ppo2_model/pi/dense/LeakyRelu	LeakyRelu"agent0/ppo2_model/pi/dense/BiasAdd*
T0*
alpha%��L>
�
#agent0/ppo2_model/pi/dense_1/MatMulMatMul$agent0/ppo2_model/pi/dense/LeakyRelu#agent0/ppo2_model/pi/dense_1/kernel*
T0*
transpose_a( *
transpose_b( 
�
$agent0/ppo2_model/pi/dense_1/BiasAddBiasAdd#agent0/ppo2_model/pi/dense_1/MatMul!agent0/ppo2_model/pi/dense_1/bias*
T0*
data_formatNHWC
r
&agent0/ppo2_model/pi/dense_1/LeakyRelu	LeakyRelu$agent0/ppo2_model/pi/dense_1/BiasAdd*
T0*
alpha%��L>
�
#agent0/ppo2_model/pi/dense_2/MatMulMatMul&agent0/ppo2_model/pi/dense_1/LeakyRelu#agent0/ppo2_model/pi/dense_2/kernel*
transpose_a( *
transpose_b( *
T0
�
$agent0/ppo2_model/pi/dense_2/BiasAddBiasAdd#agent0/ppo2_model/pi/dense_2/MatMul!agent0/ppo2_model/pi/dense_2/bias*
data_formatNHWC*
T0
r
&agent0/ppo2_model/pi/dense_2/LeakyRelu	LeakyRelu$agent0/ppo2_model/pi/dense_2/BiasAdd*
T0*
alpha%��L>
�
agent0/ppo2_model/pi_1/MatMulMatMul&agent0/ppo2_model/pi/dense_2/LeakyReluagent0/ppo2_model/pi/w*
transpose_a( *
transpose_b( *
T0
a
agent0/ppo2_model/pi_1/addAddagent0/ppo2_model/pi_1/MatMulagent0/ppo2_model/pi/b*
T0
I
agent0/ppo2_model/SoftmaxSoftmaxagent0/ppo2_model/pi_1/add*
T0
N
agent0/ppo2_model/action_probsIdentityagent0/ppo2_model/Softmax*
T0 " 