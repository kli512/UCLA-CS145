�
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8��
|
dense_117/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_117/kernel
u
$dense_117/kernel/Read/ReadVariableOpReadVariableOpdense_117/kernel*
_output_shapes

:
*
dtype0
t
dense_117/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_117/bias
m
"dense_117/bias/Read/ReadVariableOpReadVariableOpdense_117/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
z
lstm_117/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
P* 
shared_namelstm_117/kernel
s
#lstm_117/kernel/Read/ReadVariableOpReadVariableOplstm_117/kernel*
_output_shapes

:
P*
dtype0
�
lstm_117/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P**
shared_namelstm_117/recurrent_kernel
�
-lstm_117/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_117/recurrent_kernel*
_output_shapes

:P*
dtype0
r
lstm_117/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_namelstm_117/bias
k
!lstm_117/bias/Read/ReadVariableOpReadVariableOplstm_117/bias*
_output_shapes
:P*
dtype0
�
Adam/dense_117/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_117/kernel/m
�
+Adam/dense_117/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_117/kernel/m*
_output_shapes

:
*
dtype0
�
Adam/dense_117/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_117/bias/m
{
)Adam/dense_117/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_117/bias/m*
_output_shapes
:
*
dtype0
�
Adam/lstm_117/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
P*'
shared_nameAdam/lstm_117/kernel/m
�
*Adam/lstm_117/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_117/kernel/m*
_output_shapes

:
P*
dtype0
�
 Adam/lstm_117/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*1
shared_name" Adam/lstm_117/recurrent_kernel/m
�
4Adam/lstm_117/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_117/recurrent_kernel/m*
_output_shapes

:P*
dtype0
�
Adam/lstm_117/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/lstm_117/bias/m
y
(Adam/lstm_117/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_117/bias/m*
_output_shapes
:P*
dtype0
�
Adam/dense_117/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_117/kernel/v
�
+Adam/dense_117/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_117/kernel/v*
_output_shapes

:
*
dtype0
�
Adam/dense_117/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_117/bias/v
{
)Adam/dense_117/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_117/bias/v*
_output_shapes
:
*
dtype0
�
Adam/lstm_117/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
P*'
shared_nameAdam/lstm_117/kernel/v
�
*Adam/lstm_117/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_117/kernel/v*
_output_shapes

:
P*
dtype0
�
 Adam/lstm_117/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*1
shared_name" Adam/lstm_117/recurrent_kernel/v
�
4Adam/lstm_117/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_117/recurrent_kernel/v*
_output_shapes

:P*
dtype0
�
Adam/lstm_117/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*%
shared_nameAdam/lstm_117/bias/v
y
(Adam/lstm_117/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_117/bias/v*
_output_shapes
:P*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
 
l

cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�
iter

beta_1

beta_2
	decay
learning_ratem2m3m4m5m6v7v8v9v:v;
#
0
1
2
3
4
#
0
1
2
3
4
 
�
	variables

layers
layer_regularization_losses
trainable_variables
 non_trainable_variables
regularization_losses
!metrics
 
~

kernel
recurrent_kernel
bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
 

0
1
2

0
1
2
 
�
	variables
&layer_regularization_losses

'layers
(non_trainable_variables
trainable_variables
regularization_losses
)metrics
\Z
VARIABLE_VALUEdense_117/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_117/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
*layer_regularization_losses

+layers
,non_trainable_variables
trainable_variables
regularization_losses
-metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUElstm_117/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_117/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUElstm_117/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 

0
1
2

0
1
2
 
�
"	variables
.layer_regularization_losses

/layers
0non_trainable_variables
#trainable_variables
$regularization_losses
1metrics
 


0
 
 
 
 
 
 
 
 
 
 
}
VARIABLE_VALUEAdam/dense_117/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_117/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/lstm_117/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_117/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/lstm_117/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_117/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_117/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/lstm_117/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_117/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/lstm_117/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lstm_117_inputPlaceholder*+
_output_shapes
:���������
*
dtype0* 
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_117_inputlstm_117/kernellstm_117/recurrent_kernellstm_117/biasdense_117/kerneldense_117/bias*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference_signature_wrapper_1667964
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_117/kernel/Read/ReadVariableOp"dense_117/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#lstm_117/kernel/Read/ReadVariableOp-lstm_117/recurrent_kernel/Read/ReadVariableOp!lstm_117/bias/Read/ReadVariableOp+Adam/dense_117/kernel/m/Read/ReadVariableOp)Adam/dense_117/bias/m/Read/ReadVariableOp*Adam/lstm_117/kernel/m/Read/ReadVariableOp4Adam/lstm_117/recurrent_kernel/m/Read/ReadVariableOp(Adam/lstm_117/bias/m/Read/ReadVariableOp+Adam/dense_117/kernel/v/Read/ReadVariableOp)Adam/dense_117/bias/v/Read/ReadVariableOp*Adam/lstm_117/kernel/v/Read/ReadVariableOp4Adam/lstm_117/recurrent_kernel/v/Read/ReadVariableOp(Adam/lstm_117/bias/v/Read/ReadVariableOpConst*!
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_save_1669141
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_117/kerneldense_117/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_117/kernellstm_117/recurrent_kernellstm_117/biasAdam/dense_117/kernel/mAdam/dense_117/bias/mAdam/lstm_117/kernel/m Adam/lstm_117/recurrent_kernel/mAdam/lstm_117/bias/mAdam/dense_117/kernel/vAdam/dense_117/bias/vAdam/lstm_117/kernel/v Adam/lstm_117/recurrent_kernel/vAdam/lstm_117/bias/v* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__traced_restore_1669213¨
�R
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1667694

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1d
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1667609*
condR
while_cond_1667608*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�
�
*__inference_lstm_117_layer_call_fn_1668938

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16676942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
+__inference_dense_117_layer_call_fn_1668963

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_16678782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
while_cond_1667342
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1667342___redundant_placeholder0/
+while_cond_1667342___redundant_placeholder1/
+while_cond_1667342___redundant_placeholder2/
+while_cond_1667342___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�l
�
K__inference_sequential_117_layer_call_and_return_conditional_losses_1668282

inputs+
'lstm_117_matmul_readvariableop_resource-
)lstm_117_matmul_1_readvariableop_resource,
(lstm_117_biasadd_readvariableop_resource,
(dense_117_matmul_readvariableop_resource-
)dense_117_biasadd_readvariableop_resource
identity�� dense_117/BiasAdd/ReadVariableOp�dense_117/MatMul/ReadVariableOp�lstm_117/BiasAdd/ReadVariableOp�lstm_117/MatMul/ReadVariableOp� lstm_117/MatMul_1/ReadVariableOp�lstm_117/whileV
lstm_117/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_117/Shape�
lstm_117/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_117/strided_slice/stack�
lstm_117/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_117/strided_slice/stack_1�
lstm_117/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_117/strided_slice/stack_2�
lstm_117/strided_sliceStridedSlicelstm_117/Shape:output:0%lstm_117/strided_slice/stack:output:0'lstm_117/strided_slice/stack_1:output:0'lstm_117/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_117/strided_slicen
lstm_117/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros/mul/y�
lstm_117/zeros/mulMullstm_117/strided_slice:output:0lstm_117/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros/mulq
lstm_117/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_117/zeros/Less/y�
lstm_117/zeros/LessLesslstm_117/zeros/mul:z:0lstm_117/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros/Lesst
lstm_117/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros/packed/1�
lstm_117/zeros/packedPacklstm_117/strided_slice:output:0 lstm_117/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_117/zeros/packedq
lstm_117/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_117/zeros/Const�
lstm_117/zerosFilllstm_117/zeros/packed:output:0lstm_117/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
lstm_117/zerosr
lstm_117/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros_1/mul/y�
lstm_117/zeros_1/mulMullstm_117/strided_slice:output:0lstm_117/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros_1/mulu
lstm_117/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_117/zeros_1/Less/y�
lstm_117/zeros_1/LessLesslstm_117/zeros_1/mul:z:0 lstm_117/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros_1/Lessx
lstm_117/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros_1/packed/1�
lstm_117/zeros_1/packedPacklstm_117/strided_slice:output:0"lstm_117/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_117/zeros_1/packedu
lstm_117/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_117/zeros_1/Const�
lstm_117/zeros_1Fill lstm_117/zeros_1/packed:output:0lstm_117/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2
lstm_117/zeros_1�
lstm_117/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_117/transpose/perm�
lstm_117/transpose	Transposeinputs lstm_117/transpose/perm:output:0*
T0*+
_output_shapes
:���������
2
lstm_117/transposej
lstm_117/Shape_1Shapelstm_117/transpose:y:0*
T0*
_output_shapes
:2
lstm_117/Shape_1�
lstm_117/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_117/strided_slice_1/stack�
 lstm_117/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_1/stack_1�
 lstm_117/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_1/stack_2�
lstm_117/strided_slice_1StridedSlicelstm_117/Shape_1:output:0'lstm_117/strided_slice_1/stack:output:0)lstm_117/strided_slice_1/stack_1:output:0)lstm_117/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_117/strided_slice_1�
$lstm_117/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$lstm_117/TensorArrayV2/element_shape�
lstm_117/TensorArrayV2TensorListReserve-lstm_117/TensorArrayV2/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_117/TensorArrayV2�
>lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   2@
>lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape�
0lstm_117/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_117/transpose:y:0Glstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_117/TensorArrayUnstack/TensorListFromTensor�
lstm_117/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_117/strided_slice_2/stack�
 lstm_117/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_2/stack_1�
 lstm_117/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_2/stack_2�
lstm_117/strided_slice_2StridedSlicelstm_117/transpose:y:0'lstm_117/strided_slice_2/stack:output:0)lstm_117/strided_slice_2/stack_1:output:0)lstm_117/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
lstm_117/strided_slice_2�
lstm_117/MatMul/ReadVariableOpReadVariableOp'lstm_117_matmul_readvariableop_resource*
_output_shapes

:
P*
dtype02 
lstm_117/MatMul/ReadVariableOp�
lstm_117/MatMulMatMul!lstm_117/strided_slice_2:output:0&lstm_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
lstm_117/MatMul�
 lstm_117/MatMul_1/ReadVariableOpReadVariableOp)lstm_117_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_117/MatMul_1/ReadVariableOp�
lstm_117/MatMul_1MatMullstm_117/zeros:output:0(lstm_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
lstm_117/MatMul_1�
lstm_117/addAddV2lstm_117/MatMul:product:0lstm_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
lstm_117/add�
lstm_117/BiasAdd/ReadVariableOpReadVariableOp(lstm_117_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
lstm_117/BiasAdd/ReadVariableOp�
lstm_117/BiasAddBiasAddlstm_117/add:z:0'lstm_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
lstm_117/BiasAddb
lstm_117/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/Constv
lstm_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/split/split_dim�
lstm_117/splitSplit!lstm_117/split/split_dim:output:0lstm_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
lstm_117/splitz
lstm_117/SigmoidSigmoidlstm_117/split:output:0*
T0*'
_output_shapes
:���������2
lstm_117/Sigmoid~
lstm_117/Sigmoid_1Sigmoidlstm_117/split:output:1*
T0*'
_output_shapes
:���������2
lstm_117/Sigmoid_1�
lstm_117/mulMullstm_117/Sigmoid_1:y:0lstm_117/zeros_1:output:0*
T0*'
_output_shapes
:���������2
lstm_117/mulq
lstm_117/ReluRelulstm_117/split:output:2*
T0*'
_output_shapes
:���������2
lstm_117/Relu�
lstm_117/mul_1Mullstm_117/Sigmoid:y:0lstm_117/Relu:activations:0*
T0*'
_output_shapes
:���������2
lstm_117/mul_1�
lstm_117/add_1AddV2lstm_117/mul:z:0lstm_117/mul_1:z:0*
T0*'
_output_shapes
:���������2
lstm_117/add_1~
lstm_117/Sigmoid_2Sigmoidlstm_117/split:output:3*
T0*'
_output_shapes
:���������2
lstm_117/Sigmoid_2p
lstm_117/Relu_1Relulstm_117/add_1:z:0*
T0*'
_output_shapes
:���������2
lstm_117/Relu_1�
lstm_117/mul_2Mullstm_117/Sigmoid_2:y:0lstm_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������2
lstm_117/mul_2�
&lstm_117/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2(
&lstm_117/TensorArrayV2_1/element_shape�
lstm_117/TensorArrayV2_1TensorListReserve/lstm_117/TensorArrayV2_1/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_117/TensorArrayV2_1`
lstm_117/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_117/time�
!lstm_117/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!lstm_117/while/maximum_iterations|
lstm_117/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_117/while/loop_counter�
lstm_117/whileWhile$lstm_117/while/loop_counter:output:0*lstm_117/while/maximum_iterations:output:0lstm_117/time:output:0!lstm_117/TensorArrayV2_1:handle:0lstm_117/zeros:output:0lstm_117/zeros_1:output:0!lstm_117/strided_slice_1:output:0@lstm_117/TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_117_matmul_readvariableop_resource)lstm_117_matmul_1_readvariableop_resource(lstm_117_biasadd_readvariableop_resource ^lstm_117/BiasAdd/ReadVariableOp^lstm_117/MatMul/ReadVariableOp!^lstm_117/MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *'
bodyR
lstm_117_while_body_1668191*'
condR
lstm_117_while_cond_1668190*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
lstm_117/while�
9lstm_117/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2;
9lstm_117/TensorArrayV2Stack/TensorListStack/element_shape�
+lstm_117/TensorArrayV2Stack/TensorListStackTensorListStacklstm_117/while:output:3Blstm_117/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype02-
+lstm_117/TensorArrayV2Stack/TensorListStack�
lstm_117/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2 
lstm_117/strided_slice_3/stack�
 lstm_117/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_117/strided_slice_3/stack_1�
 lstm_117/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_3/stack_2�
lstm_117/strided_slice_3StridedSlice4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_117/strided_slice_3/stack:output:0)lstm_117/strided_slice_3/stack_1:output:0)lstm_117/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_117/strided_slice_3�
lstm_117/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_117/transpose_1/perm�
lstm_117/transpose_1	Transpose4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_117/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_117/transpose_1x
lstm_117/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_117/runtime�
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_117/MatMul/ReadVariableOp�
dense_117/MatMulMatMul!lstm_117/strided_slice_3:output:0'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_117/MatMul�
 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_117/BiasAdd/ReadVariableOp�
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_117/BiasAdd�
IdentityIdentitydense_117/BiasAdd:output:0!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp ^lstm_117/BiasAdd/ReadVariableOp^lstm_117/MatMul/ReadVariableOp!^lstm_117/MatMul_1/ReadVariableOp^lstm_117/while*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2B
lstm_117/BiasAdd/ReadVariableOplstm_117/BiasAdd/ReadVariableOp2@
lstm_117/MatMul/ReadVariableOplstm_117/MatMul/ReadVariableOp2D
 lstm_117/MatMul_1/ReadVariableOp lstm_117/MatMul_1/ReadVariableOp2 
lstm_117/whilelstm_117/while:& "
 
_user_specified_nameinputs
�
�
lstm_117_while_cond_1668190
lstm_117_while_loop_counter%
!lstm_117_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3!
less_lstm_117_strided_slice_18
4lstm_117_while_cond_1668190___redundant_placeholder08
4lstm_117_while_cond_1668190___redundant_placeholder18
4lstm_117_while_cond_1668190___redundant_placeholder28
4lstm_117_while_cond_1668190___redundant_placeholder3
identity
a
LessLessplaceholderless_lstm_117_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�
�
while_body_1667466
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:���������:���������:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_16670702
StatefulPartitionedCall�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::22
StatefulPartitionedCallStatefulPartitionedCall
�D
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1667409

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:���������:���������:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_16670372
StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1667343*
condR
while_cond_1667342*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������
:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
�
�
*__inference_lstm_117_layer_call_fn_1668616
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16674092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������
:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�S
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668455
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1d
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1668370*
condR
while_cond_1668369*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
�.
�
while_body_1667762
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�R
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1667847

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1d
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1667762*
condR
while_cond_1667761*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�.
�
while_body_1667609
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�D
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1667532

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:���������:���������:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_16670702
StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1667466*
condR
while_cond_1667465*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������
:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
�
�
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1668996

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������
:���������:���������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
�
�
while_cond_1668369
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1668369___redundant_placeholder0/
+while_cond_1668369___redundant_placeholder1/
+while_cond_1668369___redundant_placeholder2/
+while_cond_1668369___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�
�
*__inference_lstm_117_layer_call_fn_1668624
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16675322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������
:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�
�
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667891
lstm_117_input+
'lstm_117_statefulpartitionedcall_args_1+
'lstm_117_statefulpartitionedcall_args_2+
'lstm_117_statefulpartitionedcall_args_3,
(dense_117_statefulpartitionedcall_args_1,
(dense_117_statefulpartitionedcall_args_2
identity��!dense_117/StatefulPartitionedCall� lstm_117/StatefulPartitionedCall�
 lstm_117/StatefulPartitionedCallStatefulPartitionedCalllstm_117_input'lstm_117_statefulpartitionedcall_args_1'lstm_117_statefulpartitionedcall_args_2'lstm_117_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16676942"
 lstm_117/StatefulPartitionedCall�
!dense_117/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0(dense_117_statefulpartitionedcall_args_1(dense_117_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_16678782#
!dense_117/StatefulPartitionedCall�
IdentityIdentity*dense_117/StatefulPartitionedCall:output:0"^dense_117/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall:. *
(
_user_specified_namelstm_117_input
�
�
while_cond_1668522
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1668522___redundant_placeholder0/
+while_cond_1668522___redundant_placeholder1/
+while_cond_1668522___redundant_placeholder2/
+while_cond_1668522___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�.
�
while_body_1668523
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
F__inference_dense_117_layer_call_and_return_conditional_losses_1668956

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�/
�
lstm_117_while_body_1668032
lstm_117_while_loop_counter%
!lstm_117_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_117_strided_slice_1_0Z
Vtensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_117_strided_slice_1X
Ttensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemVtensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/yg
add_3AddV2lstm_117_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identity!lstm_117_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"6
lstm_117_strided_slice_1lstm_117_strided_slice_1_0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"�
Ttensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensorVtensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
lstm_117_while_cond_1668031
lstm_117_while_loop_counter%
!lstm_117_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3!
less_lstm_117_strided_slice_18
4lstm_117_while_cond_1668031___redundant_placeholder08
4lstm_117_while_cond_1668031___redundant_placeholder18
4lstm_117_while_cond_1668031___redundant_placeholder28
4lstm_117_while_cond_1668031___redundant_placeholder3
identity
a
LessLessplaceholderless_lstm_117_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�
�
/__inference_lstm_cell_178_layer_call_fn_1669043

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:���������:���������:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_16670372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������
:���������:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
Ì
�
"__inference__wrapped_model_1666964
lstm_117_input:
6sequential_117_lstm_117_matmul_readvariableop_resource<
8sequential_117_lstm_117_matmul_1_readvariableop_resource;
7sequential_117_lstm_117_biasadd_readvariableop_resource;
7sequential_117_dense_117_matmul_readvariableop_resource<
8sequential_117_dense_117_biasadd_readvariableop_resource
identity��/sequential_117/dense_117/BiasAdd/ReadVariableOp�.sequential_117/dense_117/MatMul/ReadVariableOp�.sequential_117/lstm_117/BiasAdd/ReadVariableOp�-sequential_117/lstm_117/MatMul/ReadVariableOp�/sequential_117/lstm_117/MatMul_1/ReadVariableOp�sequential_117/lstm_117/while|
sequential_117/lstm_117/ShapeShapelstm_117_input*
T0*
_output_shapes
:2
sequential_117/lstm_117/Shape�
+sequential_117/lstm_117/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_117/lstm_117/strided_slice/stack�
-sequential_117/lstm_117/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_117/lstm_117/strided_slice/stack_1�
-sequential_117/lstm_117/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_117/lstm_117/strided_slice/stack_2�
%sequential_117/lstm_117/strided_sliceStridedSlice&sequential_117/lstm_117/Shape:output:04sequential_117/lstm_117/strided_slice/stack:output:06sequential_117/lstm_117/strided_slice/stack_1:output:06sequential_117/lstm_117/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_117/lstm_117/strided_slice�
#sequential_117/lstm_117/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_117/lstm_117/zeros/mul/y�
!sequential_117/lstm_117/zeros/mulMul.sequential_117/lstm_117/strided_slice:output:0,sequential_117/lstm_117/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_117/lstm_117/zeros/mul�
$sequential_117/lstm_117/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$sequential_117/lstm_117/zeros/Less/y�
"sequential_117/lstm_117/zeros/LessLess%sequential_117/lstm_117/zeros/mul:z:0-sequential_117/lstm_117/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_117/lstm_117/zeros/Less�
&sequential_117/lstm_117/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_117/lstm_117/zeros/packed/1�
$sequential_117/lstm_117/zeros/packedPack.sequential_117/lstm_117/strided_slice:output:0/sequential_117/lstm_117/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_117/lstm_117/zeros/packed�
#sequential_117/lstm_117/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_117/lstm_117/zeros/Const�
sequential_117/lstm_117/zerosFill-sequential_117/lstm_117/zeros/packed:output:0,sequential_117/lstm_117/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
sequential_117/lstm_117/zeros�
%sequential_117/lstm_117/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_117/lstm_117/zeros_1/mul/y�
#sequential_117/lstm_117/zeros_1/mulMul.sequential_117/lstm_117/strided_slice:output:0.sequential_117/lstm_117/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_117/lstm_117/zeros_1/mul�
&sequential_117/lstm_117/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2(
&sequential_117/lstm_117/zeros_1/Less/y�
$sequential_117/lstm_117/zeros_1/LessLess'sequential_117/lstm_117/zeros_1/mul:z:0/sequential_117/lstm_117/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2&
$sequential_117/lstm_117/zeros_1/Less�
(sequential_117/lstm_117/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_117/lstm_117/zeros_1/packed/1�
&sequential_117/lstm_117/zeros_1/packedPack.sequential_117/lstm_117/strided_slice:output:01sequential_117/lstm_117/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_117/lstm_117/zeros_1/packed�
%sequential_117/lstm_117/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%sequential_117/lstm_117/zeros_1/Const�
sequential_117/lstm_117/zeros_1Fill/sequential_117/lstm_117/zeros_1/packed:output:0.sequential_117/lstm_117/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2!
sequential_117/lstm_117/zeros_1�
&sequential_117/lstm_117/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_117/lstm_117/transpose/perm�
!sequential_117/lstm_117/transpose	Transposelstm_117_input/sequential_117/lstm_117/transpose/perm:output:0*
T0*+
_output_shapes
:���������
2#
!sequential_117/lstm_117/transpose�
sequential_117/lstm_117/Shape_1Shape%sequential_117/lstm_117/transpose:y:0*
T0*
_output_shapes
:2!
sequential_117/lstm_117/Shape_1�
-sequential_117/lstm_117/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_117/lstm_117/strided_slice_1/stack�
/sequential_117/lstm_117/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_117/lstm_117/strided_slice_1/stack_1�
/sequential_117/lstm_117/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_117/lstm_117/strided_slice_1/stack_2�
'sequential_117/lstm_117/strided_slice_1StridedSlice(sequential_117/lstm_117/Shape_1:output:06sequential_117/lstm_117/strided_slice_1/stack:output:08sequential_117/lstm_117/strided_slice_1/stack_1:output:08sequential_117/lstm_117/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'sequential_117/lstm_117/strided_slice_1�
3sequential_117/lstm_117/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������25
3sequential_117/lstm_117/TensorArrayV2/element_shape�
%sequential_117/lstm_117/TensorArrayV2TensorListReserve<sequential_117/lstm_117/TensorArrayV2/element_shape:output:00sequential_117/lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_117/lstm_117/TensorArrayV2�
Msequential_117/lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   2O
Msequential_117/lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape�
?sequential_117/lstm_117/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%sequential_117/lstm_117/transpose:y:0Vsequential_117/lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?sequential_117/lstm_117/TensorArrayUnstack/TensorListFromTensor�
-sequential_117/lstm_117/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_117/lstm_117/strided_slice_2/stack�
/sequential_117/lstm_117/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_117/lstm_117/strided_slice_2/stack_1�
/sequential_117/lstm_117/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_117/lstm_117/strided_slice_2/stack_2�
'sequential_117/lstm_117/strided_slice_2StridedSlice%sequential_117/lstm_117/transpose:y:06sequential_117/lstm_117/strided_slice_2/stack:output:08sequential_117/lstm_117/strided_slice_2/stack_1:output:08sequential_117/lstm_117/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2)
'sequential_117/lstm_117/strided_slice_2�
-sequential_117/lstm_117/MatMul/ReadVariableOpReadVariableOp6sequential_117_lstm_117_matmul_readvariableop_resource*
_output_shapes

:
P*
dtype02/
-sequential_117/lstm_117/MatMul/ReadVariableOp�
sequential_117/lstm_117/MatMulMatMul0sequential_117/lstm_117/strided_slice_2:output:05sequential_117/lstm_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2 
sequential_117/lstm_117/MatMul�
/sequential_117/lstm_117/MatMul_1/ReadVariableOpReadVariableOp8sequential_117_lstm_117_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype021
/sequential_117/lstm_117/MatMul_1/ReadVariableOp�
 sequential_117/lstm_117/MatMul_1MatMul&sequential_117/lstm_117/zeros:output:07sequential_117/lstm_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2"
 sequential_117/lstm_117/MatMul_1�
sequential_117/lstm_117/addAddV2(sequential_117/lstm_117/MatMul:product:0*sequential_117/lstm_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
sequential_117/lstm_117/add�
.sequential_117/lstm_117/BiasAdd/ReadVariableOpReadVariableOp7sequential_117_lstm_117_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype020
.sequential_117/lstm_117/BiasAdd/ReadVariableOp�
sequential_117/lstm_117/BiasAddBiasAddsequential_117/lstm_117/add:z:06sequential_117/lstm_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2!
sequential_117/lstm_117/BiasAdd�
sequential_117/lstm_117/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
sequential_117/lstm_117/Const�
'sequential_117/lstm_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_117/lstm_117/split/split_dim�
sequential_117/lstm_117/splitSplit0sequential_117/lstm_117/split/split_dim:output:0(sequential_117/lstm_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
sequential_117/lstm_117/split�
sequential_117/lstm_117/SigmoidSigmoid&sequential_117/lstm_117/split:output:0*
T0*'
_output_shapes
:���������2!
sequential_117/lstm_117/Sigmoid�
!sequential_117/lstm_117/Sigmoid_1Sigmoid&sequential_117/lstm_117/split:output:1*
T0*'
_output_shapes
:���������2#
!sequential_117/lstm_117/Sigmoid_1�
sequential_117/lstm_117/mulMul%sequential_117/lstm_117/Sigmoid_1:y:0(sequential_117/lstm_117/zeros_1:output:0*
T0*'
_output_shapes
:���������2
sequential_117/lstm_117/mul�
sequential_117/lstm_117/ReluRelu&sequential_117/lstm_117/split:output:2*
T0*'
_output_shapes
:���������2
sequential_117/lstm_117/Relu�
sequential_117/lstm_117/mul_1Mul#sequential_117/lstm_117/Sigmoid:y:0*sequential_117/lstm_117/Relu:activations:0*
T0*'
_output_shapes
:���������2
sequential_117/lstm_117/mul_1�
sequential_117/lstm_117/add_1AddV2sequential_117/lstm_117/mul:z:0!sequential_117/lstm_117/mul_1:z:0*
T0*'
_output_shapes
:���������2
sequential_117/lstm_117/add_1�
!sequential_117/lstm_117/Sigmoid_2Sigmoid&sequential_117/lstm_117/split:output:3*
T0*'
_output_shapes
:���������2#
!sequential_117/lstm_117/Sigmoid_2�
sequential_117/lstm_117/Relu_1Relu!sequential_117/lstm_117/add_1:z:0*
T0*'
_output_shapes
:���������2 
sequential_117/lstm_117/Relu_1�
sequential_117/lstm_117/mul_2Mul%sequential_117/lstm_117/Sigmoid_2:y:0,sequential_117/lstm_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������2
sequential_117/lstm_117/mul_2�
5sequential_117/lstm_117/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5sequential_117/lstm_117/TensorArrayV2_1/element_shape�
'sequential_117/lstm_117/TensorArrayV2_1TensorListReserve>sequential_117/lstm_117/TensorArrayV2_1/element_shape:output:00sequential_117/lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'sequential_117/lstm_117/TensorArrayV2_1~
sequential_117/lstm_117/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_117/lstm_117/time�
0sequential_117/lstm_117/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_117/lstm_117/while/maximum_iterations�
*sequential_117/lstm_117/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_117/lstm_117/while/loop_counter�	
sequential_117/lstm_117/whileWhile3sequential_117/lstm_117/while/loop_counter:output:09sequential_117/lstm_117/while/maximum_iterations:output:0%sequential_117/lstm_117/time:output:00sequential_117/lstm_117/TensorArrayV2_1:handle:0&sequential_117/lstm_117/zeros:output:0(sequential_117/lstm_117/zeros_1:output:00sequential_117/lstm_117/strided_slice_1:output:0Osequential_117/lstm_117/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_117_lstm_117_matmul_readvariableop_resource8sequential_117_lstm_117_matmul_1_readvariableop_resource7sequential_117_lstm_117_biasadd_readvariableop_resource/^sequential_117/lstm_117/BiasAdd/ReadVariableOp.^sequential_117/lstm_117/MatMul/ReadVariableOp0^sequential_117/lstm_117/MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *6
body.R,
*sequential_117_lstm_117_while_body_1666873*6
cond.R,
*sequential_117_lstm_117_while_cond_1666872*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
sequential_117/lstm_117/while�
Hsequential_117/lstm_117/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hsequential_117/lstm_117/TensorArrayV2Stack/TensorListStack/element_shape�
:sequential_117/lstm_117/TensorArrayV2Stack/TensorListStackTensorListStack&sequential_117/lstm_117/while:output:3Qsequential_117/lstm_117/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype02<
:sequential_117/lstm_117/TensorArrayV2Stack/TensorListStack�
-sequential_117/lstm_117/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2/
-sequential_117/lstm_117/strided_slice_3/stack�
/sequential_117/lstm_117/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_117/lstm_117/strided_slice_3/stack_1�
/sequential_117/lstm_117/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_117/lstm_117/strided_slice_3/stack_2�
'sequential_117/lstm_117/strided_slice_3StridedSliceCsequential_117/lstm_117/TensorArrayV2Stack/TensorListStack:tensor:06sequential_117/lstm_117/strided_slice_3/stack:output:08sequential_117/lstm_117/strided_slice_3/stack_1:output:08sequential_117/lstm_117/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2)
'sequential_117/lstm_117/strided_slice_3�
(sequential_117/lstm_117/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential_117/lstm_117/transpose_1/perm�
#sequential_117/lstm_117/transpose_1	TransposeCsequential_117/lstm_117/TensorArrayV2Stack/TensorListStack:tensor:01sequential_117/lstm_117/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������2%
#sequential_117/lstm_117/transpose_1�
sequential_117/lstm_117/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_117/lstm_117/runtime�
.sequential_117/dense_117/MatMul/ReadVariableOpReadVariableOp7sequential_117_dense_117_matmul_readvariableop_resource*
_output_shapes

:
*
dtype020
.sequential_117/dense_117/MatMul/ReadVariableOp�
sequential_117/dense_117/MatMulMatMul0sequential_117/lstm_117/strided_slice_3:output:06sequential_117/dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2!
sequential_117/dense_117/MatMul�
/sequential_117/dense_117/BiasAdd/ReadVariableOpReadVariableOp8sequential_117_dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype021
/sequential_117/dense_117/BiasAdd/ReadVariableOp�
 sequential_117/dense_117/BiasAddBiasAdd)sequential_117/dense_117/MatMul:product:07sequential_117/dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2"
 sequential_117/dense_117/BiasAdd�
IdentityIdentity)sequential_117/dense_117/BiasAdd:output:00^sequential_117/dense_117/BiasAdd/ReadVariableOp/^sequential_117/dense_117/MatMul/ReadVariableOp/^sequential_117/lstm_117/BiasAdd/ReadVariableOp.^sequential_117/lstm_117/MatMul/ReadVariableOp0^sequential_117/lstm_117/MatMul_1/ReadVariableOp^sequential_117/lstm_117/while*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::2b
/sequential_117/dense_117/BiasAdd/ReadVariableOp/sequential_117/dense_117/BiasAdd/ReadVariableOp2`
.sequential_117/dense_117/MatMul/ReadVariableOp.sequential_117/dense_117/MatMul/ReadVariableOp2`
.sequential_117/lstm_117/BiasAdd/ReadVariableOp.sequential_117/lstm_117/BiasAdd/ReadVariableOp2^
-sequential_117/lstm_117/MatMul/ReadVariableOp-sequential_117/lstm_117/MatMul/ReadVariableOp2b
/sequential_117/lstm_117/MatMul_1/ReadVariableOp/sequential_117/lstm_117/MatMul_1/ReadVariableOp2>
sequential_117/lstm_117/whilesequential_117/lstm_117/while:. *
(
_user_specified_namelstm_117_input
�
�
while_cond_1667608
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1667608___redundant_placeholder0/
+while_cond_1667608___redundant_placeholder1/
+while_cond_1667608___redundant_placeholder2/
+while_cond_1667608___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�
�
while_body_1667343
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:���������:���������:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_16670372
StatefulPartitionedCall�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::22
StatefulPartitionedCallStatefulPartitionedCall
�
�
while_cond_1667761
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1667761___redundant_placeholder0/
+while_cond_1667761___redundant_placeholder1/
+while_cond_1667761___redundant_placeholder2/
+while_cond_1667761___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�
�
*sequential_117_lstm_117_while_cond_1666872.
*sequential_117_lstm_117_while_loop_counter4
0sequential_117_lstm_117_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_30
,less_sequential_117_lstm_117_strided_slice_1G
Csequential_117_lstm_117_while_cond_1666872___redundant_placeholder0G
Csequential_117_lstm_117_while_cond_1666872___redundant_placeholder1G
Csequential_117_lstm_117_while_cond_1666872___redundant_placeholder2G
Csequential_117_lstm_117_while_cond_1666872___redundant_placeholder3
identity
p
LessLessplaceholder,less_sequential_117_lstm_117_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�
�
%__inference_signature_wrapper_1667964
lstm_117_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__wrapped_model_16669642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namelstm_117_input
�V
�
#__inference__traced_restore_1669213
file_prefix%
!assignvariableop_dense_117_kernel%
!assignvariableop_1_dense_117_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate&
"assignvariableop_7_lstm_117_kernel0
,assignvariableop_8_lstm_117_recurrent_kernel$
 assignvariableop_9_lstm_117_bias/
+assignvariableop_10_adam_dense_117_kernel_m-
)assignvariableop_11_adam_dense_117_bias_m.
*assignvariableop_12_adam_lstm_117_kernel_m8
4assignvariableop_13_adam_lstm_117_recurrent_kernel_m,
(assignvariableop_14_adam_lstm_117_bias_m/
+assignvariableop_15_adam_dense_117_kernel_v-
)assignvariableop_16_adam_dense_117_bias_v.
*assignvariableop_17_adam_lstm_117_kernel_v8
4assignvariableop_18_adam_lstm_117_recurrent_kernel_v,
(assignvariableop_19_adam_lstm_117_bias_v
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_dense_117_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_117_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_lstm_117_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp,assignvariableop_8_lstm_117_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_lstm_117_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_adam_dense_117_kernel_mIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_117_bias_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_lstm_117_kernel_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_lstm_117_recurrent_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_lstm_117_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_117_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_117_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_lstm_117_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_lstm_117_recurrent_kernel_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_lstm_117_bias_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20�
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�1
�	
 __inference__traced_save_1669141
file_prefix/
+savev2_dense_117_kernel_read_readvariableop-
)savev2_dense_117_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_lstm_117_kernel_read_readvariableop8
4savev2_lstm_117_recurrent_kernel_read_readvariableop,
(savev2_lstm_117_bias_read_readvariableop6
2savev2_adam_dense_117_kernel_m_read_readvariableop4
0savev2_adam_dense_117_bias_m_read_readvariableop5
1savev2_adam_lstm_117_kernel_m_read_readvariableop?
;savev2_adam_lstm_117_recurrent_kernel_m_read_readvariableop3
/savev2_adam_lstm_117_bias_m_read_readvariableop6
2savev2_adam_dense_117_kernel_v_read_readvariableop4
0savev2_adam_dense_117_bias_v_read_readvariableop5
1savev2_adam_lstm_117_kernel_v_read_readvariableop?
;savev2_adam_lstm_117_recurrent_kernel_v_read_readvariableop3
/savev2_adam_lstm_117_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0a8c1274775d4bc4ae3a502e056f6381/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_117_kernel_read_readvariableop)savev2_dense_117_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_lstm_117_kernel_read_readvariableop4savev2_lstm_117_recurrent_kernel_read_readvariableop(savev2_lstm_117_bias_read_readvariableop2savev2_adam_dense_117_kernel_m_read_readvariableop0savev2_adam_dense_117_bias_m_read_readvariableop1savev2_adam_lstm_117_kernel_m_read_readvariableop;savev2_adam_lstm_117_recurrent_kernel_m_read_readvariableop/savev2_adam_lstm_117_bias_m_read_readvariableop2savev2_adam_dense_117_kernel_v_read_readvariableop0savev2_adam_dense_117_bias_v_read_readvariableop1savev2_adam_lstm_117_kernel_v_read_readvariableop;savev2_adam_lstm_117_recurrent_kernel_v_read_readvariableop/savev2_adam_lstm_117_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
:
: : : : : :
P:P:P:
:
:
P:P:P:
:
:
P:P:P: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1667037

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������
:���������:���������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates
�
�
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1669029

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������
:���������:���������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
�
�
0__inference_sequential_117_layer_call_fn_1667924
lstm_117_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_117_layer_call_and_return_conditional_losses_16679162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namelstm_117_input
�.
�
while_body_1668845
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�l
�
K__inference_sequential_117_layer_call_and_return_conditional_losses_1668123

inputs+
'lstm_117_matmul_readvariableop_resource-
)lstm_117_matmul_1_readvariableop_resource,
(lstm_117_biasadd_readvariableop_resource,
(dense_117_matmul_readvariableop_resource-
)dense_117_biasadd_readvariableop_resource
identity�� dense_117/BiasAdd/ReadVariableOp�dense_117/MatMul/ReadVariableOp�lstm_117/BiasAdd/ReadVariableOp�lstm_117/MatMul/ReadVariableOp� lstm_117/MatMul_1/ReadVariableOp�lstm_117/whileV
lstm_117/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_117/Shape�
lstm_117/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_117/strided_slice/stack�
lstm_117/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_117/strided_slice/stack_1�
lstm_117/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_117/strided_slice/stack_2�
lstm_117/strided_sliceStridedSlicelstm_117/Shape:output:0%lstm_117/strided_slice/stack:output:0'lstm_117/strided_slice/stack_1:output:0'lstm_117/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_117/strided_slicen
lstm_117/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros/mul/y�
lstm_117/zeros/mulMullstm_117/strided_slice:output:0lstm_117/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros/mulq
lstm_117/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_117/zeros/Less/y�
lstm_117/zeros/LessLesslstm_117/zeros/mul:z:0lstm_117/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros/Lesst
lstm_117/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros/packed/1�
lstm_117/zeros/packedPacklstm_117/strided_slice:output:0 lstm_117/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_117/zeros/packedq
lstm_117/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_117/zeros/Const�
lstm_117/zerosFilllstm_117/zeros/packed:output:0lstm_117/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
lstm_117/zerosr
lstm_117/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros_1/mul/y�
lstm_117/zeros_1/mulMullstm_117/strided_slice:output:0lstm_117/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros_1/mulu
lstm_117/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_117/zeros_1/Less/y�
lstm_117/zeros_1/LessLesslstm_117/zeros_1/mul:z:0 lstm_117/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_117/zeros_1/Lessx
lstm_117/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/zeros_1/packed/1�
lstm_117/zeros_1/packedPacklstm_117/strided_slice:output:0"lstm_117/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_117/zeros_1/packedu
lstm_117/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_117/zeros_1/Const�
lstm_117/zeros_1Fill lstm_117/zeros_1/packed:output:0lstm_117/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2
lstm_117/zeros_1�
lstm_117/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_117/transpose/perm�
lstm_117/transpose	Transposeinputs lstm_117/transpose/perm:output:0*
T0*+
_output_shapes
:���������
2
lstm_117/transposej
lstm_117/Shape_1Shapelstm_117/transpose:y:0*
T0*
_output_shapes
:2
lstm_117/Shape_1�
lstm_117/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_117/strided_slice_1/stack�
 lstm_117/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_1/stack_1�
 lstm_117/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_1/stack_2�
lstm_117/strided_slice_1StridedSlicelstm_117/Shape_1:output:0'lstm_117/strided_slice_1/stack:output:0)lstm_117/strided_slice_1/stack_1:output:0)lstm_117/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_117/strided_slice_1�
$lstm_117/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$lstm_117/TensorArrayV2/element_shape�
lstm_117/TensorArrayV2TensorListReserve-lstm_117/TensorArrayV2/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_117/TensorArrayV2�
>lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   2@
>lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape�
0lstm_117/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_117/transpose:y:0Glstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0lstm_117/TensorArrayUnstack/TensorListFromTensor�
lstm_117/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_117/strided_slice_2/stack�
 lstm_117/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_2/stack_1�
 lstm_117/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_2/stack_2�
lstm_117/strided_slice_2StridedSlicelstm_117/transpose:y:0'lstm_117/strided_slice_2/stack:output:0)lstm_117/strided_slice_2/stack_1:output:0)lstm_117/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
lstm_117/strided_slice_2�
lstm_117/MatMul/ReadVariableOpReadVariableOp'lstm_117_matmul_readvariableop_resource*
_output_shapes

:
P*
dtype02 
lstm_117/MatMul/ReadVariableOp�
lstm_117/MatMulMatMul!lstm_117/strided_slice_2:output:0&lstm_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
lstm_117/MatMul�
 lstm_117/MatMul_1/ReadVariableOpReadVariableOp)lstm_117_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02"
 lstm_117/MatMul_1/ReadVariableOp�
lstm_117/MatMul_1MatMullstm_117/zeros:output:0(lstm_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
lstm_117/MatMul_1�
lstm_117/addAddV2lstm_117/MatMul:product:0lstm_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
lstm_117/add�
lstm_117/BiasAdd/ReadVariableOpReadVariableOp(lstm_117_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02!
lstm_117/BiasAdd/ReadVariableOp�
lstm_117/BiasAddBiasAddlstm_117/add:z:0'lstm_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
lstm_117/BiasAddb
lstm_117/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/Constv
lstm_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_117/split/split_dim�
lstm_117/splitSplit!lstm_117/split/split_dim:output:0lstm_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
lstm_117/splitz
lstm_117/SigmoidSigmoidlstm_117/split:output:0*
T0*'
_output_shapes
:���������2
lstm_117/Sigmoid~
lstm_117/Sigmoid_1Sigmoidlstm_117/split:output:1*
T0*'
_output_shapes
:���������2
lstm_117/Sigmoid_1�
lstm_117/mulMullstm_117/Sigmoid_1:y:0lstm_117/zeros_1:output:0*
T0*'
_output_shapes
:���������2
lstm_117/mulq
lstm_117/ReluRelulstm_117/split:output:2*
T0*'
_output_shapes
:���������2
lstm_117/Relu�
lstm_117/mul_1Mullstm_117/Sigmoid:y:0lstm_117/Relu:activations:0*
T0*'
_output_shapes
:���������2
lstm_117/mul_1�
lstm_117/add_1AddV2lstm_117/mul:z:0lstm_117/mul_1:z:0*
T0*'
_output_shapes
:���������2
lstm_117/add_1~
lstm_117/Sigmoid_2Sigmoidlstm_117/split:output:3*
T0*'
_output_shapes
:���������2
lstm_117/Sigmoid_2p
lstm_117/Relu_1Relulstm_117/add_1:z:0*
T0*'
_output_shapes
:���������2
lstm_117/Relu_1�
lstm_117/mul_2Mullstm_117/Sigmoid_2:y:0lstm_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������2
lstm_117/mul_2�
&lstm_117/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2(
&lstm_117/TensorArrayV2_1/element_shape�
lstm_117/TensorArrayV2_1TensorListReserve/lstm_117/TensorArrayV2_1/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_117/TensorArrayV2_1`
lstm_117/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_117/time�
!lstm_117/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!lstm_117/while/maximum_iterations|
lstm_117/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_117/while/loop_counter�
lstm_117/whileWhile$lstm_117/while/loop_counter:output:0*lstm_117/while/maximum_iterations:output:0lstm_117/time:output:0!lstm_117/TensorArrayV2_1:handle:0lstm_117/zeros:output:0lstm_117/zeros_1:output:0!lstm_117/strided_slice_1:output:0@lstm_117/TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_117_matmul_readvariableop_resource)lstm_117_matmul_1_readvariableop_resource(lstm_117_biasadd_readvariableop_resource ^lstm_117/BiasAdd/ReadVariableOp^lstm_117/MatMul/ReadVariableOp!^lstm_117/MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *'
bodyR
lstm_117_while_body_1668032*'
condR
lstm_117_while_cond_1668031*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
lstm_117/while�
9lstm_117/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2;
9lstm_117/TensorArrayV2Stack/TensorListStack/element_shape�
+lstm_117/TensorArrayV2Stack/TensorListStackTensorListStacklstm_117/while:output:3Blstm_117/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype02-
+lstm_117/TensorArrayV2Stack/TensorListStack�
lstm_117/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2 
lstm_117/strided_slice_3/stack�
 lstm_117/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 lstm_117/strided_slice_3/stack_1�
 lstm_117/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lstm_117/strided_slice_3/stack_2�
lstm_117/strided_slice_3StridedSlice4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_117/strided_slice_3/stack:output:0)lstm_117/strided_slice_3/stack_1:output:0)lstm_117/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_117/strided_slice_3�
lstm_117/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_117/transpose_1/perm�
lstm_117/transpose_1	Transpose4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_117/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_117/transpose_1x
lstm_117/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_117/runtime�
dense_117/MatMul/ReadVariableOpReadVariableOp(dense_117_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_117/MatMul/ReadVariableOp�
dense_117/MatMulMatMul!lstm_117/strided_slice_3:output:0'dense_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_117/MatMul�
 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_117/BiasAdd/ReadVariableOp�
dense_117/BiasAddBiasAdddense_117/MatMul:product:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_117/BiasAdd�
IdentityIdentitydense_117/BiasAdd:output:0!^dense_117/BiasAdd/ReadVariableOp ^dense_117/MatMul/ReadVariableOp ^lstm_117/BiasAdd/ReadVariableOp^lstm_117/MatMul/ReadVariableOp!^lstm_117/MatMul_1/ReadVariableOp^lstm_117/while*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2B
dense_117/MatMul/ReadVariableOpdense_117/MatMul/ReadVariableOp2B
lstm_117/BiasAdd/ReadVariableOplstm_117/BiasAdd/ReadVariableOp2@
lstm_117/MatMul/ReadVariableOplstm_117/MatMul/ReadVariableOp2D
 lstm_117/MatMul_1/ReadVariableOp lstm_117/MatMul_1/ReadVariableOp2 
lstm_117/whilelstm_117/while:& "
 
_user_specified_nameinputs
�.
�
while_body_1668692
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1667070

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������
:���������:���������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates
�
�
while_cond_1668691
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1668691___redundant_placeholder0/
+while_cond_1668691___redundant_placeholder1/
+while_cond_1668691___redundant_placeholder2/
+while_cond_1668691___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�R
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668930

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1d
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1668845*
condR
while_cond_1668844*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�
�
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667902
lstm_117_input+
'lstm_117_statefulpartitionedcall_args_1+
'lstm_117_statefulpartitionedcall_args_2+
'lstm_117_statefulpartitionedcall_args_3,
(dense_117_statefulpartitionedcall_args_1,
(dense_117_statefulpartitionedcall_args_2
identity��!dense_117/StatefulPartitionedCall� lstm_117/StatefulPartitionedCall�
 lstm_117/StatefulPartitionedCallStatefulPartitionedCalllstm_117_input'lstm_117_statefulpartitionedcall_args_1'lstm_117_statefulpartitionedcall_args_2'lstm_117_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16678472"
 lstm_117/StatefulPartitionedCall�
!dense_117/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0(dense_117_statefulpartitionedcall_args_1(dense_117_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_16678782#
!dense_117/StatefulPartitionedCall�
IdentityIdentity*dense_117/StatefulPartitionedCall:output:0"^dense_117/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall:. *
(
_user_specified_namelstm_117_input
�
�
while_cond_1668844
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1668844___redundant_placeholder0/
+while_cond_1668844___redundant_placeholder1/
+while_cond_1668844___redundant_placeholder2/
+while_cond_1668844___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::
�0
�
*sequential_117_lstm_117_while_body_1666873.
*sequential_117_lstm_117_while_loop_counter4
0sequential_117_lstm_117_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3-
)sequential_117_lstm_117_strided_slice_1_0i
etensorarrayv2read_tensorlistgetitem_sequential_117_lstm_117_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5+
'sequential_117_lstm_117_strided_slice_1g
ctensorarrayv2read_tensorlistgetitem_sequential_117_lstm_117_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemetensorarrayv2read_tensorlistgetitem_sequential_117_lstm_117_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/yv
add_3AddV2*sequential_117_lstm_117_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identity0sequential_117_lstm_117_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"T
'sequential_117_lstm_117_strided_slice_1)sequential_117_lstm_117_strided_slice_1_0"�
ctensorarrayv2read_tensorlistgetitem_sequential_117_lstm_117_tensorarrayunstack_tensorlistfromtensoretensorarrayv2read_tensorlistgetitem_sequential_117_lstm_117_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
F__inference_dense_117_layer_call_and_return_conditional_losses_1667878

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
*__inference_lstm_117_layer_call_fn_1668946

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16678472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
0__inference_sequential_117_layer_call_fn_1668292

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_117_layer_call_and_return_conditional_losses_16679162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
0__inference_sequential_117_layer_call_fn_1668302

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_117_layer_call_and_return_conditional_losses_16679372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
/__inference_lstm_cell_178_layer_call_fn_1669057

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:���������:���������:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_16670702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:���������
:���������:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
�.
�
while_body_1668370
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667937

inputs+
'lstm_117_statefulpartitionedcall_args_1+
'lstm_117_statefulpartitionedcall_args_2+
'lstm_117_statefulpartitionedcall_args_3,
(dense_117_statefulpartitionedcall_args_1,
(dense_117_statefulpartitionedcall_args_2
identity��!dense_117/StatefulPartitionedCall� lstm_117/StatefulPartitionedCall�
 lstm_117/StatefulPartitionedCallStatefulPartitionedCallinputs'lstm_117_statefulpartitionedcall_args_1'lstm_117_statefulpartitionedcall_args_2'lstm_117_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16678472"
 lstm_117/StatefulPartitionedCall�
!dense_117/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0(dense_117_statefulpartitionedcall_args_1(dense_117_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_16678782#
!dense_117/StatefulPartitionedCall�
IdentityIdentity*dense_117/StatefulPartitionedCall:output:0"^dense_117/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667916

inputs+
'lstm_117_statefulpartitionedcall_args_1+
'lstm_117_statefulpartitionedcall_args_2+
'lstm_117_statefulpartitionedcall_args_3,
(dense_117_statefulpartitionedcall_args_1,
(dense_117_statefulpartitionedcall_args_2
identity��!dense_117/StatefulPartitionedCall� lstm_117/StatefulPartitionedCall�
 lstm_117/StatefulPartitionedCallStatefulPartitionedCallinputs'lstm_117_statefulpartitionedcall_args_1'lstm_117_statefulpartitionedcall_args_2'lstm_117_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_lstm_117_layer_call_and_return_conditional_losses_16676942"
 lstm_117/StatefulPartitionedCall�
!dense_117/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0(dense_117_statefulpartitionedcall_args_1(dense_117_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_117_layer_call_and_return_conditional_losses_16678782#
!dense_117/StatefulPartitionedCall�
IdentityIdentity*dense_117/StatefulPartitionedCall:output:0"^dense_117/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�R
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668777

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1d
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1668692*
condR
while_cond_1668691*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�/
�
lstm_117_while_body_1668191
lstm_117_while_loop_counter%
!lstm_117_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_117_strided_slice_1_0Z
Vtensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_117_strided_slice_1X
Ttensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemVtensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������
*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/yg
add_3AddV2lstm_117_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identity!lstm_117_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"6
lstm_117_strided_slice_1lstm_117_strided_slice_1_0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"�
Ttensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensorVtensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :���������:���������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�S
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668608
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������
2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*
shrink_axis_mask2
strided_slice_2�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
P*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������P2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������2
	Sigmoid_1d
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *
bodyR
while_body_1668523*
condR
while_cond_1668522*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
�
�
0__inference_sequential_117_layer_call_fn_1667945
lstm_117_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_117_layer_call_and_return_conditional_losses_16679372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������
:::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namelstm_117_input
�
�
while_cond_1667465
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1/
+while_cond_1667465___redundant_placeholder0/
+while_cond_1667465___redundant_placeholder1/
+while_cond_1667465___redundant_placeholder2/
+while_cond_1667465___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :���������:���������: ::::"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
lstm_117_input;
 serving_default_lstm_117_input:0���������
=
	dense_1170
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
<_default_save_signature
*=&call_and_return_all_conditional_losses
>__call__"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_117", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_117", "layers": [{"class_name": "LSTM", "config": {"name": "lstm_117", "trainable": true, "batch_input_shape": [null, 14, 10], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_117", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 10], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_117", "layers": [{"class_name": "LSTM", "config": {"name": "lstm_117", "trainable": true, "batch_input_shape": [null, 14, 10], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_117", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "lstm_117_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 14, 10], "config": {"batch_input_shape": [null, 14, 10], "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_117_input"}}
�


cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"�	
_tf_keras_layer�	{"class_name": "LSTM", "name": "lstm_117", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 14, 10], "config": {"name": "lstm_117", "trainable": true, "batch_input_shape": [null, 14, 10], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 20, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 10], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_117", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_117", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}}
�
iter

beta_1

beta_2
	decay
learning_ratem2m3m4m5m6v7v8v9v:v;"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables

layers
layer_regularization_losses
trainable_variables
 non_trainable_variables
regularization_losses
!metrics
>__call__
<_default_save_signature
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
,
Cserving_default"
signature_map
�

kernel
recurrent_kernel
bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
*D&call_and_return_all_conditional_losses
E__call__"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell_178", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_178", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
&layer_regularization_losses

'layers
(non_trainable_variables
trainable_variables
regularization_losses
)metrics
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_117/kernel
:
2dense_117/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
*layer_regularization_losses

+layers
,non_trainable_variables
trainable_variables
regularization_losses
-metrics
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
!:
P2lstm_117/kernel
+:)P2lstm_117/recurrent_kernel
:P2lstm_117/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
"	variables
.layer_regularization_losses

/layers
0non_trainable_variables
#trainable_variables
$regularization_losses
1metrics
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
':%
2Adam/dense_117/kernel/m
!:
2Adam/dense_117/bias/m
&:$
P2Adam/lstm_117/kernel/m
0:.P2 Adam/lstm_117/recurrent_kernel/m
 :P2Adam/lstm_117/bias/m
':%
2Adam/dense_117/kernel/v
!:
2Adam/dense_117/bias/v
&:$
P2Adam/lstm_117/kernel/v
0:.P2 Adam/lstm_117/recurrent_kernel/v
 :P2Adam/lstm_117/bias/v
�2�
"__inference__wrapped_model_1666964�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *1�.
,�)
lstm_117_input���������

�2�
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667902
K__inference_sequential_117_layer_call_and_return_conditional_losses_1668282
K__inference_sequential_117_layer_call_and_return_conditional_losses_1668123
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667891�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_sequential_117_layer_call_fn_1668292
0__inference_sequential_117_layer_call_fn_1668302
0__inference_sequential_117_layer_call_fn_1667924
0__inference_sequential_117_layer_call_fn_1667945�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668455
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668608
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668930
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668777�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_lstm_117_layer_call_fn_1668616
*__inference_lstm_117_layer_call_fn_1668624
*__inference_lstm_117_layer_call_fn_1668938
*__inference_lstm_117_layer_call_fn_1668946�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dense_117_layer_call_and_return_conditional_losses_1668956�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_117_layer_call_fn_1668963�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
;B9
%__inference_signature_wrapper_1667964lstm_117_input
�2�
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1668996
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1669029�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_lstm_cell_178_layer_call_fn_1669043
/__inference_lstm_cell_178_layer_call_fn_1669057�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
"__inference__wrapped_model_1666964{;�8
1�.
,�)
lstm_117_input���������

� "5�2
0
	dense_117#� 
	dense_117���������
�
F__inference_dense_117_layer_call_and_return_conditional_losses_1668956\/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� ~
+__inference_dense_117_layer_call_fn_1668963O/�,
%�"
 �
inputs���������
� "����������
�
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668455}O�L
E�B
4�1
/�,
inputs/0������������������


 
p

 
� "%�"
�
0���������
� �
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668608}O�L
E�B
4�1
/�,
inputs/0������������������


 
p 

 
� "%�"
�
0���������
� �
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668777m?�<
5�2
$�!
inputs���������


 
p

 
� "%�"
�
0���������
� �
E__inference_lstm_117_layer_call_and_return_conditional_losses_1668930m?�<
5�2
$�!
inputs���������


 
p 

 
� "%�"
�
0���������
� �
*__inference_lstm_117_layer_call_fn_1668616pO�L
E�B
4�1
/�,
inputs/0������������������


 
p

 
� "�����������
*__inference_lstm_117_layer_call_fn_1668624pO�L
E�B
4�1
/�,
inputs/0������������������


 
p 

 
� "�����������
*__inference_lstm_117_layer_call_fn_1668938`?�<
5�2
$�!
inputs���������


 
p

 
� "�����������
*__inference_lstm_117_layer_call_fn_1668946`?�<
5�2
$�!
inputs���������


 
p 

 
� "�����������
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1668996���}
v�s
 �
inputs���������

K�H
"�
states/0���������
"�
states/1���������
p
� "s�p
i�f
�
0/0���������
E�B
�
0/1/0���������
�
0/1/1���������
� �
J__inference_lstm_cell_178_layer_call_and_return_conditional_losses_1669029���}
v�s
 �
inputs���������

K�H
"�
states/0���������
"�
states/1���������
p 
� "s�p
i�f
�
0/0���������
E�B
�
0/1/0���������
�
0/1/1���������
� �
/__inference_lstm_cell_178_layer_call_fn_1669043���}
v�s
 �
inputs���������

K�H
"�
states/0���������
"�
states/1���������
p
� "c�`
�
0���������
A�>
�
1/0���������
�
1/1����������
/__inference_lstm_cell_178_layer_call_fn_1669057���}
v�s
 �
inputs���������

K�H
"�
states/0���������
"�
states/1���������
p 
� "c�`
�
0���������
A�>
�
1/0���������
�
1/1����������
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667891sC�@
9�6
,�)
lstm_117_input���������

p

 
� "%�"
�
0���������

� �
K__inference_sequential_117_layer_call_and_return_conditional_losses_1667902sC�@
9�6
,�)
lstm_117_input���������

p 

 
� "%�"
�
0���������

� �
K__inference_sequential_117_layer_call_and_return_conditional_losses_1668123k;�8
1�.
$�!
inputs���������

p

 
� "%�"
�
0���������

� �
K__inference_sequential_117_layer_call_and_return_conditional_losses_1668282k;�8
1�.
$�!
inputs���������

p 

 
� "%�"
�
0���������

� �
0__inference_sequential_117_layer_call_fn_1667924fC�@
9�6
,�)
lstm_117_input���������

p

 
� "����������
�
0__inference_sequential_117_layer_call_fn_1667945fC�@
9�6
,�)
lstm_117_input���������

p 

 
� "����������
�
0__inference_sequential_117_layer_call_fn_1668292^;�8
1�.
$�!
inputs���������

p

 
� "����������
�
0__inference_sequential_117_layer_call_fn_1668302^;�8
1�.
$�!
inputs���������

p 

 
� "����������
�
%__inference_signature_wrapper_1667964�M�J
� 
C�@
>
lstm_117_input,�)
lstm_117_input���������
"5�2
0
	dense_117#� 
	dense_117���������
