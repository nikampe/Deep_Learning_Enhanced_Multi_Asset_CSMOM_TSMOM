��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
(Adam/v/list_net_model_144/dense_878/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/v/list_net_model_144/dense_878/bias
�
<Adam/v/list_net_model_144/dense_878/bias/Read/ReadVariableOpReadVariableOp(Adam/v/list_net_model_144/dense_878/bias*
_output_shapes
:*
dtype0
�
(Adam/m/list_net_model_144/dense_878/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/m/list_net_model_144/dense_878/bias
�
<Adam/m/list_net_model_144/dense_878/bias/Read/ReadVariableOpReadVariableOp(Adam/m/list_net_model_144/dense_878/bias*
_output_shapes
:*
dtype0
�
*Adam/v/list_net_model_144/dense_878/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*;
shared_name,*Adam/v/list_net_model_144/dense_878/kernel
�
>Adam/v/list_net_model_144/dense_878/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/list_net_model_144/dense_878/kernel*
_output_shapes
:	�*
dtype0
�
*Adam/m/list_net_model_144/dense_878/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*;
shared_name,*Adam/m/list_net_model_144/dense_878/kernel
�
>Adam/m/list_net_model_144/dense_878/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/list_net_model_144/dense_878/kernel*
_output_shapes
:	�*
dtype0
�
(Adam/v/list_net_model_144/dense_877/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/v/list_net_model_144/dense_877/bias
�
<Adam/v/list_net_model_144/dense_877/bias/Read/ReadVariableOpReadVariableOp(Adam/v/list_net_model_144/dense_877/bias*
_output_shapes	
:�*
dtype0
�
(Adam/m/list_net_model_144/dense_877/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/m/list_net_model_144/dense_877/bias
�
<Adam/m/list_net_model_144/dense_877/bias/Read/ReadVariableOpReadVariableOp(Adam/m/list_net_model_144/dense_877/bias*
_output_shapes	
:�*
dtype0
�
*Adam/v/list_net_model_144/dense_877/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*;
shared_name,*Adam/v/list_net_model_144/dense_877/kernel
�
>Adam/v/list_net_model_144/dense_877/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/list_net_model_144/dense_877/kernel*
_output_shapes
:	 �*
dtype0
�
*Adam/m/list_net_model_144/dense_877/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*;
shared_name,*Adam/m/list_net_model_144/dense_877/kernel
�
>Adam/m/list_net_model_144/dense_877/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/list_net_model_144/dense_877/kernel*
_output_shapes
:	 �*
dtype0
�
(Adam/v/list_net_model_144/dense_876/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/v/list_net_model_144/dense_876/bias
�
<Adam/v/list_net_model_144/dense_876/bias/Read/ReadVariableOpReadVariableOp(Adam/v/list_net_model_144/dense_876/bias*
_output_shapes
: *
dtype0
�
(Adam/m/list_net_model_144/dense_876/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/m/list_net_model_144/dense_876/bias
�
<Adam/m/list_net_model_144/dense_876/bias/Read/ReadVariableOpReadVariableOp(Adam/m/list_net_model_144/dense_876/bias*
_output_shapes
: *
dtype0
�
*Adam/v/list_net_model_144/dense_876/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*Adam/v/list_net_model_144/dense_876/kernel
�
>Adam/v/list_net_model_144/dense_876/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/list_net_model_144/dense_876/kernel*
_output_shapes

: *
dtype0
�
*Adam/m/list_net_model_144/dense_876/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*Adam/m/list_net_model_144/dense_876/kernel
�
>Adam/m/list_net_model_144/dense_876/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/list_net_model_144/dense_876/kernel*
_output_shapes

: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
!list_net_model_144/dense_878/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!list_net_model_144/dense_878/bias
�
5list_net_model_144/dense_878/bias/Read/ReadVariableOpReadVariableOp!list_net_model_144/dense_878/bias*
_output_shapes
:*
dtype0
�
#list_net_model_144/dense_878/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#list_net_model_144/dense_878/kernel
�
7list_net_model_144/dense_878/kernel/Read/ReadVariableOpReadVariableOp#list_net_model_144/dense_878/kernel*
_output_shapes
:	�*
dtype0
�
!list_net_model_144/dense_877/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!list_net_model_144/dense_877/bias
�
5list_net_model_144/dense_877/bias/Read/ReadVariableOpReadVariableOp!list_net_model_144/dense_877/bias*
_output_shapes	
:�*
dtype0
�
#list_net_model_144/dense_877/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*4
shared_name%#list_net_model_144/dense_877/kernel
�
7list_net_model_144/dense_877/kernel/Read/ReadVariableOpReadVariableOp#list_net_model_144/dense_877/kernel*
_output_shapes
:	 �*
dtype0
�
!list_net_model_144/dense_876/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!list_net_model_144/dense_876/bias
�
5list_net_model_144/dense_876/bias/Read/ReadVariableOpReadVariableOp!list_net_model_144/dense_876/bias*
_output_shapes
: *
dtype0
�
#list_net_model_144/dense_876/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#list_net_model_144/dense_876/kernel
�
7list_net_model_144/dense_876/kernel/Read/ReadVariableOpReadVariableOp#list_net_model_144/dense_876/kernel*
_output_shapes

: *
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1#list_net_model_144/dense_876/kernel!list_net_model_144/dense_876/bias#list_net_model_144/dense_877/kernel!list_net_model_144/dense_877/bias#list_net_model_144/dense_878/kernel!list_net_model_144/dense_878/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2416755

NoOpNoOp
�3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�2
value�2B�2 B�2
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
first_layer
	first_dropout_layer

second_layer
second_dropout_layer
output_layer
	optimizer

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_random_generator* 
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator* 
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

kernel
bias*
�
>
_variables
?_iterations
@_learning_rate
A_index_dict
B
_momentums
C_velocities
D_update_step_xla*

Eserving_default* 
c]
VARIABLE_VALUE#list_net_model_144/dense_876/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!list_net_model_144/dense_876/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#list_net_model_144/dense_877/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!list_net_model_144/dense_877/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#list_net_model_144/dense_878/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!list_net_model_144/dense_878/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
	1

2
3
4*

F0*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 
* 
* 
* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

Strace_0
Ttrace_1* 

Utrace_0
Vtrace_1* 
* 

0
1*

0
1*
* 
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

\trace_0* 

]trace_0* 
* 
* 
* 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 

ctrace_0
dtrace_1* 

etrace_0
ftrace_1* 
* 

0
1*

0
1*
* 
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

ltrace_0* 

mtrace_0* 
b
?0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
n0
p1
r2
t3
v4
x5*
.
o0
q1
s2
u3
w4
y5*
* 
* 
8
z	variables
{	keras_api
	|total
	}count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
uo
VARIABLE_VALUE*Adam/m/list_net_model_144/dense_876/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/list_net_model_144/dense_876/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/list_net_model_144/dense_876/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/list_net_model_144/dense_876/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/list_net_model_144/dense_877/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/list_net_model_144/dense_877/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/list_net_model_144/dense_877/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/list_net_model_144/dense_877/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/list_net_model_144/dense_878/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/list_net_model_144/dense_878/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/list_net_model_144/dense_878/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/list_net_model_144/dense_878/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*

|0
}1*

z	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#list_net_model_144/dense_876/kernel!list_net_model_144/dense_876/bias#list_net_model_144/dense_877/kernel!list_net_model_144/dense_877/bias#list_net_model_144/dense_878/kernel!list_net_model_144/dense_878/bias	iterationlearning_rate*Adam/m/list_net_model_144/dense_876/kernel*Adam/v/list_net_model_144/dense_876/kernel(Adam/m/list_net_model_144/dense_876/bias(Adam/v/list_net_model_144/dense_876/bias*Adam/m/list_net_model_144/dense_877/kernel*Adam/v/list_net_model_144/dense_877/kernel(Adam/m/list_net_model_144/dense_877/bias(Adam/v/list_net_model_144/dense_877/bias*Adam/m/list_net_model_144/dense_878/kernel*Adam/v/list_net_model_144/dense_878/kernel(Adam/m/list_net_model_144/dense_878/bias(Adam/v/list_net_model_144/dense_878/biastotalcountConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2417022
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#list_net_model_144/dense_876/kernel!list_net_model_144/dense_876/bias#list_net_model_144/dense_877/kernel!list_net_model_144/dense_877/bias#list_net_model_144/dense_878/kernel!list_net_model_144/dense_878/bias	iterationlearning_rate*Adam/m/list_net_model_144/dense_876/kernel*Adam/v/list_net_model_144/dense_876/kernel(Adam/m/list_net_model_144/dense_876/bias(Adam/v/list_net_model_144/dense_876/bias*Adam/m/list_net_model_144/dense_877/kernel*Adam/v/list_net_model_144/dense_877/kernel(Adam/m/list_net_model_144/dense_877/bias(Adam/v/list_net_model_144/dense_877/bias*Adam/m/list_net_model_144/dense_878/kernel*Adam/v/list_net_model_144/dense_878/kernel(Adam/m/list_net_model_144/dense_878/bias(Adam/v/list_net_model_144/dense_878/biastotalcount*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2417097��
�

�
%__inference_signature_wrapper_2416755
input_1
unknown: 
	unknown_0: 
	unknown_1:	 �
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2416546o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2416751:'#
!
_user_specified_name	2416749:'#
!
_user_specified_name	2416747:'#
!
_user_specified_name	2416745:'#
!
_user_specified_name	2416743:'#
!
_user_specified_name	2416741:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

g
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416605

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_878_layer_call_and_return_conditional_losses_2416616

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_585_layer_call_fn_2416827

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416605p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_878_layer_call_fn_2416858

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_2416616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2416854:'#
!
_user_specified_name	2416852:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_584_layer_call_fn_2416780

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416576o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
4__inference_list_net_model_144_layer_call_fn_2416688
input_1
unknown: 
	unknown_0: 
	unknown_1:	 �
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416654o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2416684:'#
!
_user_specified_name	2416682:'#
!
_user_specified_name	2416680:'#
!
_user_specified_name	2416678:'#
!
_user_specified_name	2416676:'#
!
_user_specified_name	2416674:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

g
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416797

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

g
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416576

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�o
�
#__inference__traced_restore_2417097
file_prefixF
4assignvariableop_list_net_model_144_dense_876_kernel: B
4assignvariableop_1_list_net_model_144_dense_876_bias: I
6assignvariableop_2_list_net_model_144_dense_877_kernel:	 �C
4assignvariableop_3_list_net_model_144_dense_877_bias:	�I
6assignvariableop_4_list_net_model_144_dense_878_kernel:	�B
4assignvariableop_5_list_net_model_144_dense_878_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: O
=assignvariableop_8_adam_m_list_net_model_144_dense_876_kernel: O
=assignvariableop_9_adam_v_list_net_model_144_dense_876_kernel: J
<assignvariableop_10_adam_m_list_net_model_144_dense_876_bias: J
<assignvariableop_11_adam_v_list_net_model_144_dense_876_bias: Q
>assignvariableop_12_adam_m_list_net_model_144_dense_877_kernel:	 �Q
>assignvariableop_13_adam_v_list_net_model_144_dense_877_kernel:	 �K
<assignvariableop_14_adam_m_list_net_model_144_dense_877_bias:	�K
<assignvariableop_15_adam_v_list_net_model_144_dense_877_bias:	�Q
>assignvariableop_16_adam_m_list_net_model_144_dense_878_kernel:	�Q
>assignvariableop_17_adam_v_list_net_model_144_dense_878_kernel:	�J
<assignvariableop_18_adam_m_list_net_model_144_dense_878_bias:J
<assignvariableop_19_adam_v_list_net_model_144_dense_878_bias:#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp4assignvariableop_list_net_model_144_dense_876_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp4assignvariableop_1_list_net_model_144_dense_876_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp6assignvariableop_2_list_net_model_144_dense_877_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp4assignvariableop_3_list_net_model_144_dense_877_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_list_net_model_144_dense_878_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp4assignvariableop_5_list_net_model_144_dense_878_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp=assignvariableop_8_adam_m_list_net_model_144_dense_876_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp=assignvariableop_9_adam_v_list_net_model_144_dense_876_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp<assignvariableop_10_adam_m_list_net_model_144_dense_876_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp<assignvariableop_11_adam_v_list_net_model_144_dense_876_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp>assignvariableop_12_adam_m_list_net_model_144_dense_877_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp>assignvariableop_13_adam_v_list_net_model_144_dense_877_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp<assignvariableop_14_adam_m_list_net_model_144_dense_877_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp<assignvariableop_15_adam_v_list_net_model_144_dense_877_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp>assignvariableop_16_adam_m_list_net_model_144_dense_878_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp>assignvariableop_17_adam_v_list_net_model_144_dense_878_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp<assignvariableop_18_adam_m_list_net_model_144_dense_878_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp<assignvariableop_19_adam_v_list_net_model_144_dense_878_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:HD
B
_user_specified_name*(Adam/v/list_net_model_144/dense_878/bias:HD
B
_user_specified_name*(Adam/m/list_net_model_144/dense_878/bias:JF
D
_user_specified_name,*Adam/v/list_net_model_144/dense_878/kernel:JF
D
_user_specified_name,*Adam/m/list_net_model_144/dense_878/kernel:HD
B
_user_specified_name*(Adam/v/list_net_model_144/dense_877/bias:HD
B
_user_specified_name*(Adam/m/list_net_model_144/dense_877/bias:JF
D
_user_specified_name,*Adam/v/list_net_model_144/dense_877/kernel:JF
D
_user_specified_name,*Adam/m/list_net_model_144/dense_877/kernel:HD
B
_user_specified_name*(Adam/v/list_net_model_144/dense_876/bias:HD
B
_user_specified_name*(Adam/m/list_net_model_144/dense_876/bias:J
F
D
_user_specified_name,*Adam/v/list_net_model_144/dense_876/kernel:J	F
D
_user_specified_name,*Adam/m/list_net_model_144/dense_876/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:A=
;
_user_specified_name#!list_net_model_144/dense_878/bias:C?
=
_user_specified_name%#list_net_model_144/dense_878/kernel:A=
;
_user_specified_name#!list_net_model_144/dense_877/bias:C?
=
_user_specified_name%#list_net_model_144/dense_877/kernel:A=
;
_user_specified_name#!list_net_model_144/dense_876/bias:C?
=
_user_specified_name%#list_net_model_144/dense_876/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
+__inference_dense_877_layer_call_fn_2416811

inputs
unknown:	 �
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_877_layer_call_and_return_conditional_losses_2416588p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2416807:'#
!
_user_specified_name	2416805:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_876_layer_call_and_return_conditional_losses_2416559

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_dense_876_layer_call_and_return_conditional_losses_2416775

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_dropout_585_layer_call_fn_2416832

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416646a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_878_layer_call_and_return_conditional_losses_2416868

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416646

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_877_layer_call_and_return_conditional_losses_2416822

inputs1
matmul_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_dense_876_layer_call_fn_2416764

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_876_layer_call_and_return_conditional_losses_2416559o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2416760:'#
!
_user_specified_name	2416758:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
Ը
�
 __inference__traced_save_2417022
file_prefixL
:read_disablecopyonread_list_net_model_144_dense_876_kernel: H
:read_1_disablecopyonread_list_net_model_144_dense_876_bias: O
<read_2_disablecopyonread_list_net_model_144_dense_877_kernel:	 �I
:read_3_disablecopyonread_list_net_model_144_dense_877_bias:	�O
<read_4_disablecopyonread_list_net_model_144_dense_878_kernel:	�H
:read_5_disablecopyonread_list_net_model_144_dense_878_bias:,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: U
Cread_8_disablecopyonread_adam_m_list_net_model_144_dense_876_kernel: U
Cread_9_disablecopyonread_adam_v_list_net_model_144_dense_876_kernel: P
Bread_10_disablecopyonread_adam_m_list_net_model_144_dense_876_bias: P
Bread_11_disablecopyonread_adam_v_list_net_model_144_dense_876_bias: W
Dread_12_disablecopyonread_adam_m_list_net_model_144_dense_877_kernel:	 �W
Dread_13_disablecopyonread_adam_v_list_net_model_144_dense_877_kernel:	 �Q
Bread_14_disablecopyonread_adam_m_list_net_model_144_dense_877_bias:	�Q
Bread_15_disablecopyonread_adam_v_list_net_model_144_dense_877_bias:	�W
Dread_16_disablecopyonread_adam_m_list_net_model_144_dense_878_kernel:	�W
Dread_17_disablecopyonread_adam_v_list_net_model_144_dense_878_kernel:	�P
Bread_18_disablecopyonread_adam_m_list_net_model_144_dense_878_bias:P
Bread_19_disablecopyonread_adam_v_list_net_model_144_dense_878_bias:)
read_20_disablecopyonread_total: )
read_21_disablecopyonread_count: 
savev2_const
identity_45��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead:read_disablecopyonread_list_net_model_144_dense_876_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp:read_disablecopyonread_list_net_model_144_dense_876_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_1/DisableCopyOnReadDisableCopyOnRead:read_1_disablecopyonread_list_net_model_144_dense_876_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp:read_1_disablecopyonread_list_net_model_144_dense_876_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnRead<read_2_disablecopyonread_list_net_model_144_dense_877_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp<read_2_disablecopyonread_list_net_model_144_dense_877_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ��
Read_3/DisableCopyOnReadDisableCopyOnRead:read_3_disablecopyonread_list_net_model_144_dense_877_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp:read_3_disablecopyonread_list_net_model_144_dense_877_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnRead<read_4_disablecopyonread_list_net_model_144_dense_878_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp<read_4_disablecopyonread_list_net_model_144_dense_878_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_5/DisableCopyOnReadDisableCopyOnRead:read_5_disablecopyonread_list_net_model_144_dense_878_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp:read_5_disablecopyonread_list_net_model_144_dense_878_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnReadCread_8_disablecopyonread_adam_m_list_net_model_144_dense_876_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpCread_8_disablecopyonread_adam_m_list_net_model_144_dense_876_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_9/DisableCopyOnReadDisableCopyOnReadCread_9_disablecopyonread_adam_v_list_net_model_144_dense_876_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpCread_9_disablecopyonread_adam_v_list_net_model_144_dense_876_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_10/DisableCopyOnReadDisableCopyOnReadBread_10_disablecopyonread_adam_m_list_net_model_144_dense_876_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpBread_10_disablecopyonread_adam_m_list_net_model_144_dense_876_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnReadBread_11_disablecopyonread_adam_v_list_net_model_144_dense_876_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpBread_11_disablecopyonread_adam_v_list_net_model_144_dense_876_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnReadDread_12_disablecopyonread_adam_m_list_net_model_144_dense_877_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpDread_12_disablecopyonread_adam_m_list_net_model_144_dense_877_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ��
Read_13/DisableCopyOnReadDisableCopyOnReadDread_13_disablecopyonread_adam_v_list_net_model_144_dense_877_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpDread_13_disablecopyonread_adam_v_list_net_model_144_dense_877_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	 ��
Read_14/DisableCopyOnReadDisableCopyOnReadBread_14_disablecopyonread_adam_m_list_net_model_144_dense_877_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpBread_14_disablecopyonread_adam_m_list_net_model_144_dense_877_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnReadBread_15_disablecopyonread_adam_v_list_net_model_144_dense_877_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpBread_15_disablecopyonread_adam_v_list_net_model_144_dense_877_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnReadDread_16_disablecopyonread_adam_m_list_net_model_144_dense_878_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpDread_16_disablecopyonread_adam_m_list_net_model_144_dense_878_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_17/DisableCopyOnReadDisableCopyOnReadDread_17_disablecopyonread_adam_v_list_net_model_144_dense_878_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpDread_17_disablecopyonread_adam_v_list_net_model_144_dense_878_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_18/DisableCopyOnReadDisableCopyOnReadBread_18_disablecopyonread_adam_m_list_net_model_144_dense_878_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpBread_18_disablecopyonread_adam_m_list_net_model_144_dense_878_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnReadBread_19_disablecopyonread_adam_v_list_net_model_144_dense_878_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpBread_19_disablecopyonread_adam_v_list_net_model_144_dense_878_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_20/DisableCopyOnReadDisableCopyOnReadread_20_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpread_20_disablecopyonread_total^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_21/DisableCopyOnReadDisableCopyOnReadread_21_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpread_21_disablecopyonread_count^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *%
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_44Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_45IdentityIdentity_44:output:0^NoOp*
T0*
_output_shapes
: �	
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_45Identity_45:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:HD
B
_user_specified_name*(Adam/v/list_net_model_144/dense_878/bias:HD
B
_user_specified_name*(Adam/m/list_net_model_144/dense_878/bias:JF
D
_user_specified_name,*Adam/v/list_net_model_144/dense_878/kernel:JF
D
_user_specified_name,*Adam/m/list_net_model_144/dense_878/kernel:HD
B
_user_specified_name*(Adam/v/list_net_model_144/dense_877/bias:HD
B
_user_specified_name*(Adam/m/list_net_model_144/dense_877/bias:JF
D
_user_specified_name,*Adam/v/list_net_model_144/dense_877/kernel:JF
D
_user_specified_name,*Adam/m/list_net_model_144/dense_877/kernel:HD
B
_user_specified_name*(Adam/v/list_net_model_144/dense_876/bias:HD
B
_user_specified_name*(Adam/m/list_net_model_144/dense_876/bias:J
F
D
_user_specified_name,*Adam/v/list_net_model_144/dense_876/kernel:J	F
D
_user_specified_name,*Adam/m/list_net_model_144/dense_876/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:A=
;
_user_specified_name#!list_net_model_144/dense_878/bias:C?
=
_user_specified_name%#list_net_model_144/dense_878/kernel:A=
;
_user_specified_name#!list_net_model_144/dense_877/bias:C?
=
_user_specified_name%#list_net_model_144/dense_877/kernel:A=
;
_user_specified_name#!list_net_model_144/dense_876/bias:C?
=
_user_specified_name%#list_net_model_144/dense_876/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�)
�
"__inference__wrapped_model_2416546
input_1M
;list_net_model_144_dense_876_matmul_readvariableop_resource: J
<list_net_model_144_dense_876_biasadd_readvariableop_resource: N
;list_net_model_144_dense_877_matmul_readvariableop_resource:	 �K
<list_net_model_144_dense_877_biasadd_readvariableop_resource:	�N
;list_net_model_144_dense_878_matmul_readvariableop_resource:	�J
<list_net_model_144_dense_878_biasadd_readvariableop_resource:
identity��3list_net_model_144/dense_876/BiasAdd/ReadVariableOp�2list_net_model_144/dense_876/MatMul/ReadVariableOp�3list_net_model_144/dense_877/BiasAdd/ReadVariableOp�2list_net_model_144/dense_877/MatMul/ReadVariableOp�3list_net_model_144/dense_878/BiasAdd/ReadVariableOp�2list_net_model_144/dense_878/MatMul/ReadVariableOp�
2list_net_model_144/dense_876/MatMul/ReadVariableOpReadVariableOp;list_net_model_144_dense_876_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
#list_net_model_144/dense_876/MatMulMatMulinput_1:list_net_model_144/dense_876/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
3list_net_model_144/dense_876/BiasAdd/ReadVariableOpReadVariableOp<list_net_model_144_dense_876_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
$list_net_model_144/dense_876/BiasAddBiasAdd-list_net_model_144/dense_876/MatMul:product:0;list_net_model_144/dense_876/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!list_net_model_144/dense_876/ReluRelu-list_net_model_144/dense_876/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'list_net_model_144/dropout_584/IdentityIdentity/list_net_model_144/dense_876/Relu:activations:0*
T0*'
_output_shapes
:��������� �
2list_net_model_144/dense_877/MatMul/ReadVariableOpReadVariableOp;list_net_model_144_dense_877_matmul_readvariableop_resource*
_output_shapes
:	 �*
dtype0�
#list_net_model_144/dense_877/MatMulMatMul0list_net_model_144/dropout_584/Identity:output:0:list_net_model_144/dense_877/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3list_net_model_144/dense_877/BiasAdd/ReadVariableOpReadVariableOp<list_net_model_144_dense_877_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$list_net_model_144/dense_877/BiasAddBiasAdd-list_net_model_144/dense_877/MatMul:product:0;list_net_model_144/dense_877/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!list_net_model_144/dense_877/ReluRelu-list_net_model_144/dense_877/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'list_net_model_144/dropout_585/IdentityIdentity/list_net_model_144/dense_877/Relu:activations:0*
T0*(
_output_shapes
:�����������
2list_net_model_144/dense_878/MatMul/ReadVariableOpReadVariableOp;list_net_model_144_dense_878_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
#list_net_model_144/dense_878/MatMulMatMul0list_net_model_144/dropout_585/Identity:output:0:list_net_model_144/dense_878/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
3list_net_model_144/dense_878/BiasAdd/ReadVariableOpReadVariableOp<list_net_model_144_dense_878_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
$list_net_model_144/dense_878/BiasAddBiasAdd-list_net_model_144/dense_878/MatMul:product:0;list_net_model_144/dense_878/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
IdentityIdentity-list_net_model_144/dense_878/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp4^list_net_model_144/dense_876/BiasAdd/ReadVariableOp3^list_net_model_144/dense_876/MatMul/ReadVariableOp4^list_net_model_144/dense_877/BiasAdd/ReadVariableOp3^list_net_model_144/dense_877/MatMul/ReadVariableOp4^list_net_model_144/dense_878/BiasAdd/ReadVariableOp3^list_net_model_144/dense_878/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2j
3list_net_model_144/dense_876/BiasAdd/ReadVariableOp3list_net_model_144/dense_876/BiasAdd/ReadVariableOp2h
2list_net_model_144/dense_876/MatMul/ReadVariableOp2list_net_model_144/dense_876/MatMul/ReadVariableOp2j
3list_net_model_144/dense_877/BiasAdd/ReadVariableOp3list_net_model_144/dense_877/BiasAdd/ReadVariableOp2h
2list_net_model_144/dense_877/MatMul/ReadVariableOp2list_net_model_144/dense_877/MatMul/ReadVariableOp2j
3list_net_model_144/dense_878/BiasAdd/ReadVariableOp3list_net_model_144/dense_878/BiasAdd/ReadVariableOp2h
2list_net_model_144/dense_878/MatMul/ReadVariableOp2list_net_model_144/dense_878/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
4__inference_list_net_model_144_layer_call_fn_2416671
input_1
unknown: 
	unknown_0: 
	unknown_1:	 �
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416623o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2416667:'#
!
_user_specified_name	2416665:'#
!
_user_specified_name	2416663:'#
!
_user_specified_name	2416661:'#
!
_user_specified_name	2416659:'#
!
_user_specified_name	2416657:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
f
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416849

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416654
input_1#
dense_876_2416626: 
dense_876_2416628: $
dense_877_2416637:	 � 
dense_877_2416639:	�$
dense_878_2416648:	�
dense_878_2416650:
identity��!dense_876/StatefulPartitionedCall�!dense_877/StatefulPartitionedCall�!dense_878/StatefulPartitionedCall�
!dense_876/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_876_2416626dense_876_2416628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_876_layer_call_and_return_conditional_losses_2416559�
dropout_584/PartitionedCallPartitionedCall*dense_876/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416635�
!dense_877/StatefulPartitionedCallStatefulPartitionedCall$dropout_584/PartitionedCall:output:0dense_877_2416637dense_877_2416639*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_877_layer_call_and_return_conditional_losses_2416588�
dropout_585/PartitionedCallPartitionedCall*dense_877/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416646�
!dense_878/StatefulPartitionedCallStatefulPartitionedCall$dropout_585/PartitionedCall:output:0dense_878_2416648dense_878_2416650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_2416616y
IdentityIdentity*dense_878/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_876/StatefulPartitionedCall"^dense_877/StatefulPartitionedCall"^dense_878/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2F
!dense_876/StatefulPartitionedCall!dense_876/StatefulPartitionedCall2F
!dense_877/StatefulPartitionedCall!dense_877/StatefulPartitionedCall2F
!dense_878/StatefulPartitionedCall!dense_878/StatefulPartitionedCall:'#
!
_user_specified_name	2416650:'#
!
_user_specified_name	2416648:'#
!
_user_specified_name	2416639:'#
!
_user_specified_name	2416637:'#
!
_user_specified_name	2416628:'#
!
_user_specified_name	2416626:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
f
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416802

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
F__inference_dense_877_layer_call_and_return_conditional_losses_2416588

inputs1
matmul_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
I
-__inference_dropout_584_layer_call_fn_2416785

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416635`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

g
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416844

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416635

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416623
input_1#
dense_876_2416560: 
dense_876_2416562: $
dense_877_2416589:	 � 
dense_877_2416591:	�$
dense_878_2416617:	�
dense_878_2416619:
identity��!dense_876/StatefulPartitionedCall�!dense_877/StatefulPartitionedCall�!dense_878/StatefulPartitionedCall�#dropout_584/StatefulPartitionedCall�#dropout_585/StatefulPartitionedCall�
!dense_876/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_876_2416560dense_876_2416562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_876_layer_call_and_return_conditional_losses_2416559�
#dropout_584/StatefulPartitionedCallStatefulPartitionedCall*dense_876/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416576�
!dense_877/StatefulPartitionedCallStatefulPartitionedCall,dropout_584/StatefulPartitionedCall:output:0dense_877_2416589dense_877_2416591*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_877_layer_call_and_return_conditional_losses_2416588�
#dropout_585/StatefulPartitionedCallStatefulPartitionedCall*dense_877/StatefulPartitionedCall:output:0$^dropout_584/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416605�
!dense_878/StatefulPartitionedCallStatefulPartitionedCall,dropout_585/StatefulPartitionedCall:output:0dense_878_2416617dense_878_2416619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_878_layer_call_and_return_conditional_losses_2416616y
IdentityIdentity*dense_878/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_876/StatefulPartitionedCall"^dense_877/StatefulPartitionedCall"^dense_878/StatefulPartitionedCall$^dropout_584/StatefulPartitionedCall$^dropout_585/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2F
!dense_876/StatefulPartitionedCall!dense_876/StatefulPartitionedCall2F
!dense_877/StatefulPartitionedCall!dense_877/StatefulPartitionedCall2F
!dense_878/StatefulPartitionedCall!dense_878/StatefulPartitionedCall2J
#dropout_584/StatefulPartitionedCall#dropout_584/StatefulPartitionedCall2J
#dropout_585/StatefulPartitionedCall#dropout_585/StatefulPartitionedCall:'#
!
_user_specified_name	2416619:'#
!
_user_specified_name	2416617:'#
!
_user_specified_name	2416591:'#
!
_user_specified_name	2416589:'#
!
_user_specified_name	2416562:'#
!
_user_specified_name	2416560:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
first_layer
	first_dropout_layer

second_layer
second_dropout_layer
output_layer
	optimizer

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
4__inference_list_net_model_144_layer_call_fn_2416671
4__inference_list_net_model_144_layer_call_fn_2416688�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416623
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416654�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1
�B�
"__inference__wrapped_model_2416546input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_random_generator"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
>
_variables
?_iterations
@_learning_rate
A_index_dict
B
_momentums
C_velocities
D_update_step_xla"
experimentalOptimizer
,
Eserving_default"
signature_map
5:3 2#list_net_model_144/dense_876/kernel
/:- 2!list_net_model_144/dense_876/bias
6:4	 �2#list_net_model_144/dense_877/kernel
0:.�2!list_net_model_144/dense_877/bias
6:4	�2#list_net_model_144/dense_878/kernel
/:-2!list_net_model_144/dense_878/bias
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
'
F0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_list_net_model_144_layer_call_fn_2416671input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
4__inference_list_net_model_144_layer_call_fn_2416688input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416623input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
�B�
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416654input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_02�
+__inference_dense_876_layer_call_fn_2416764�
���
FullArgSpec
args�

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
annotations� *
 zLtrace_0
�
Mtrace_02�
F__inference_dense_876_layer_call_and_return_conditional_losses_2416775�
���
FullArgSpec
args�

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
annotations� *
 zMtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
Strace_0
Ttrace_12�
-__inference_dropout_584_layer_call_fn_2416780
-__inference_dropout_584_layer_call_fn_2416785�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0zTtrace_1
�
Utrace_0
Vtrace_12�
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416797
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416802�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0zVtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
\trace_02�
+__inference_dense_877_layer_call_fn_2416811�
���
FullArgSpec
args�

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
annotations� *
 z\trace_0
�
]trace_02�
F__inference_dense_877_layer_call_and_return_conditional_losses_2416822�
���
FullArgSpec
args�

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
annotations� *
 z]trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
ctrace_0
dtrace_12�
-__inference_dropout_585_layer_call_fn_2416827
-__inference_dropout_585_layer_call_fn_2416832�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0zdtrace_1
�
etrace_0
ftrace_12�
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416844
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416849�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zetrace_0zftrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
ltrace_02�
+__inference_dense_878_layer_call_fn_2416858�
���
FullArgSpec
args�

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
annotations� *
 zltrace_0
�
mtrace_02�
F__inference_dense_878_layer_call_and_return_conditional_losses_2416868�
���
FullArgSpec
args�

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
annotations� *
 zmtrace_0
~
?0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
n0
p1
r2
t3
v4
x5"
trackable_list_wrapper
J
o0
q1
s2
u3
w4
y5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_2416755input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
N
z	variables
{	keras_api
	|total
	}count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_876_layer_call_fn_2416764inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_dense_876_layer_call_and_return_conditional_losses_2416775inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dropout_584_layer_call_fn_2416780inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_584_layer_call_fn_2416785inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416797inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416802inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_877_layer_call_fn_2416811inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_dense_877_layer_call_and_return_conditional_losses_2416822inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dropout_585_layer_call_fn_2416827inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_585_layer_call_fn_2416832inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416844inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416849inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_878_layer_call_fn_2416858inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_dense_878_layer_call_and_return_conditional_losses_2416868inputs"�
���
FullArgSpec
args�

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
annotations� *
 
::8 2*Adam/m/list_net_model_144/dense_876/kernel
::8 2*Adam/v/list_net_model_144/dense_876/kernel
4:2 2(Adam/m/list_net_model_144/dense_876/bias
4:2 2(Adam/v/list_net_model_144/dense_876/bias
;:9	 �2*Adam/m/list_net_model_144/dense_877/kernel
;:9	 �2*Adam/v/list_net_model_144/dense_877/kernel
5:3�2(Adam/m/list_net_model_144/dense_877/bias
5:3�2(Adam/v/list_net_model_144/dense_877/bias
;:9	�2*Adam/m/list_net_model_144/dense_878/kernel
;:9	�2*Adam/v/list_net_model_144/dense_878/kernel
4:22(Adam/m/list_net_model_144/dense_878/bias
4:22(Adam/v/list_net_model_144/dense_878/bias
.
|0
}1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count�
"__inference__wrapped_model_2416546o0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
F__inference_dense_876_layer_call_and_return_conditional_losses_2416775c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0��������� 
� �
+__inference_dense_876_layer_call_fn_2416764X/�,
%�"
 �
inputs���������
� "!�
unknown��������� �
F__inference_dense_877_layer_call_and_return_conditional_losses_2416822d/�,
%�"
 �
inputs��������� 
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_877_layer_call_fn_2416811Y/�,
%�"
 �
inputs��������� 
� ""�
unknown�����������
F__inference_dense_878_layer_call_and_return_conditional_losses_2416868d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_878_layer_call_fn_2416858Y0�-
&�#
!�
inputs����������
� "!�
unknown����������
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416797c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
H__inference_dropout_584_layer_call_and_return_conditional_losses_2416802c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
-__inference_dropout_584_layer_call_fn_2416780X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
-__inference_dropout_584_layer_call_fn_2416785X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416844e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
H__inference_dropout_585_layer_call_and_return_conditional_losses_2416849e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
-__inference_dropout_585_layer_call_fn_2416827Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
-__inference_dropout_585_layer_call_fn_2416832Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416623x@�=
&�#
!�
input_1���������
�

trainingp",�)
"�
tensor_0���������
� �
O__inference_list_net_model_144_layer_call_and_return_conditional_losses_2416654x@�=
&�#
!�
input_1���������
�

trainingp ",�)
"�
tensor_0���������
� �
4__inference_list_net_model_144_layer_call_fn_2416671m@�=
&�#
!�
input_1���������
�

trainingp"!�
unknown����������
4__inference_list_net_model_144_layer_call_fn_2416688m@�=
&�#
!�
input_1���������
�

trainingp "!�
unknown����������
%__inference_signature_wrapper_2416755z;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������