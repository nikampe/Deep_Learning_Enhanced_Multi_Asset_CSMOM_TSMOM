��#
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
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
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
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
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��!
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
5Adam/v/csmom_transformer_reranker_117/dense_2782/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/v/csmom_transformer_reranker_117/dense_2782/bias
�
IAdam/v/csmom_transformer_reranker_117/dense_2782/bias/Read/ReadVariableOpReadVariableOp5Adam/v/csmom_transformer_reranker_117/dense_2782/bias*
_output_shapes
:*
dtype0
�
5Adam/m/csmom_transformer_reranker_117/dense_2782/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/m/csmom_transformer_reranker_117/dense_2782/bias
�
IAdam/m/csmom_transformer_reranker_117/dense_2782/bias/Read/ReadVariableOpReadVariableOp5Adam/m/csmom_transformer_reranker_117/dense_2782/bias*
_output_shapes
:*
dtype0
�
7Adam/v/csmom_transformer_reranker_117/dense_2782/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*H
shared_name97Adam/v/csmom_transformer_reranker_117/dense_2782/kernel
�
KAdam/v/csmom_transformer_reranker_117/dense_2782/kernel/Read/ReadVariableOpReadVariableOp7Adam/v/csmom_transformer_reranker_117/dense_2782/kernel*
_output_shapes
:	�*
dtype0
�
7Adam/m/csmom_transformer_reranker_117/dense_2782/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*H
shared_name97Adam/m/csmom_transformer_reranker_117/dense_2782/kernel
�
KAdam/m/csmom_transformer_reranker_117/dense_2782/kernel/Read/ReadVariableOpReadVariableOp7Adam/m/csmom_transformer_reranker_117/dense_2782/kernel*
_output_shapes
:	�*
dtype0
�
TAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*e
shared_nameVTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta
�
hAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta/Read/ReadVariableOpReadVariableOpTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta*
_output_shapes	
:�*
dtype0
�
TAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*e
shared_nameVTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta
�
hAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta/Read/ReadVariableOpReadVariableOpTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta*
_output_shapes	
:�*
dtype0
�
UAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*f
shared_nameWUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma
�
iAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma/Read/ReadVariableOpReadVariableOpUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma*
_output_shapes	
:�*
dtype0
�
UAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*f
shared_nameWUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma
�
iAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma/Read/ReadVariableOpReadVariableOpUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma*
_output_shapes	
:�*
dtype0
�
TAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*e
shared_nameVTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta
�
hAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta/Read/ReadVariableOpReadVariableOpTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta*
_output_shapes	
:�*
dtype0
�
TAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*e
shared_nameVTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta
�
hAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta/Read/ReadVariableOpReadVariableOpTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta*
_output_shapes	
:�*
dtype0
�
UAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*f
shared_nameWUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma
�
iAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma/Read/ReadVariableOpReadVariableOpUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma*
_output_shapes	
:�*
dtype0
�
UAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*f
shared_nameWUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma
�
iAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma/Read/ReadVariableOpReadVariableOpUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_2781/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/v/dense_2781/bias
~
*Adam/v/dense_2781/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2781/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_2781/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/m/dense_2781/bias
~
*Adam/m/dense_2781/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2781/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_2781/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/v/dense_2781/kernel
�
,Adam/v/dense_2781/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2781/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_2781/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/m/dense_2781/kernel
�
,Adam/m/dense_2781/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2781/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_2780/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_2780/bias
}
*Adam/v/dense_2780/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2780/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_2780/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_2780/bias
}
*Adam/m/dense_2780/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2780/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_2780/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/v/dense_2780/kernel
�
,Adam/v/dense_2780/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2780/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_2780/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/m/dense_2780/kernel
�
,Adam/m/dense_2780/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2780/kernel*
_output_shapes
:	�*
dtype0
�
eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias
�
yAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias/Read/ReadVariableOpReadVariableOpeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias*
_output_shapes	
:�*
dtype0
�
eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias
�
yAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias/Read/ReadVariableOpReadVariableOpeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias*
_output_shapes	
:�*
dtype0
�
gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel
�
{Adam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel/Read/ReadVariableOpReadVariableOpgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel* 
_output_shapes
:
��*
dtype0
�
gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel
�
{Adam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel/Read/ReadVariableOpReadVariableOpgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel* 
_output_shapes
:
��*
dtype0
�
eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias
�
yAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias/Read/ReadVariableOpReadVariableOpeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias*
_output_shapes	
:�*
dtype0
�
eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias
�
yAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias/Read/ReadVariableOpReadVariableOpeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias*
_output_shapes	
:�*
dtype0
�
gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel
�
{Adam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel/Read/ReadVariableOpReadVariableOpgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel* 
_output_shapes
:
��*
dtype0
�
gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel
�
{Adam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel/Read/ReadVariableOpReadVariableOpgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel* 
_output_shapes
:
��*
dtype0
�
eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias
�
yAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias/Read/ReadVariableOpReadVariableOpeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias*
_output_shapes	
:�*
dtype0
�
eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias
�
yAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias/Read/ReadVariableOpReadVariableOpeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias*
_output_shapes	
:�*
dtype0
�
gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel
�
{Adam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel/Read/ReadVariableOpReadVariableOpgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel* 
_output_shapes
:
��*
dtype0
�
gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel
�
{Adam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel/Read/ReadVariableOpReadVariableOpgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel* 
_output_shapes
:
��*
dtype0
�
eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias
�
yAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias/Read/ReadVariableOpReadVariableOpeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias*
_output_shapes	
:�*
dtype0
�
eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*v
shared_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias
�
yAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias/Read/ReadVariableOpReadVariableOpeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias*
_output_shapes	
:�*
dtype0
�
gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel
�
{Adam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel/Read/ReadVariableOpReadVariableOpgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel* 
_output_shapes
:
��*
dtype0
�
gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*x
shared_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel
�
{Adam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel/Read/ReadVariableOpReadVariableOpgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel* 
_output_shapes
:
��*
dtype0
�
5Adam/v/csmom_transformer_reranker_117/dense_2775/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75Adam/v/csmom_transformer_reranker_117/dense_2775/bias
�
IAdam/v/csmom_transformer_reranker_117/dense_2775/bias/Read/ReadVariableOpReadVariableOp5Adam/v/csmom_transformer_reranker_117/dense_2775/bias*
_output_shapes	
:�*
dtype0
�
5Adam/m/csmom_transformer_reranker_117/dense_2775/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75Adam/m/csmom_transformer_reranker_117/dense_2775/bias
�
IAdam/m/csmom_transformer_reranker_117/dense_2775/bias/Read/ReadVariableOpReadVariableOp5Adam/m/csmom_transformer_reranker_117/dense_2775/bias*
_output_shapes	
:�*
dtype0
�
7Adam/v/csmom_transformer_reranker_117/dense_2775/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*H
shared_name97Adam/v/csmom_transformer_reranker_117/dense_2775/kernel
�
KAdam/v/csmom_transformer_reranker_117/dense_2775/kernel/Read/ReadVariableOpReadVariableOp7Adam/v/csmom_transformer_reranker_117/dense_2775/kernel*
_output_shapes
:	�*
dtype0
�
7Adam/m/csmom_transformer_reranker_117/dense_2775/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*H
shared_name97Adam/m/csmom_transformer_reranker_117/dense_2775/kernel
�
KAdam/m/csmom_transformer_reranker_117/dense_2775/kernel/Read/ReadVariableOpReadVariableOp7Adam/m/csmom_transformer_reranker_117/dense_2775/kernel*
_output_shapes
:	�*
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
.csmom_transformer_reranker_117/dense_2782/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.csmom_transformer_reranker_117/dense_2782/bias
�
Bcsmom_transformer_reranker_117/dense_2782/bias/Read/ReadVariableOpReadVariableOp.csmom_transformer_reranker_117/dense_2782/bias*
_output_shapes
:*
dtype0
�
0csmom_transformer_reranker_117/dense_2782/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*A
shared_name20csmom_transformer_reranker_117/dense_2782/kernel
�
Dcsmom_transformer_reranker_117/dense_2782/kernel/Read/ReadVariableOpReadVariableOp0csmom_transformer_reranker_117/dense_2782/kernel*
_output_shapes
:	�*
dtype0
�
Mcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*^
shared_nameOMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta
�
acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta/Read/ReadVariableOpReadVariableOpMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta*
_output_shapes	
:�*
dtype0
�
Ncsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*_
shared_namePNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma
�
bcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma/Read/ReadVariableOpReadVariableOpNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma*
_output_shapes	
:�*
dtype0
�
Mcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*^
shared_nameOMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta
�
acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta/Read/ReadVariableOpReadVariableOpMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta*
_output_shapes	
:�*
dtype0
�
Ncsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*_
shared_namePNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma
�
bcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma/Read/ReadVariableOpReadVariableOpNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma*
_output_shapes	
:�*
dtype0
w
dense_2781/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namedense_2781/bias
p
#dense_2781/bias/Read/ReadVariableOpReadVariableOpdense_2781/bias*
_output_shapes	
:�*
dtype0

dense_2781/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_namedense_2781/kernel
x
%dense_2781/kernel/Read/ReadVariableOpReadVariableOpdense_2781/kernel*
_output_shapes
:	�*
dtype0
v
dense_2780/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_2780/bias
o
#dense_2780/bias/Read/ReadVariableOpReadVariableOpdense_2780/bias*
_output_shapes
:*
dtype0

dense_2780/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_namedense_2780/kernel
x
%dense_2780/kernel/Read/ReadVariableOpReadVariableOpdense_2780/kernel*
_output_shapes
:	�*
dtype0
�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*o
shared_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias
�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias/Read/ReadVariableOpReadVariableOp^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias*
_output_shapes	
:�*
dtype0
�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*q
shared_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel
�
tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel/Read/ReadVariableOpReadVariableOp`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel* 
_output_shapes
:
��*
dtype0
�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*o
shared_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias
�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias/Read/ReadVariableOpReadVariableOp^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias*
_output_shapes	
:�*
dtype0
�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*q
shared_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel
�
tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel/Read/ReadVariableOpReadVariableOp`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel* 
_output_shapes
:
��*
dtype0
�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*o
shared_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias
�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias/Read/ReadVariableOpReadVariableOp^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias*
_output_shapes	
:�*
dtype0
�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*q
shared_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel
�
tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel/Read/ReadVariableOpReadVariableOp`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel* 
_output_shapes
:
��*
dtype0
�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*o
shared_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias
�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias/Read/ReadVariableOpReadVariableOp^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias*
_output_shapes	
:�*
dtype0
�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*q
shared_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel
�
tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel/Read/ReadVariableOpReadVariableOp`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel* 
_output_shapes
:
��*
dtype0
�
.csmom_transformer_reranker_117/dense_2775/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.csmom_transformer_reranker_117/dense_2775/bias
�
Bcsmom_transformer_reranker_117/dense_2775/bias/Read/ReadVariableOpReadVariableOp.csmom_transformer_reranker_117/dense_2775/bias*
_output_shapes	
:�*
dtype0
�
0csmom_transformer_reranker_117/dense_2775/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*A
shared_name20csmom_transformer_reranker_117/dense_2775/kernel
�
Dcsmom_transformer_reranker_117/dense_2775/kernel/Read/ReadVariableOpReadVariableOp0csmom_transformer_reranker_117/dense_2775/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_input_1Placeholder*+
_output_shapes
:���������
*
dtype0* 
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10csmom_transformer_reranker_117/dense_2775/kernel.csmom_transformer_reranker_117/dense_2775/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betadense_2780/kerneldense_2780/biasdense_2781/kerneldense_2781/biasNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta0csmom_transformer_reranker_117/dense_2782/kernel.csmom_transformer_reranker_117/dense_2782/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_5177186

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ś
value��B�� B��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
input_layer
	
enc_layers


ffn_output
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19*
* 
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

&trace_0
'trace_1* 

(trace_0
)trace_1* 
* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias*

00*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
 bias*
�
7
_variables
8_iterations
9_learning_rate
:_index_dict
;
_momentums
<_velocities
=_update_step_xla*

>serving_default* 
pj
VARIABLE_VALUE0csmom_transformer_reranker_117/dense_2775/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE.csmom_transformer_reranker_117/dense_2775/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense_2780/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_2780/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense_2781/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_2781/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUENcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUENcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0csmom_transformer_reranker_117/dense_2782/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.csmom_transformer_reranker_117/dense_2782/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
01

2*

?0*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Etrace_0* 

Ftrace_0* 
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Matt
Nffn
O
layernorm1
P
layernorm2
Qdropout1
Rdropout2*

0
 1*

0
 1*
* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
�
80
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25
s26
t27
u28
v29
w30
x31
y32
z33
{34
|35
}36
~37
38
�39
�40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
Z0
\1
^2
`3
b4
d5
f6
h7
j8
l9
n10
p11
r12
t13
v14
x15
z16
|17
~18
�19*
�
[0
]1
_2
a3
c4
e5
g6
i7
k8
m9
o10
q11
s12
u13
w14
y15
{16
}17
18
�19*
* 
* 
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 
* 
* 
* 
* 
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15*
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�query_dense
�	key_dense
�value_dense
�combine_heads*
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
* 
* 
* 
* 
* 
* 
* 
�|
VARIABLE_VALUE7Adam/m/csmom_transformer_reranker_117/dense_2775/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE7Adam/v/csmom_transformer_reranker_117/dense_2775/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE5Adam/m/csmom_transformer_reranker_117/dense_2775/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE5Adam/v/csmom_transformer_reranker_117/dense_2775/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_2780/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_2780/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_2780/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_2780/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_2781/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_2781/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_2781/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_2781/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUETAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUETAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUETAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUETAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE7Adam/m/csmom_transformer_reranker_117/dense_2782/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE7Adam/v/csmom_transformer_reranker_117/dense_2782/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/m/csmom_transformer_reranker_117/dense_2782/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/v/csmom_transformer_reranker_117/dense_2782/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
M0
N1
O2
P3
Q4
R5*
* 
* 
* 
* 
* 
* 
* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
$
�0
�1
�2
�3*
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0csmom_transformer_reranker_117/dense_2775/kernel.csmom_transformer_reranker_117/dense_2775/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasdense_2780/kerneldense_2780/biasdense_2781/kerneldense_2781/biasNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta0csmom_transformer_reranker_117/dense_2782/kernel.csmom_transformer_reranker_117/dense_2782/bias	iterationlearning_rate7Adam/m/csmom_transformer_reranker_117/dense_2775/kernel7Adam/v/csmom_transformer_reranker_117/dense_2775/kernel5Adam/m/csmom_transformer_reranker_117/dense_2775/bias5Adam/v/csmom_transformer_reranker_117/dense_2775/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasAdam/m/dense_2780/kernelAdam/v/dense_2780/kernelAdam/m/dense_2780/biasAdam/v/dense_2780/biasAdam/m/dense_2781/kernelAdam/v/dense_2781/kernelAdam/m/dense_2781/biasAdam/v/dense_2781/biasUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/betaTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta7Adam/m/csmom_transformer_reranker_117/dense_2782/kernel7Adam/v/csmom_transformer_reranker_117/dense_2782/kernel5Adam/m/csmom_transformer_reranker_117/dense_2782/bias5Adam/v/csmom_transformer_reranker_117/dense_2782/biastotalcountConst*M
TinF
D2B*
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
 __inference__traced_save_5178345
�%
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename0csmom_transformer_reranker_117/dense_2775/kernel.csmom_transformer_reranker_117/dense_2775/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasdense_2780/kerneldense_2780/biasdense_2781/kerneldense_2781/biasNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta0csmom_transformer_reranker_117/dense_2782/kernel.csmom_transformer_reranker_117/dense_2782/bias	iterationlearning_rate7Adam/m/csmom_transformer_reranker_117/dense_2775/kernel7Adam/v/csmom_transformer_reranker_117/dense_2775/kernel5Adam/m/csmom_transformer_reranker_117/dense_2775/bias5Adam/v/csmom_transformer_reranker_117/dense_2775/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/biasgAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernelgAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kerneleAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biaseAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/biasAdam/m/dense_2780/kernelAdam/v/dense_2780/kernelAdam/m/dense_2780/biasAdam/v/dense_2780/biasAdam/m/dense_2781/kernelAdam/v/dense_2781/kernelAdam/m/dense_2781/biasAdam/v/dense_2781/biasUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gammaTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/betaUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gammaTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/betaTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta7Adam/m/csmom_transformer_reranker_117/dense_2782/kernel7Adam/v/csmom_transformer_reranker_117/dense_2782/kernel5Adam/m/csmom_transformer_reranker_117/dense_2782/bias5Adam/v/csmom_transformer_reranker_117/dense_2782/biastotalcount*L
TinE
C2A*
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
#__inference__traced_restore_5178546��
��
�R
 __inference__traced_save_5178345
file_prefixZ
Gread_disablecopyonread_csmom_transformer_reranker_117_dense_2775_kernel:	�V
Gread_1_disablecopyonread_csmom_transformer_reranker_117_dense_2775_bias:	��
yread_2_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel:
���
wread_3_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias:	��
yread_4_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel:
���
wread_5_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias:	��
yread_6_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel:
���
wread_7_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias:	��
yread_8_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel:
���
wread_9_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias:	�>
+read_10_disablecopyonread_dense_2780_kernel:	�7
)read_11_disablecopyonread_dense_2780_bias:>
+read_12_disablecopyonread_dense_2781_kernel:	�8
)read_13_disablecopyonread_dense_2781_bias:	�w
hread_14_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma:	�v
gread_15_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta:	�w
hread_16_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma:	�v
gread_17_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta:	�]
Jread_18_disablecopyonread_csmom_transformer_reranker_117_dense_2782_kernel:	�V
Hread_19_disablecopyonread_csmom_transformer_reranker_117_dense_2782_bias:-
#read_20_disablecopyonread_iteration:	 1
'read_21_disablecopyonread_learning_rate: d
Qread_22_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2775_kernel:	�d
Qread_23_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2775_kernel:	�^
Oread_24_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2775_bias:	�^
Oread_25_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2775_bias:	��
�read_26_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel:
���
�read_27_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel:
���
read_28_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias:	��
read_29_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias:	��
�read_30_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel:
���
�read_31_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel:
���
read_32_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias:	��
read_33_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias:	��
�read_34_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel:
���
�read_35_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel:
���
read_36_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias:	��
read_37_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias:	��
�read_38_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel:
���
�read_39_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel:
���
read_40_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias:	��
read_41_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias:	�E
2read_42_disablecopyonread_adam_m_dense_2780_kernel:	�E
2read_43_disablecopyonread_adam_v_dense_2780_kernel:	�>
0read_44_disablecopyonread_adam_m_dense_2780_bias:>
0read_45_disablecopyonread_adam_v_dense_2780_bias:E
2read_46_disablecopyonread_adam_m_dense_2781_kernel:	�E
2read_47_disablecopyonread_adam_v_dense_2781_kernel:	�?
0read_48_disablecopyonread_adam_m_dense_2781_bias:	�?
0read_49_disablecopyonread_adam_v_dense_2781_bias:	�~
oread_50_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma:	�~
oread_51_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma:	�}
nread_52_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta:	�}
nread_53_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta:	�~
oread_54_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma:	�~
oread_55_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma:	�}
nread_56_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta:	�}
nread_57_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta:	�d
Qread_58_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2782_kernel:	�d
Qread_59_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2782_kernel:	�]
Oread_60_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2782_bias:]
Oread_61_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2782_bias:)
read_62_disablecopyonread_total: )
read_63_disablecopyonread_count: 
savev2_const
identity_129��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnReadGread_disablecopyonread_csmom_transformer_reranker_117_dense_2775_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpGread_disablecopyonread_csmom_transformer_reranker_117_dense_2775_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_1/DisableCopyOnReadDisableCopyOnReadGread_1_disablecopyonread_csmom_transformer_reranker_117_dense_2775_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOpGread_1_disablecopyonread_csmom_transformer_reranker_117_dense_2775_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_2/DisableCopyOnReadDisableCopyOnReadyread_2_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOpyread_2_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_3/DisableCopyOnReadDisableCopyOnReadwread_3_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOpwread_3_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnReadyread_4_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpyread_4_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_5/DisableCopyOnReadDisableCopyOnReadwread_5_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpwread_5_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnReadyread_6_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpyread_6_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_7/DisableCopyOnReadDisableCopyOnReadwread_7_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpwread_7_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnReadyread_8_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpyread_8_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_9/DisableCopyOnReadDisableCopyOnReadwread_9_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpwread_9_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnRead+read_10_disablecopyonread_dense_2780_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp+read_10_disablecopyonread_dense_2780_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	�~
Read_11/DisableCopyOnReadDisableCopyOnRead)read_11_disablecopyonread_dense_2780_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp)read_11_disablecopyonread_dense_2780_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead+read_12_disablecopyonread_dense_2781_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp+read_12_disablecopyonread_dense_2781_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�~
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_dense_2781_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_dense_2781_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnReadhread_14_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOphread_14_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnReadgread_15_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpgread_15_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnReadhread_16_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOphread_16_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnReadgread_17_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpgread_17_disablecopyonread_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnReadJread_18_disablecopyonread_csmom_transformer_reranker_117_dense_2782_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpJread_18_disablecopyonread_csmom_transformer_reranker_117_dense_2782_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_19/DisableCopyOnReadDisableCopyOnReadHread_19_disablecopyonread_csmom_transformer_reranker_117_dense_2782_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpHread_19_disablecopyonread_csmom_transformer_reranker_117_dense_2782_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
:x
Read_20/DisableCopyOnReadDisableCopyOnRead#read_20_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp#read_20_disablecopyonread_iteration^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_learning_rate^Read_21/DisableCopyOnRead"/device:CPU:0*
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
: �
Read_22/DisableCopyOnReadDisableCopyOnReadQread_22_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2775_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpQread_22_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2775_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_23/DisableCopyOnReadDisableCopyOnReadQread_23_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2775_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpQread_23_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2775_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_24/DisableCopyOnReadDisableCopyOnReadOread_24_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2775_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpOread_24_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2775_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_25/DisableCopyOnReadDisableCopyOnReadOread_25_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2775_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpOread_25_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2775_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead�read_26_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp�read_26_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_27/DisableCopyOnReadDisableCopyOnRead�read_27_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp�read_27_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead�read_30_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp�read_30_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel^Read_30/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_31/DisableCopyOnReadDisableCopyOnRead�read_31_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp�read_31_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_32/DisableCopyOnReadDisableCopyOnReadread_32_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpread_32_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnReadread_33_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpread_33_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead�read_34_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp�read_34_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel^Read_34/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_35/DisableCopyOnReadDisableCopyOnRead�read_35_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp�read_35_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel^Read_35/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_36/DisableCopyOnReadDisableCopyOnReadread_36_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpread_36_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_37/DisableCopyOnReadDisableCopyOnReadread_37_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpread_37_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_38/DisableCopyOnReadDisableCopyOnRead�read_38_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp�read_38_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel^Read_38/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_39/DisableCopyOnReadDisableCopyOnRead�read_39_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp�read_39_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel^Read_39/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_40/DisableCopyOnReadDisableCopyOnReadread_40_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpread_40_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_41/DisableCopyOnReadDisableCopyOnReadread_41_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpread_41_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead2read_42_disablecopyonread_adam_m_dense_2780_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp2read_42_disablecopyonread_adam_m_dense_2780_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_43/DisableCopyOnReadDisableCopyOnRead2read_43_disablecopyonread_adam_v_dense_2780_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp2read_43_disablecopyonread_adam_v_dense_2780_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_44/DisableCopyOnReadDisableCopyOnRead0read_44_disablecopyonread_adam_m_dense_2780_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp0read_44_disablecopyonread_adam_m_dense_2780_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead0read_45_disablecopyonread_adam_v_dense_2780_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp0read_45_disablecopyonread_adam_v_dense_2780_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead2read_46_disablecopyonread_adam_m_dense_2781_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp2read_46_disablecopyonread_adam_m_dense_2781_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_47/DisableCopyOnReadDisableCopyOnRead2read_47_disablecopyonread_adam_v_dense_2781_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp2read_47_disablecopyonread_adam_v_dense_2781_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_dense_2781_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_dense_2781_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_dense_2781_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_dense_2781_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnReadoread_50_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOporead_50_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnReadoread_51_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOporead_51_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_52/DisableCopyOnReadDisableCopyOnReadnread_52_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpnread_52_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_53/DisableCopyOnReadDisableCopyOnReadnread_53_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpnread_53_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_54/DisableCopyOnReadDisableCopyOnReadoread_54_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOporead_54_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_55/DisableCopyOnReadDisableCopyOnReadoread_55_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOporead_55_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_56/DisableCopyOnReadDisableCopyOnReadnread_56_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpnread_56_disablecopyonread_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_57/DisableCopyOnReadDisableCopyOnReadnread_57_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpnread_57_disablecopyonread_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_58/DisableCopyOnReadDisableCopyOnReadQread_58_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2782_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpQread_58_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2782_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_59/DisableCopyOnReadDisableCopyOnReadQread_59_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2782_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpQread_59_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2782_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_60/DisableCopyOnReadDisableCopyOnReadOread_60_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2782_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOpOread_60_disablecopyonread_adam_m_csmom_transformer_reranker_117_dense_2782_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_61/DisableCopyOnReadDisableCopyOnReadOread_61_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2782_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOpOread_61_disablecopyonread_adam_v_csmom_transformer_reranker_117_dense_2782_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_62/DisableCopyOnReadDisableCopyOnReadread_62_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOpread_62_disablecopyonread_total^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_63/DisableCopyOnReadDisableCopyOnReadread_63_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOpread_63_disablecopyonread_count^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *O
dtypesE
C2A	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_128Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_129IdentityIdentity_128:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_129Identity_129:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=A9

_output_shapes
: 

_user_specified_nameConst:%@!

_user_specified_namecount:%?!

_user_specified_nametotal:U>Q
O
_user_specified_name75Adam/v/csmom_transformer_reranker_117/dense_2782/bias:U=Q
O
_user_specified_name75Adam/m/csmom_transformer_reranker_117/dense_2782/bias:W<S
Q
_user_specified_name97Adam/v/csmom_transformer_reranker_117/dense_2782/kernel:W;S
Q
_user_specified_name97Adam/m/csmom_transformer_reranker_117/dense_2782/kernel:t:p
n
_user_specified_nameVTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta:t9p
n
_user_specified_nameVTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta:u8q
o
_user_specified_nameWUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma:u7q
o
_user_specified_nameWUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma:t6p
n
_user_specified_nameVTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta:t5p
n
_user_specified_nameVTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta:u4q
o
_user_specified_nameWUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma:u3q
o
_user_specified_nameWUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma:622
0
_user_specified_nameAdam/v/dense_2781/bias:612
0
_user_specified_nameAdam/m/dense_2781/bias:804
2
_user_specified_nameAdam/v/dense_2781/kernel:8/4
2
_user_specified_nameAdam/m/dense_2781/kernel:6.2
0
_user_specified_nameAdam/v/dense_2780/bias:6-2
0
_user_specified_nameAdam/m/dense_2780/bias:8,4
2
_user_specified_nameAdam/v/dense_2780/kernel:8+4
2
_user_specified_nameAdam/m/dense_2780/kernel:�*�

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias:�)�

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias:�(�
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel:�'�
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel:�&�

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias:�%�

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias:�$�
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel:�#�
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel:�"�

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias:�!�

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias:� �
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel:��
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel:��

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias:��

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias:��
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel:��
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel:UQ
O
_user_specified_name75Adam/v/csmom_transformer_reranker_117/dense_2775/bias:UQ
O
_user_specified_name75Adam/m/csmom_transformer_reranker_117/dense_2775/bias:WS
Q
_user_specified_name97Adam/v/csmom_transformer_reranker_117/dense_2775/kernel:WS
Q
_user_specified_name97Adam/m/csmom_transformer_reranker_117/dense_2775/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:NJ
H
_user_specified_name0.csmom_transformer_reranker_117/dense_2782/bias:PL
J
_user_specified_name20csmom_transformer_reranker_117/dense_2782/kernel:mi
g
_user_specified_nameOMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta:nj
h
_user_specified_namePNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma:mi
g
_user_specified_nameOMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta:nj
h
_user_specified_namePNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma:/+
)
_user_specified_namedense_2781/bias:1-
+
_user_specified_namedense_2781/kernel:/+
)
_user_specified_namedense_2780/bias:1-
+
_user_specified_namedense_2780/kernel:~
z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias:�	|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel:~z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias:�|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel:~z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias:�|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel:~z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias:�|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel:NJ
H
_user_specified_name0.csmom_transformer_reranker_117/dense_2775/bias:PL
J
_user_specified_name20csmom_transformer_reranker_117/dense_2775/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5176585

inputs^
Jmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource:	�L
=layer_normalization_740_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_740_batchnorm_readvariableop_resource:	�N
;sequential_370_dense_2780_tensordot_readvariableop_resource:	�G
9sequential_370_dense_2780_biasadd_readvariableop_resource:N
;sequential_370_dense_2781_tensordot_readvariableop_resource:	�H
9sequential_370_dense_2781_biasadd_readvariableop_resource:	�L
=layer_normalization_741_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_741_batchnorm_readvariableop_resource:	�
identity��0layer_normalization_740/batchnorm/ReadVariableOp�4layer_normalization_740/batchnorm/mul/ReadVariableOp�0layer_normalization_741/batchnorm/ReadVariableOp�4layer_normalization_741/batchnorm/mul/ReadVariableOp�?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp�0sequential_370/dense_2780/BiasAdd/ReadVariableOp�2sequential_370/dense_2780/Tensordot/ReadVariableOp�0sequential_370/dense_2781/BiasAdd/ReadVariableOp�2sequential_370/dense_2781/Tensordot/ReadVariableOp�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2776/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2776/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2776/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2776/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2776/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2776/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2776/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2776/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2776/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2776/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2776/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/stackPack@multi_head_self_attention_370/dense_2776/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2776/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2776/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2776/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2776/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2776/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2776/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2776/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2776/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2776/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2776/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2776/TensordotReshapeCmulti_head_self_attention_370/dense_2776/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2776/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2776/BiasAddBiasAdd;multi_head_self_attention_370/dense_2776/Tensordot:output:0Gmulti_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2777/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2777/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2777/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2777/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2777/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2777/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2777/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2777/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2777/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2777/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2777/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/stackPack@multi_head_self_attention_370/dense_2777/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2777/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2777/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2777/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2777/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2777/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2777/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2777/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2777/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2777/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2777/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2777/TensordotReshapeCmulti_head_self_attention_370/dense_2777/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2777/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2777/BiasAddBiasAdd;multi_head_self_attention_370/dense_2777/Tensordot:output:0Gmulti_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2778/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2778/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2778/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2778/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2778/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2778/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2778/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2778/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2778/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2778/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2778/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/stackPack@multi_head_self_attention_370/dense_2778/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2778/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2778/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2778/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2778/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2778/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2778/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2778/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2778/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2778/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2778/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2778/TensordotReshapeCmulti_head_self_attention_370/dense_2778/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2778/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2778/BiasAddBiasAdd;multi_head_self_attention_370/dense_2778/Tensordot:output:0Gmulti_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
#multi_head_self_attention_370/ShapeShape9multi_head_self_attention_370/dense_2776/BiasAdd:output:0*
T0*
_output_shapes
::��{
1multi_head_self_attention_370/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_370/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3multi_head_self_attention_370/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+multi_head_self_attention_370/strided_sliceStridedSlice,multi_head_self_attention_370/Shape:output:0:multi_head_self_attention_370/strided_slice/stack:output:0<multi_head_self_attention_370/strided_slice/stack_1:output:0<multi_head_self_attention_370/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-multi_head_self_attention_370/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������o
-multi_head_self_attention_370/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p
-multi_head_self_attention_370/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
+multi_head_self_attention_370/Reshape/shapePack4multi_head_self_attention_370/strided_slice:output:06multi_head_self_attention_370/Reshape/shape/1:output:06multi_head_self_attention_370/Reshape/shape/2:output:06multi_head_self_attention_370/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
%multi_head_self_attention_370/ReshapeReshape9multi_head_self_attention_370/dense_2776/BiasAdd:output:04multi_head_self_attention_370/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
,multi_head_self_attention_370/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
'multi_head_self_attention_370/transpose	Transpose.multi_head_self_attention_370/Reshape:output:05multi_head_self_attention_370/transpose/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_1Shape9multi_head_self_attention_370/dense_2777/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_1StridedSlice.multi_head_self_attention_370/Shape_1:output:0<multi_head_self_attention_370/strided_slice_1/stack:output:0>multi_head_self_attention_370/strided_slice_1/stack_1:output:0>multi_head_self_attention_370/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_1/shapePack6multi_head_self_attention_370/strided_slice_1:output:08multi_head_self_attention_370/Reshape_1/shape/1:output:08multi_head_self_attention_370/Reshape_1/shape/2:output:08multi_head_self_attention_370/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_1Reshape9multi_head_self_attention_370/dense_2777/BiasAdd:output:06multi_head_self_attention_370/Reshape_1/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_1	Transpose0multi_head_self_attention_370/Reshape_1:output:07multi_head_self_attention_370/transpose_1/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_2Shape9multi_head_self_attention_370/dense_2778/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_2StridedSlice.multi_head_self_attention_370/Shape_2:output:0<multi_head_self_attention_370/strided_slice_2/stack:output:0>multi_head_self_attention_370/strided_slice_2/stack_1:output:0>multi_head_self_attention_370/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_2/shapePack6multi_head_self_attention_370/strided_slice_2:output:08multi_head_self_attention_370/Reshape_2/shape/1:output:08multi_head_self_attention_370/Reshape_2/shape/2:output:08multi_head_self_attention_370/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_2Reshape9multi_head_self_attention_370/dense_2778/BiasAdd:output:06multi_head_self_attention_370/Reshape_2/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_2	Transpose0multi_head_self_attention_370/Reshape_2:output:07multi_head_self_attention_370/transpose_2/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
$multi_head_self_attention_370/MatMulBatchMatMulV2+multi_head_self_attention_370/transpose:y:0-multi_head_self_attention_370/transpose_1:y:0*
T0*A
_output_shapes/
-:+���������������������������*
adj_y(i
$multi_head_self_attention_370/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *   Cz
"multi_head_self_attention_370/SqrtSqrt-multi_head_self_attention_370/Sqrt/x:output:0*
T0*
_output_shapes
: �
%multi_head_self_attention_370/truedivRealDiv-multi_head_self_attention_370/MatMul:output:0&multi_head_self_attention_370/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
%multi_head_self_attention_370/SoftmaxSoftmax)multi_head_self_attention_370/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
&multi_head_self_attention_370/MatMul_1BatchMatMulV2/multi_head_self_attention_370/Softmax:softmax:0-multi_head_self_attention_370/transpose_2:y:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_3	Transpose/multi_head_self_attention_370/MatMul_1:output:07multi_head_self_attention_370/transpose_3/perm:output:0*
T0*9
_output_shapes'
%:#�������������������i
%multi_head_self_attention_370/Shape_3Shapeinputs*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_3StridedSlice.multi_head_self_attention_370/Shape_3:output:0<multi_head_self_attention_370/strided_slice_3/stack:output:0>multi_head_self_attention_370/strided_slice_3/stack_1:output:0>multi_head_self_attention_370/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/multi_head_self_attention_370/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_3/shapePack6multi_head_self_attention_370/strided_slice_3:output:08multi_head_self_attention_370/Reshape_3/shape/1:output:08multi_head_self_attention_370/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_3Reshape-multi_head_self_attention_370/transpose_3:y:06multi_head_self_attention_370/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2779/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2779/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8multi_head_self_attention_370/dense_2779/Tensordot/ShapeShape0multi_head_self_attention_370/Reshape_3:output:0*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2779/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2779/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2779/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2779/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2779/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2779/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2779/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2779/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/stackPack@multi_head_self_attention_370/dense_2779/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2779/Tensordot/transpose	Transpose0multi_head_self_attention_370/Reshape_3:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
:multi_head_self_attention_370/dense_2779/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2779/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2779/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2779/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2779/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2779/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2779/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2779/TensordotReshapeCmulti_head_self_attention_370/dense_2779/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2779/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2779/BiasAddBiasAdd;multi_head_self_attention_370/dense_2779/Tensordot:output:0Gmulti_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
dropout_874/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_874/dropout/MulMul9multi_head_self_attention_370/dense_2779/BiasAdd:output:0"dropout_874/dropout/Const:output:0*
T0*5
_output_shapes#
!:��������������������
dropout_874/dropout/ShapeShape9multi_head_self_attention_370/dense_2779/BiasAdd:output:0*
T0*
_output_shapes
::���
0dropout_874/dropout/random_uniform/RandomUniformRandomUniform"dropout_874/dropout/Shape:output:0*
T0*5
_output_shapes#
!:�������������������*
dtype0g
"dropout_874/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
 dropout_874/dropout/GreaterEqualGreaterEqual9dropout_874/dropout/random_uniform/RandomUniform:output:0+dropout_874/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:�������������������`
dropout_874/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_874/dropout/SelectV2SelectV2$dropout_874/dropout/GreaterEqual:z:0dropout_874/dropout/Mul:z:0$dropout_874/dropout/Const_1:output:0*
T0*5
_output_shapes#
!:�������������������r
addAddV2inputs%dropout_874/dropout/SelectV2:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_740/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_740/moments/meanMeanadd:z:0?layer_normalization_740/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_740/moments/StopGradientStopGradient-layer_normalization_740/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_740/moments/SquaredDifferenceSquaredDifferenceadd:z:05layer_normalization_740/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_740/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_740/moments/varianceMean5layer_normalization_740/moments/SquaredDifference:z:0Clayer_normalization_740/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_740/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_740/batchnorm/addAddV21layer_normalization_740/moments/variance:output:00layer_normalization_740/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_740/batchnorm/RsqrtRsqrt)layer_normalization_740/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_740/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_740_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/mulMul+layer_normalization_740/batchnorm/Rsqrt:y:0<layer_normalization_740/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_1Muladd:z:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_2Mul-layer_normalization_740/moments/mean:output:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_740/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_740_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/subSub8layer_normalization_740/batchnorm/ReadVariableOp:value:0+layer_normalization_740/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/add_1AddV2+layer_normalization_740/batchnorm/mul_1:z:0)layer_normalization_740/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
��
2sequential_370/dense_2780/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2780_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2780/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2780/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2780/Tensordot/ShapeShape+layer_normalization_740/batchnorm/add_1:z:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2780/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/GatherV2GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/free:output:0:sequential_370/dense_2780/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2780/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2780/Tensordot/GatherV2_1GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/axes:output:0<sequential_370/dense_2780/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2780/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2780/Tensordot/ProdProd5sequential_370/dense_2780/Tensordot/GatherV2:output:02sequential_370/dense_2780/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2780/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2780/Tensordot/Prod_1Prod7sequential_370/dense_2780/Tensordot/GatherV2_1:output:04sequential_370/dense_2780/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2780/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2780/Tensordot/concatConcatV21sequential_370/dense_2780/Tensordot/free:output:01sequential_370/dense_2780/Tensordot/axes:output:08sequential_370/dense_2780/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2780/Tensordot/stackPack1sequential_370/dense_2780/Tensordot/Prod:output:03sequential_370/dense_2780/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2780/Tensordot/transpose	Transpose+layer_normalization_740/batchnorm/add_1:z:03sequential_370/dense_2780/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
+sequential_370/dense_2780/Tensordot/ReshapeReshape1sequential_370/dense_2780/Tensordot/transpose:y:02sequential_370/dense_2780/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2780/Tensordot/MatMulMatMul4sequential_370/dense_2780/Tensordot/Reshape:output:0:sequential_370/dense_2780/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
+sequential_370/dense_2780/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:s
1sequential_370/dense_2780/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/concat_1ConcatV25sequential_370/dense_2780/Tensordot/GatherV2:output:04sequential_370/dense_2780/Tensordot/Const_2:output:0:sequential_370/dense_2780/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2780/TensordotReshape4sequential_370/dense_2780/Tensordot/MatMul:product:05sequential_370/dense_2780/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
�
0sequential_370/dense_2780/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2780_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_370/dense_2780/BiasAddBiasAdd,sequential_370/dense_2780/Tensordot:output:08sequential_370/dense_2780/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
�
sequential_370/dense_2780/ReluRelu*sequential_370/dense_2780/BiasAdd:output:0*
T0*+
_output_shapes
:���������
�
2sequential_370/dense_2781/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2781_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2781/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2781/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2781/Tensordot/ShapeShape,sequential_370/dense_2780/Relu:activations:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2781/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/GatherV2GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/free:output:0:sequential_370/dense_2781/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2781/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2781/Tensordot/GatherV2_1GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/axes:output:0<sequential_370/dense_2781/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2781/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2781/Tensordot/ProdProd5sequential_370/dense_2781/Tensordot/GatherV2:output:02sequential_370/dense_2781/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2781/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2781/Tensordot/Prod_1Prod7sequential_370/dense_2781/Tensordot/GatherV2_1:output:04sequential_370/dense_2781/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2781/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2781/Tensordot/concatConcatV21sequential_370/dense_2781/Tensordot/free:output:01sequential_370/dense_2781/Tensordot/axes:output:08sequential_370/dense_2781/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2781/Tensordot/stackPack1sequential_370/dense_2781/Tensordot/Prod:output:03sequential_370/dense_2781/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2781/Tensordot/transpose	Transpose,sequential_370/dense_2780/Relu:activations:03sequential_370/dense_2781/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
+sequential_370/dense_2781/Tensordot/ReshapeReshape1sequential_370/dense_2781/Tensordot/transpose:y:02sequential_370/dense_2781/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2781/Tensordot/MatMulMatMul4sequential_370/dense_2781/Tensordot/Reshape:output:0:sequential_370/dense_2781/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
+sequential_370/dense_2781/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�s
1sequential_370/dense_2781/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/concat_1ConcatV25sequential_370/dense_2781/Tensordot/GatherV2:output:04sequential_370/dense_2781/Tensordot/Const_2:output:0:sequential_370/dense_2781/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2781/TensordotReshape4sequential_370/dense_2781/Tensordot/MatMul:product:05sequential_370/dense_2781/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
0sequential_370/dense_2781/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2781_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!sequential_370/dense_2781/BiasAddBiasAdd,sequential_370/dense_2781/Tensordot:output:08sequential_370/dense_2781/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
�^
dropout_875/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_875/dropout/MulMul*sequential_370/dense_2781/BiasAdd:output:0"dropout_875/dropout/Const:output:0*
T0*,
_output_shapes
:���������
��
dropout_875/dropout/ShapeShape*sequential_370/dense_2781/BiasAdd:output:0*
T0*
_output_shapes
::���
0dropout_875/dropout/random_uniform/RandomUniformRandomUniform"dropout_875/dropout/Shape:output:0*
T0*,
_output_shapes
:���������
�*
dtype0g
"dropout_875/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
 dropout_875/dropout/GreaterEqualGreaterEqual9dropout_875/dropout/random_uniform/RandomUniform:output:0+dropout_875/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������
�`
dropout_875/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_875/dropout/SelectV2SelectV2$dropout_875/dropout/GreaterEqual:z:0dropout_875/dropout/Mul:z:0$dropout_875/dropout/Const_1:output:0*
T0*,
_output_shapes
:���������
��
add_1AddV2+layer_normalization_740/batchnorm/add_1:z:0%dropout_875/dropout/SelectV2:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_741/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_741/moments/meanMean	add_1:z:0?layer_normalization_741/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_741/moments/StopGradientStopGradient-layer_normalization_741/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_741/moments/SquaredDifferenceSquaredDifference	add_1:z:05layer_normalization_741/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_741/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_741/moments/varianceMean5layer_normalization_741/moments/SquaredDifference:z:0Clayer_normalization_741/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_741/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_741/batchnorm/addAddV21layer_normalization_741/moments/variance:output:00layer_normalization_741/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_741/batchnorm/RsqrtRsqrt)layer_normalization_741/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_741/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_741_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/mulMul+layer_normalization_741/batchnorm/Rsqrt:y:0<layer_normalization_741/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_1Mul	add_1:z:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_2Mul-layer_normalization_741/moments/mean:output:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_741/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_741_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/subSub8layer_normalization_741/batchnorm/ReadVariableOp:value:0+layer_normalization_741/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/add_1AddV2+layer_normalization_741/batchnorm/mul_1:z:0)layer_normalization_741/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
�
IdentityIdentity+layer_normalization_741/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������
��
NoOpNoOp1^layer_normalization_740/batchnorm/ReadVariableOp5^layer_normalization_740/batchnorm/mul/ReadVariableOp1^layer_normalization_741/batchnorm/ReadVariableOp5^layer_normalization_741/batchnorm/mul/ReadVariableOp@^multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp1^sequential_370/dense_2780/BiasAdd/ReadVariableOp3^sequential_370/dense_2780/Tensordot/ReadVariableOp1^sequential_370/dense_2781/BiasAdd/ReadVariableOp3^sequential_370/dense_2781/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������
�: : : : : : : : : : : : : : : : 2d
0layer_normalization_740/batchnorm/ReadVariableOp0layer_normalization_740/batchnorm/ReadVariableOp2l
4layer_normalization_740/batchnorm/mul/ReadVariableOp4layer_normalization_740/batchnorm/mul/ReadVariableOp2d
0layer_normalization_741/batchnorm/ReadVariableOp0layer_normalization_741/batchnorm/ReadVariableOp2l
4layer_normalization_741/batchnorm/mul/ReadVariableOp4layer_normalization_741/batchnorm/mul/ReadVariableOp2�
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp2d
0sequential_370/dense_2780/BiasAdd/ReadVariableOp0sequential_370/dense_2780/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2780/Tensordot/ReadVariableOp2sequential_370/dense_2780/Tensordot/ReadVariableOp2d
0sequential_370/dense_2781/BiasAdd/ReadVariableOp0sequential_370/dense_2781/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2781/Tensordot/ReadVariableOp2sequential_370/dense_2781/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
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
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
G__inference_dense_2782_layer_call_and_return_conditional_losses_5176648

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������
V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
3__inference_encoder_layer_370_layer_call_fn_5177338

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5176916t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������
�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177334:'#
!
_user_specified_name	5177332:'#
!
_user_specified_name	5177330:'#
!
_user_specified_name	5177328:'#
!
_user_specified_name	5177326:'#
!
_user_specified_name	5177324:'
#
!
_user_specified_name	5177322:'	#
!
_user_specified_name	5177320:'#
!
_user_specified_name	5177318:'#
!
_user_specified_name	5177316:'#
!
_user_specified_name	5177314:'#
!
_user_specified_name	5177312:'#
!
_user_specified_name	5177310:'#
!
_user_specified_name	5177308:'#
!
_user_specified_name	5177306:'#
!
_user_specified_name	5177304:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176236
dense_2780_input%
dense_2780_5176225:	� 
dense_2780_5176227:%
dense_2781_5176230:	�!
dense_2781_5176232:	�
identity��"dense_2780/StatefulPartitionedCall�"dense_2781/StatefulPartitionedCall�
"dense_2780/StatefulPartitionedCallStatefulPartitionedCalldense_2780_inputdense_2780_5176225dense_2780_5176227*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2780_layer_call_and_return_conditional_losses_5176180�
"dense_2781/StatefulPartitionedCallStatefulPartitionedCall+dense_2780/StatefulPartitionedCall:output:0dense_2781_5176230dense_2781_5176232*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2781_layer_call_and_return_conditional_losses_5176215
IdentityIdentity+dense_2781/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�l
NoOpNoOp#^dense_2780/StatefulPartitionedCall#^dense_2781/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������
�: : : : 2H
"dense_2780/StatefulPartitionedCall"dense_2780/StatefulPartitionedCall2H
"dense_2781/StatefulPartitionedCall"dense_2781/StatefulPartitionedCall:'#
!
_user_specified_name	5176232:'#
!
_user_specified_name	5176230:'#
!
_user_specified_name	5176227:'#
!
_user_specified_name	5176225:^ Z
,
_output_shapes
:���������
�
*
_user_specified_namedense_2780_input
�!
�
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176956
input_1%
dense_2775_5176658:	�!
dense_2775_5176660:	�-
encoder_layer_370_5176917:
��(
encoder_layer_370_5176919:	�-
encoder_layer_370_5176921:
��(
encoder_layer_370_5176923:	�-
encoder_layer_370_5176925:
��(
encoder_layer_370_5176927:	�-
encoder_layer_370_5176929:
��(
encoder_layer_370_5176931:	�(
encoder_layer_370_5176933:	�(
encoder_layer_370_5176935:	�,
encoder_layer_370_5176937:	�'
encoder_layer_370_5176939:,
encoder_layer_370_5176941:	�(
encoder_layer_370_5176943:	�(
encoder_layer_370_5176945:	�(
encoder_layer_370_5176947:	�%
dense_2782_5176950:	� 
dense_2782_5176952:
identity��"dense_2775/StatefulPartitionedCall�"dense_2782/StatefulPartitionedCall�)encoder_layer_370/StatefulPartitionedCall�
"dense_2775/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2775_5176658dense_2775_5176660*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2775_layer_call_and_return_conditional_losses_5176312�
)encoder_layer_370/StatefulPartitionedCallStatefulPartitionedCall+dense_2775/StatefulPartitionedCall:output:0encoder_layer_370_5176917encoder_layer_370_5176919encoder_layer_370_5176921encoder_layer_370_5176923encoder_layer_370_5176925encoder_layer_370_5176927encoder_layer_370_5176929encoder_layer_370_5176931encoder_layer_370_5176933encoder_layer_370_5176935encoder_layer_370_5176937encoder_layer_370_5176939encoder_layer_370_5176941encoder_layer_370_5176943encoder_layer_370_5176945encoder_layer_370_5176947*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5176916�
"dense_2782/StatefulPartitionedCallStatefulPartitionedCall2encoder_layer_370/StatefulPartitionedCall:output:0dense_2782_5176950dense_2782_5176952*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2782_layer_call_and_return_conditional_losses_5176648~
IdentityIdentity+dense_2782/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
�
NoOpNoOp#^dense_2775/StatefulPartitionedCall#^dense_2782/StatefulPartitionedCall*^encoder_layer_370/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������
: : : : : : : : : : : : : : : : : : : : 2H
"dense_2775/StatefulPartitionedCall"dense_2775/StatefulPartitionedCall2H
"dense_2782/StatefulPartitionedCall"dense_2782/StatefulPartitionedCall2V
)encoder_layer_370/StatefulPartitionedCall)encoder_layer_370/StatefulPartitionedCall:'#
!
_user_specified_name	5176952:'#
!
_user_specified_name	5176950:'#
!
_user_specified_name	5176947:'#
!
_user_specified_name	5176945:'#
!
_user_specified_name	5176943:'#
!
_user_specified_name	5176941:'#
!
_user_specified_name	5176939:'#
!
_user_specified_name	5176937:'#
!
_user_specified_name	5176935:'#
!
_user_specified_name	5176933:'
#
!
_user_specified_name	5176931:'	#
!
_user_specified_name	5176929:'#
!
_user_specified_name	5176927:'#
!
_user_specified_name	5176925:'#
!
_user_specified_name	5176923:'#
!
_user_specified_name	5176921:'#
!
_user_specified_name	5176919:'#
!
_user_specified_name	5176917:'#
!
_user_specified_name	5176660:'#
!
_user_specified_name	5176658:T P
+
_output_shapes
:���������

!
_user_specified_name	input_1
�	
�
0__inference_sequential_370_layer_call_fn_5176262
dense_2780_input
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_2780_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176236t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������
�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5176258:'#
!
_user_specified_name	5176256:'#
!
_user_specified_name	5176254:'#
!
_user_specified_name	5176252:^ Z
,
_output_shapes
:���������
�
*
_user_specified_namedense_2780_input
��
�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5177606

inputs^
Jmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource:	�L
=layer_normalization_740_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_740_batchnorm_readvariableop_resource:	�N
;sequential_370_dense_2780_tensordot_readvariableop_resource:	�G
9sequential_370_dense_2780_biasadd_readvariableop_resource:N
;sequential_370_dense_2781_tensordot_readvariableop_resource:	�H
9sequential_370_dense_2781_biasadd_readvariableop_resource:	�L
=layer_normalization_741_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_741_batchnorm_readvariableop_resource:	�
identity��0layer_normalization_740/batchnorm/ReadVariableOp�4layer_normalization_740/batchnorm/mul/ReadVariableOp�0layer_normalization_741/batchnorm/ReadVariableOp�4layer_normalization_741/batchnorm/mul/ReadVariableOp�?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp�0sequential_370/dense_2780/BiasAdd/ReadVariableOp�2sequential_370/dense_2780/Tensordot/ReadVariableOp�0sequential_370/dense_2781/BiasAdd/ReadVariableOp�2sequential_370/dense_2781/Tensordot/ReadVariableOp�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2776/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2776/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2776/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2776/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2776/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2776/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2776/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2776/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2776/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2776/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2776/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/stackPack@multi_head_self_attention_370/dense_2776/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2776/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2776/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2776/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2776/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2776/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2776/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2776/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2776/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2776/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2776/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2776/TensordotReshapeCmulti_head_self_attention_370/dense_2776/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2776/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2776/BiasAddBiasAdd;multi_head_self_attention_370/dense_2776/Tensordot:output:0Gmulti_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2777/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2777/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2777/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2777/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2777/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2777/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2777/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2777/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2777/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2777/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2777/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/stackPack@multi_head_self_attention_370/dense_2777/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2777/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2777/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2777/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2777/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2777/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2777/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2777/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2777/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2777/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2777/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2777/TensordotReshapeCmulti_head_self_attention_370/dense_2777/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2777/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2777/BiasAddBiasAdd;multi_head_self_attention_370/dense_2777/Tensordot:output:0Gmulti_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2778/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2778/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2778/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2778/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2778/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2778/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2778/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2778/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2778/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2778/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2778/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/stackPack@multi_head_self_attention_370/dense_2778/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2778/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2778/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2778/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2778/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2778/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2778/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2778/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2778/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2778/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2778/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2778/TensordotReshapeCmulti_head_self_attention_370/dense_2778/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2778/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2778/BiasAddBiasAdd;multi_head_self_attention_370/dense_2778/Tensordot:output:0Gmulti_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
#multi_head_self_attention_370/ShapeShape9multi_head_self_attention_370/dense_2776/BiasAdd:output:0*
T0*
_output_shapes
::��{
1multi_head_self_attention_370/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_370/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3multi_head_self_attention_370/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+multi_head_self_attention_370/strided_sliceStridedSlice,multi_head_self_attention_370/Shape:output:0:multi_head_self_attention_370/strided_slice/stack:output:0<multi_head_self_attention_370/strided_slice/stack_1:output:0<multi_head_self_attention_370/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-multi_head_self_attention_370/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������o
-multi_head_self_attention_370/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p
-multi_head_self_attention_370/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
+multi_head_self_attention_370/Reshape/shapePack4multi_head_self_attention_370/strided_slice:output:06multi_head_self_attention_370/Reshape/shape/1:output:06multi_head_self_attention_370/Reshape/shape/2:output:06multi_head_self_attention_370/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
%multi_head_self_attention_370/ReshapeReshape9multi_head_self_attention_370/dense_2776/BiasAdd:output:04multi_head_self_attention_370/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
,multi_head_self_attention_370/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
'multi_head_self_attention_370/transpose	Transpose.multi_head_self_attention_370/Reshape:output:05multi_head_self_attention_370/transpose/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_1Shape9multi_head_self_attention_370/dense_2777/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_1StridedSlice.multi_head_self_attention_370/Shape_1:output:0<multi_head_self_attention_370/strided_slice_1/stack:output:0>multi_head_self_attention_370/strided_slice_1/stack_1:output:0>multi_head_self_attention_370/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_1/shapePack6multi_head_self_attention_370/strided_slice_1:output:08multi_head_self_attention_370/Reshape_1/shape/1:output:08multi_head_self_attention_370/Reshape_1/shape/2:output:08multi_head_self_attention_370/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_1Reshape9multi_head_self_attention_370/dense_2777/BiasAdd:output:06multi_head_self_attention_370/Reshape_1/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_1	Transpose0multi_head_self_attention_370/Reshape_1:output:07multi_head_self_attention_370/transpose_1/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_2Shape9multi_head_self_attention_370/dense_2778/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_2StridedSlice.multi_head_self_attention_370/Shape_2:output:0<multi_head_self_attention_370/strided_slice_2/stack:output:0>multi_head_self_attention_370/strided_slice_2/stack_1:output:0>multi_head_self_attention_370/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_2/shapePack6multi_head_self_attention_370/strided_slice_2:output:08multi_head_self_attention_370/Reshape_2/shape/1:output:08multi_head_self_attention_370/Reshape_2/shape/2:output:08multi_head_self_attention_370/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_2Reshape9multi_head_self_attention_370/dense_2778/BiasAdd:output:06multi_head_self_attention_370/Reshape_2/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_2	Transpose0multi_head_self_attention_370/Reshape_2:output:07multi_head_self_attention_370/transpose_2/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
$multi_head_self_attention_370/MatMulBatchMatMulV2+multi_head_self_attention_370/transpose:y:0-multi_head_self_attention_370/transpose_1:y:0*
T0*A
_output_shapes/
-:+���������������������������*
adj_y(i
$multi_head_self_attention_370/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *   Cz
"multi_head_self_attention_370/SqrtSqrt-multi_head_self_attention_370/Sqrt/x:output:0*
T0*
_output_shapes
: �
%multi_head_self_attention_370/truedivRealDiv-multi_head_self_attention_370/MatMul:output:0&multi_head_self_attention_370/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
%multi_head_self_attention_370/SoftmaxSoftmax)multi_head_self_attention_370/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
&multi_head_self_attention_370/MatMul_1BatchMatMulV2/multi_head_self_attention_370/Softmax:softmax:0-multi_head_self_attention_370/transpose_2:y:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_3	Transpose/multi_head_self_attention_370/MatMul_1:output:07multi_head_self_attention_370/transpose_3/perm:output:0*
T0*9
_output_shapes'
%:#�������������������i
%multi_head_self_attention_370/Shape_3Shapeinputs*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_3StridedSlice.multi_head_self_attention_370/Shape_3:output:0<multi_head_self_attention_370/strided_slice_3/stack:output:0>multi_head_self_attention_370/strided_slice_3/stack_1:output:0>multi_head_self_attention_370/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/multi_head_self_attention_370/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_3/shapePack6multi_head_self_attention_370/strided_slice_3:output:08multi_head_self_attention_370/Reshape_3/shape/1:output:08multi_head_self_attention_370/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_3Reshape-multi_head_self_attention_370/transpose_3:y:06multi_head_self_attention_370/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2779/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2779/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8multi_head_self_attention_370/dense_2779/Tensordot/ShapeShape0multi_head_self_attention_370/Reshape_3:output:0*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2779/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2779/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2779/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2779/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2779/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2779/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2779/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2779/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/stackPack@multi_head_self_attention_370/dense_2779/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2779/Tensordot/transpose	Transpose0multi_head_self_attention_370/Reshape_3:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
:multi_head_self_attention_370/dense_2779/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2779/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2779/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2779/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2779/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2779/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2779/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2779/TensordotReshapeCmulti_head_self_attention_370/dense_2779/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2779/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2779/BiasAddBiasAdd;multi_head_self_attention_370/dense_2779/Tensordot:output:0Gmulti_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������^
dropout_874/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_874/dropout/MulMul9multi_head_self_attention_370/dense_2779/BiasAdd:output:0"dropout_874/dropout/Const:output:0*
T0*5
_output_shapes#
!:��������������������
dropout_874/dropout/ShapeShape9multi_head_self_attention_370/dense_2779/BiasAdd:output:0*
T0*
_output_shapes
::���
0dropout_874/dropout/random_uniform/RandomUniformRandomUniform"dropout_874/dropout/Shape:output:0*
T0*5
_output_shapes#
!:�������������������*
dtype0g
"dropout_874/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
 dropout_874/dropout/GreaterEqualGreaterEqual9dropout_874/dropout/random_uniform/RandomUniform:output:0+dropout_874/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:�������������������`
dropout_874/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_874/dropout/SelectV2SelectV2$dropout_874/dropout/GreaterEqual:z:0dropout_874/dropout/Mul:z:0$dropout_874/dropout/Const_1:output:0*
T0*5
_output_shapes#
!:�������������������r
addAddV2inputs%dropout_874/dropout/SelectV2:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_740/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_740/moments/meanMeanadd:z:0?layer_normalization_740/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_740/moments/StopGradientStopGradient-layer_normalization_740/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_740/moments/SquaredDifferenceSquaredDifferenceadd:z:05layer_normalization_740/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_740/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_740/moments/varianceMean5layer_normalization_740/moments/SquaredDifference:z:0Clayer_normalization_740/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_740/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_740/batchnorm/addAddV21layer_normalization_740/moments/variance:output:00layer_normalization_740/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_740/batchnorm/RsqrtRsqrt)layer_normalization_740/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_740/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_740_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/mulMul+layer_normalization_740/batchnorm/Rsqrt:y:0<layer_normalization_740/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_1Muladd:z:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_2Mul-layer_normalization_740/moments/mean:output:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_740/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_740_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/subSub8layer_normalization_740/batchnorm/ReadVariableOp:value:0+layer_normalization_740/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/add_1AddV2+layer_normalization_740/batchnorm/mul_1:z:0)layer_normalization_740/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
��
2sequential_370/dense_2780/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2780_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2780/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2780/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2780/Tensordot/ShapeShape+layer_normalization_740/batchnorm/add_1:z:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2780/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/GatherV2GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/free:output:0:sequential_370/dense_2780/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2780/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2780/Tensordot/GatherV2_1GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/axes:output:0<sequential_370/dense_2780/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2780/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2780/Tensordot/ProdProd5sequential_370/dense_2780/Tensordot/GatherV2:output:02sequential_370/dense_2780/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2780/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2780/Tensordot/Prod_1Prod7sequential_370/dense_2780/Tensordot/GatherV2_1:output:04sequential_370/dense_2780/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2780/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2780/Tensordot/concatConcatV21sequential_370/dense_2780/Tensordot/free:output:01sequential_370/dense_2780/Tensordot/axes:output:08sequential_370/dense_2780/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2780/Tensordot/stackPack1sequential_370/dense_2780/Tensordot/Prod:output:03sequential_370/dense_2780/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2780/Tensordot/transpose	Transpose+layer_normalization_740/batchnorm/add_1:z:03sequential_370/dense_2780/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
+sequential_370/dense_2780/Tensordot/ReshapeReshape1sequential_370/dense_2780/Tensordot/transpose:y:02sequential_370/dense_2780/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2780/Tensordot/MatMulMatMul4sequential_370/dense_2780/Tensordot/Reshape:output:0:sequential_370/dense_2780/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
+sequential_370/dense_2780/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:s
1sequential_370/dense_2780/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/concat_1ConcatV25sequential_370/dense_2780/Tensordot/GatherV2:output:04sequential_370/dense_2780/Tensordot/Const_2:output:0:sequential_370/dense_2780/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2780/TensordotReshape4sequential_370/dense_2780/Tensordot/MatMul:product:05sequential_370/dense_2780/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
�
0sequential_370/dense_2780/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2780_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_370/dense_2780/BiasAddBiasAdd,sequential_370/dense_2780/Tensordot:output:08sequential_370/dense_2780/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
�
sequential_370/dense_2780/ReluRelu*sequential_370/dense_2780/BiasAdd:output:0*
T0*+
_output_shapes
:���������
�
2sequential_370/dense_2781/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2781_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2781/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2781/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2781/Tensordot/ShapeShape,sequential_370/dense_2780/Relu:activations:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2781/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/GatherV2GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/free:output:0:sequential_370/dense_2781/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2781/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2781/Tensordot/GatherV2_1GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/axes:output:0<sequential_370/dense_2781/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2781/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2781/Tensordot/ProdProd5sequential_370/dense_2781/Tensordot/GatherV2:output:02sequential_370/dense_2781/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2781/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2781/Tensordot/Prod_1Prod7sequential_370/dense_2781/Tensordot/GatherV2_1:output:04sequential_370/dense_2781/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2781/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2781/Tensordot/concatConcatV21sequential_370/dense_2781/Tensordot/free:output:01sequential_370/dense_2781/Tensordot/axes:output:08sequential_370/dense_2781/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2781/Tensordot/stackPack1sequential_370/dense_2781/Tensordot/Prod:output:03sequential_370/dense_2781/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2781/Tensordot/transpose	Transpose,sequential_370/dense_2780/Relu:activations:03sequential_370/dense_2781/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
+sequential_370/dense_2781/Tensordot/ReshapeReshape1sequential_370/dense_2781/Tensordot/transpose:y:02sequential_370/dense_2781/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2781/Tensordot/MatMulMatMul4sequential_370/dense_2781/Tensordot/Reshape:output:0:sequential_370/dense_2781/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
+sequential_370/dense_2781/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�s
1sequential_370/dense_2781/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/concat_1ConcatV25sequential_370/dense_2781/Tensordot/GatherV2:output:04sequential_370/dense_2781/Tensordot/Const_2:output:0:sequential_370/dense_2781/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2781/TensordotReshape4sequential_370/dense_2781/Tensordot/MatMul:product:05sequential_370/dense_2781/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
0sequential_370/dense_2781/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2781_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!sequential_370/dense_2781/BiasAddBiasAdd,sequential_370/dense_2781/Tensordot:output:08sequential_370/dense_2781/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
�^
dropout_875/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_875/dropout/MulMul*sequential_370/dense_2781/BiasAdd:output:0"dropout_875/dropout/Const:output:0*
T0*,
_output_shapes
:���������
��
dropout_875/dropout/ShapeShape*sequential_370/dense_2781/BiasAdd:output:0*
T0*
_output_shapes
::���
0dropout_875/dropout/random_uniform/RandomUniformRandomUniform"dropout_875/dropout/Shape:output:0*
T0*,
_output_shapes
:���������
�*
dtype0g
"dropout_875/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
 dropout_875/dropout/GreaterEqualGreaterEqual9dropout_875/dropout/random_uniform/RandomUniform:output:0+dropout_875/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������
�`
dropout_875/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_875/dropout/SelectV2SelectV2$dropout_875/dropout/GreaterEqual:z:0dropout_875/dropout/Mul:z:0$dropout_875/dropout/Const_1:output:0*
T0*,
_output_shapes
:���������
��
add_1AddV2+layer_normalization_740/batchnorm/add_1:z:0%dropout_875/dropout/SelectV2:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_741/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_741/moments/meanMean	add_1:z:0?layer_normalization_741/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_741/moments/StopGradientStopGradient-layer_normalization_741/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_741/moments/SquaredDifferenceSquaredDifference	add_1:z:05layer_normalization_741/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_741/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_741/moments/varianceMean5layer_normalization_741/moments/SquaredDifference:z:0Clayer_normalization_741/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_741/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_741/batchnorm/addAddV21layer_normalization_741/moments/variance:output:00layer_normalization_741/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_741/batchnorm/RsqrtRsqrt)layer_normalization_741/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_741/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_741_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/mulMul+layer_normalization_741/batchnorm/Rsqrt:y:0<layer_normalization_741/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_1Mul	add_1:z:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_2Mul-layer_normalization_741/moments/mean:output:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_741/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_741_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/subSub8layer_normalization_741/batchnorm/ReadVariableOp:value:0+layer_normalization_741/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/add_1AddV2+layer_normalization_741/batchnorm/mul_1:z:0)layer_normalization_741/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
�
IdentityIdentity+layer_normalization_741/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������
��
NoOpNoOp1^layer_normalization_740/batchnorm/ReadVariableOp5^layer_normalization_740/batchnorm/mul/ReadVariableOp1^layer_normalization_741/batchnorm/ReadVariableOp5^layer_normalization_741/batchnorm/mul/ReadVariableOp@^multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp1^sequential_370/dense_2780/BiasAdd/ReadVariableOp3^sequential_370/dense_2780/Tensordot/ReadVariableOp1^sequential_370/dense_2781/BiasAdd/ReadVariableOp3^sequential_370/dense_2781/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������
�: : : : : : : : : : : : : : : : 2d
0layer_normalization_740/batchnorm/ReadVariableOp0layer_normalization_740/batchnorm/ReadVariableOp2l
4layer_normalization_740/batchnorm/mul/ReadVariableOp4layer_normalization_740/batchnorm/mul/ReadVariableOp2d
0layer_normalization_741/batchnorm/ReadVariableOp0layer_normalization_741/batchnorm/ReadVariableOp2l
4layer_normalization_741/batchnorm/mul/ReadVariableOp4layer_normalization_741/batchnorm/mul/ReadVariableOp2�
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp2d
0sequential_370/dense_2780/BiasAdd/ReadVariableOp0sequential_370/dense_2780/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2780/Tensordot/ReadVariableOp2sequential_370/dense_2780/Tensordot/ReadVariableOp2d
0sequential_370/dense_2781/BiasAdd/ReadVariableOp0sequential_370/dense_2781/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2781/Tensordot/ReadVariableOp2sequential_370/dense_2781/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
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
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177046
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176956s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177042:'#
!
_user_specified_name	5177040:'#
!
_user_specified_name	5177038:'#
!
_user_specified_name	5177036:'#
!
_user_specified_name	5177034:'#
!
_user_specified_name	5177032:'#
!
_user_specified_name	5177030:'#
!
_user_specified_name	5177028:'#
!
_user_specified_name	5177026:'#
!
_user_specified_name	5177024:'
#
!
_user_specified_name	5177022:'	#
!
_user_specified_name	5177020:'#
!
_user_specified_name	5177018:'#
!
_user_specified_name	5177016:'#
!
_user_specified_name	5177014:'#
!
_user_specified_name	5177012:'#
!
_user_specified_name	5177010:'#
!
_user_specified_name	5177008:'#
!
_user_specified_name	5177006:'#
!
_user_specified_name	5177004:T P
+
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
,__inference_dense_2782_layer_call_fn_5177234

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2782_layer_call_and_return_conditional_losses_5176648s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177230:'#
!
_user_specified_name	5177228:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
��
�@
#__inference__traced_restore_5178546
file_prefixT
Aassignvariableop_csmom_transformer_reranker_117_dense_2775_kernel:	�P
Aassignvariableop_1_csmom_transformer_reranker_117_dense_2775_bias:	��
sassignvariableop_2_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel:
���
qassignvariableop_3_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias:	��
sassignvariableop_4_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel:
���
qassignvariableop_5_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias:	��
sassignvariableop_6_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel:
���
qassignvariableop_7_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias:	��
sassignvariableop_8_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel:
���
qassignvariableop_9_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias:	�8
%assignvariableop_10_dense_2780_kernel:	�1
#assignvariableop_11_dense_2780_bias:8
%assignvariableop_12_dense_2781_kernel:	�2
#assignvariableop_13_dense_2781_bias:	�q
bassignvariableop_14_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma:	�p
aassignvariableop_15_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta:	�q
bassignvariableop_16_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma:	�p
aassignvariableop_17_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta:	�W
Dassignvariableop_18_csmom_transformer_reranker_117_dense_2782_kernel:	�P
Bassignvariableop_19_csmom_transformer_reranker_117_dense_2782_bias:'
assignvariableop_20_iteration:	 +
!assignvariableop_21_learning_rate: ^
Kassignvariableop_22_adam_m_csmom_transformer_reranker_117_dense_2775_kernel:	�^
Kassignvariableop_23_adam_v_csmom_transformer_reranker_117_dense_2775_kernel:	�X
Iassignvariableop_24_adam_m_csmom_transformer_reranker_117_dense_2775_bias:	�X
Iassignvariableop_25_adam_v_csmom_transformer_reranker_117_dense_2775_bias:	��
{assignvariableop_26_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel:
���
{assignvariableop_27_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernel:
���
yassignvariableop_28_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias:	��
yassignvariableop_29_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_bias:	��
{assignvariableop_30_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel:
���
{assignvariableop_31_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernel:
���
yassignvariableop_32_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias:	��
yassignvariableop_33_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_bias:	��
{assignvariableop_34_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel:
���
{assignvariableop_35_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernel:
���
yassignvariableop_36_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias:	��
yassignvariableop_37_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_bias:	��
{assignvariableop_38_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel:
���
{assignvariableop_39_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernel:
���
yassignvariableop_40_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias:	��
yassignvariableop_41_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_bias:	�?
,assignvariableop_42_adam_m_dense_2780_kernel:	�?
,assignvariableop_43_adam_v_dense_2780_kernel:	�8
*assignvariableop_44_adam_m_dense_2780_bias:8
*assignvariableop_45_adam_v_dense_2780_bias:?
,assignvariableop_46_adam_m_dense_2781_kernel:	�?
,assignvariableop_47_adam_v_dense_2781_kernel:	�9
*assignvariableop_48_adam_m_dense_2781_bias:	�9
*assignvariableop_49_adam_v_dense_2781_bias:	�x
iassignvariableop_50_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma:	�x
iassignvariableop_51_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gamma:	�w
hassignvariableop_52_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta:	�w
hassignvariableop_53_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_beta:	�x
iassignvariableop_54_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma:	�x
iassignvariableop_55_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gamma:	�w
hassignvariableop_56_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta:	�w
hassignvariableop_57_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_beta:	�^
Kassignvariableop_58_adam_m_csmom_transformer_reranker_117_dense_2782_kernel:	�^
Kassignvariableop_59_adam_v_csmom_transformer_reranker_117_dense_2782_kernel:	�W
Iassignvariableop_60_adam_m_csmom_transformer_reranker_117_dense_2782_bias:W
Iassignvariableop_61_adam_v_csmom_transformer_reranker_117_dense_2782_bias:#
assignvariableop_62_total: #
assignvariableop_63_count: 
identity_65��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpAassignvariableop_csmom_transformer_reranker_117_dense_2775_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpAassignvariableop_1_csmom_transformer_reranker_117_dense_2775_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpsassignvariableop_2_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpqassignvariableop_3_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpsassignvariableop_4_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpqassignvariableop_5_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpsassignvariableop_6_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpqassignvariableop_7_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpsassignvariableop_8_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpqassignvariableop_9_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp%assignvariableop_10_dense_2780_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_2780_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp%assignvariableop_12_dense_2781_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_2781_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpbassignvariableop_14_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpaassignvariableop_15_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpbassignvariableop_16_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpaassignvariableop_17_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpDassignvariableop_18_csmom_transformer_reranker_117_dense_2782_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpBassignvariableop_19_csmom_transformer_reranker_117_dense_2782_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_iterationIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpKassignvariableop_22_adam_m_csmom_transformer_reranker_117_dense_2775_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpKassignvariableop_23_adam_v_csmom_transformer_reranker_117_dense_2775_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpIassignvariableop_24_adam_m_csmom_transformer_reranker_117_dense_2775_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpIassignvariableop_25_adam_v_csmom_transformer_reranker_117_dense_2775_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp{assignvariableop_26_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp{assignvariableop_27_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpyassignvariableop_28_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpyassignvariableop_29_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp{assignvariableop_30_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp{assignvariableop_31_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpyassignvariableop_32_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpyassignvariableop_33_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp{assignvariableop_34_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp{assignvariableop_35_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpyassignvariableop_36_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpyassignvariableop_37_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp{assignvariableop_38_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp{assignvariableop_39_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpyassignvariableop_40_adam_m_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpyassignvariableop_41_adam_v_csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_m_dense_2780_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_v_dense_2780_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_m_dense_2780_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_v_dense_2780_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_m_dense_2781_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_v_dense_2781_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_dense_2781_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_dense_2781_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpiassignvariableop_50_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpiassignvariableop_51_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_gammaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOphassignvariableop_52_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_betaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOphassignvariableop_53_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_betaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpiassignvariableop_54_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gammaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpiassignvariableop_55_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_gammaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOphassignvariableop_56_adam_m_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_betaIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOphassignvariableop_57_adam_v_csmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_betaIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpKassignvariableop_58_adam_m_csmom_transformer_reranker_117_dense_2782_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpKassignvariableop_59_adam_v_csmom_transformer_reranker_117_dense_2782_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpIassignvariableop_60_adam_m_csmom_transformer_reranker_117_dense_2782_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpIassignvariableop_61_adam_v_csmom_transformer_reranker_117_dense_2782_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_totalIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_countIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_65IdentityIdentity_64:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_65Identity_65:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%@!

_user_specified_namecount:%?!

_user_specified_nametotal:U>Q
O
_user_specified_name75Adam/v/csmom_transformer_reranker_117/dense_2782/bias:U=Q
O
_user_specified_name75Adam/m/csmom_transformer_reranker_117/dense_2782/bias:W<S
Q
_user_specified_name97Adam/v/csmom_transformer_reranker_117/dense_2782/kernel:W;S
Q
_user_specified_name97Adam/m/csmom_transformer_reranker_117/dense_2782/kernel:t:p
n
_user_specified_nameVTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta:t9p
n
_user_specified_nameVTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta:u8q
o
_user_specified_nameWUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma:u7q
o
_user_specified_nameWUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma:t6p
n
_user_specified_nameVTAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta:t5p
n
_user_specified_nameVTAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta:u4q
o
_user_specified_nameWUAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma:u3q
o
_user_specified_nameWUAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma:622
0
_user_specified_nameAdam/v/dense_2781/bias:612
0
_user_specified_nameAdam/m/dense_2781/bias:804
2
_user_specified_nameAdam/v/dense_2781/kernel:8/4
2
_user_specified_nameAdam/m/dense_2781/kernel:6.2
0
_user_specified_nameAdam/v/dense_2780/bias:6-2
0
_user_specified_nameAdam/m/dense_2780/bias:8,4
2
_user_specified_nameAdam/v/dense_2780/kernel:8+4
2
_user_specified_nameAdam/m/dense_2780/kernel:�*�

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias:�)�

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias:�(�
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel:�'�
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel:�&�

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias:�%�

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias:�$�
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel:�#�
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel:�"�

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias:�!�

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias:� �
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel:��
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel:��

_user_specified_namegeAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias:��

_user_specified_namegeAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias:��
�
_user_specified_nameigAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel:��
�
_user_specified_nameigAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel:UQ
O
_user_specified_name75Adam/v/csmom_transformer_reranker_117/dense_2775/bias:UQ
O
_user_specified_name75Adam/m/csmom_transformer_reranker_117/dense_2775/bias:WS
Q
_user_specified_name97Adam/v/csmom_transformer_reranker_117/dense_2775/kernel:WS
Q
_user_specified_name97Adam/m/csmom_transformer_reranker_117/dense_2775/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:NJ
H
_user_specified_name0.csmom_transformer_reranker_117/dense_2782/bias:PL
J
_user_specified_name20csmom_transformer_reranker_117/dense_2782/kernel:mi
g
_user_specified_nameOMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta:nj
h
_user_specified_namePNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma:mi
g
_user_specified_nameOMcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta:nj
h
_user_specified_namePNcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma:/+
)
_user_specified_namedense_2781/bias:1-
+
_user_specified_namedense_2781/kernel:/+
)
_user_specified_namedense_2780/bias:1-
+
_user_specified_namedense_2780/kernel:~
z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias:�	|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel:~z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias:�|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel:~z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias:�|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel:~z
x
_user_specified_name`^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias:�|
z
_user_specified_nameb`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel:NJ
H
_user_specified_name0.csmom_transformer_reranker_117/dense_2775/bias:PL
J
_user_specified_name20csmom_transformer_reranker_117/dense_2775/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
G__inference_dense_2780_layer_call_and_return_conditional_losses_5176180

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������
V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
G__inference_dense_2780_layer_call_and_return_conditional_losses_5177900

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������
e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������
V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�!
�
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176655
input_1%
dense_2775_5176313:	�!
dense_2775_5176315:	�-
encoder_layer_370_5176586:
��(
encoder_layer_370_5176588:	�-
encoder_layer_370_5176590:
��(
encoder_layer_370_5176592:	�-
encoder_layer_370_5176594:
��(
encoder_layer_370_5176596:	�-
encoder_layer_370_5176598:
��(
encoder_layer_370_5176600:	�(
encoder_layer_370_5176602:	�(
encoder_layer_370_5176604:	�,
encoder_layer_370_5176606:	�'
encoder_layer_370_5176608:,
encoder_layer_370_5176610:	�(
encoder_layer_370_5176612:	�(
encoder_layer_370_5176614:	�(
encoder_layer_370_5176616:	�%
dense_2782_5176649:	� 
dense_2782_5176651:
identity��"dense_2775/StatefulPartitionedCall�"dense_2782/StatefulPartitionedCall�)encoder_layer_370/StatefulPartitionedCall�
"dense_2775/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2775_5176313dense_2775_5176315*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2775_layer_call_and_return_conditional_losses_5176312�
)encoder_layer_370/StatefulPartitionedCallStatefulPartitionedCall+dense_2775/StatefulPartitionedCall:output:0encoder_layer_370_5176586encoder_layer_370_5176588encoder_layer_370_5176590encoder_layer_370_5176592encoder_layer_370_5176594encoder_layer_370_5176596encoder_layer_370_5176598encoder_layer_370_5176600encoder_layer_370_5176602encoder_layer_370_5176604encoder_layer_370_5176606encoder_layer_370_5176608encoder_layer_370_5176610encoder_layer_370_5176612encoder_layer_370_5176614encoder_layer_370_5176616*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5176585�
"dense_2782/StatefulPartitionedCallStatefulPartitionedCall2encoder_layer_370/StatefulPartitionedCall:output:0dense_2782_5176649dense_2782_5176651*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2782_layer_call_and_return_conditional_losses_5176648~
IdentityIdentity+dense_2782/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
�
NoOpNoOp#^dense_2775/StatefulPartitionedCall#^dense_2782/StatefulPartitionedCall*^encoder_layer_370/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������
: : : : : : : : : : : : : : : : : : : : 2H
"dense_2775/StatefulPartitionedCall"dense_2775/StatefulPartitionedCall2H
"dense_2782/StatefulPartitionedCall"dense_2782/StatefulPartitionedCall2V
)encoder_layer_370/StatefulPartitionedCall)encoder_layer_370/StatefulPartitionedCall:'#
!
_user_specified_name	5176651:'#
!
_user_specified_name	5176649:'#
!
_user_specified_name	5176616:'#
!
_user_specified_name	5176614:'#
!
_user_specified_name	5176612:'#
!
_user_specified_name	5176610:'#
!
_user_specified_name	5176608:'#
!
_user_specified_name	5176606:'#
!
_user_specified_name	5176604:'#
!
_user_specified_name	5176602:'
#
!
_user_specified_name	5176600:'	#
!
_user_specified_name	5176598:'#
!
_user_specified_name	5176596:'#
!
_user_specified_name	5176594:'#
!
_user_specified_name	5176592:'#
!
_user_specified_name	5176590:'#
!
_user_specified_name	5176588:'#
!
_user_specified_name	5176586:'#
!
_user_specified_name	5176315:'#
!
_user_specified_name	5176313:T P
+
_output_shapes
:���������

!
_user_specified_name	input_1
�	
�
0__inference_sequential_370_layer_call_fn_5176249
dense_2780_input
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_2780_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176222t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������
�: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5176245:'#
!
_user_specified_name	5176243:'#
!
_user_specified_name	5176241:'#
!
_user_specified_name	5176239:^ Z
,
_output_shapes
:���������
�
*
_user_specified_namedense_2780_input
��
�#
"__inference__wrapped_model_5176147
input_1^
Kcsmom_transformer_reranker_117_dense_2775_tensordot_readvariableop_resource:	�X
Icsmom_transformer_reranker_117_dense_2775_biasadd_readvariableop_resource:	��
{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_tensordot_readvariableop_resource:
���
ycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_biasadd_readvariableop_resource:	��
{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_tensordot_readvariableop_resource:
���
ycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_biasadd_readvariableop_resource:	��
{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_tensordot_readvariableop_resource:
���
ycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_biasadd_readvariableop_resource:	��
{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_tensordot_readvariableop_resource:
���
ycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_biasadd_readvariableop_resource:	�}
ncsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_batchnorm_mul_readvariableop_resource:	�y
jcsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_batchnorm_readvariableop_resource:	�
lcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2780_tensordot_readvariableop_resource:	�x
jcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2780_biasadd_readvariableop_resource:
lcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2781_tensordot_readvariableop_resource:	�y
jcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2781_biasadd_readvariableop_resource:	�}
ncsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_batchnorm_mul_readvariableop_resource:	�y
jcsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_batchnorm_readvariableop_resource:	�^
Kcsmom_transformer_reranker_117_dense_2782_tensordot_readvariableop_resource:	�W
Icsmom_transformer_reranker_117_dense_2782_biasadd_readvariableop_resource:
identity��@csmom_transformer_reranker_117/dense_2775/BiasAdd/ReadVariableOp�Bcsmom_transformer_reranker_117/dense_2775/Tensordot/ReadVariableOp�@csmom_transformer_reranker_117/dense_2782/BiasAdd/ReadVariableOp�Bcsmom_transformer_reranker_117/dense_2782/Tensordot/ReadVariableOp�acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/ReadVariableOp�ecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul/ReadVariableOp�acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/ReadVariableOp�ecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul/ReadVariableOp�pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp�rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp�pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp�rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp�pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp�rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp�pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp�rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp�acsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAdd/ReadVariableOp�ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ReadVariableOp�acsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAdd/ReadVariableOp�ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ReadVariableOp�
Bcsmom_transformer_reranker_117/dense_2775/Tensordot/ReadVariableOpReadVariableOpKcsmom_transformer_reranker_117_dense_2775_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8csmom_transformer_reranker_117/dense_2775/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
8csmom_transformer_reranker_117/dense_2775/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
9csmom_transformer_reranker_117/dense_2775/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
::���
Acsmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<csmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2GatherV2Bcsmom_transformer_reranker_117/dense_2775/Tensordot/Shape:output:0Acsmom_transformer_reranker_117/dense_2775/Tensordot/free:output:0Jcsmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ccsmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>csmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2_1GatherV2Bcsmom_transformer_reranker_117/dense_2775/Tensordot/Shape:output:0Acsmom_transformer_reranker_117/dense_2775/Tensordot/axes:output:0Lcsmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
9csmom_transformer_reranker_117/dense_2775/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
8csmom_transformer_reranker_117/dense_2775/Tensordot/ProdProdEcsmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2:output:0Bcsmom_transformer_reranker_117/dense_2775/Tensordot/Const:output:0*
T0*
_output_shapes
: �
;csmom_transformer_reranker_117/dense_2775/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
:csmom_transformer_reranker_117/dense_2775/Tensordot/Prod_1ProdGcsmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2_1:output:0Dcsmom_transformer_reranker_117/dense_2775/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
?csmom_transformer_reranker_117/dense_2775/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:csmom_transformer_reranker_117/dense_2775/Tensordot/concatConcatV2Acsmom_transformer_reranker_117/dense_2775/Tensordot/free:output:0Acsmom_transformer_reranker_117/dense_2775/Tensordot/axes:output:0Hcsmom_transformer_reranker_117/dense_2775/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
9csmom_transformer_reranker_117/dense_2775/Tensordot/stackPackAcsmom_transformer_reranker_117/dense_2775/Tensordot/Prod:output:0Ccsmom_transformer_reranker_117/dense_2775/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
=csmom_transformer_reranker_117/dense_2775/Tensordot/transpose	Transposeinput_1Ccsmom_transformer_reranker_117/dense_2775/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
;csmom_transformer_reranker_117/dense_2775/Tensordot/ReshapeReshapeAcsmom_transformer_reranker_117/dense_2775/Tensordot/transpose:y:0Bcsmom_transformer_reranker_117/dense_2775/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:csmom_transformer_reranker_117/dense_2775/Tensordot/MatMulMatMulDcsmom_transformer_reranker_117/dense_2775/Tensordot/Reshape:output:0Jcsmom_transformer_reranker_117/dense_2775/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
;csmom_transformer_reranker_117/dense_2775/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Acsmom_transformer_reranker_117/dense_2775/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<csmom_transformer_reranker_117/dense_2775/Tensordot/concat_1ConcatV2Ecsmom_transformer_reranker_117/dense_2775/Tensordot/GatherV2:output:0Dcsmom_transformer_reranker_117/dense_2775/Tensordot/Const_2:output:0Jcsmom_transformer_reranker_117/dense_2775/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
3csmom_transformer_reranker_117/dense_2775/TensordotReshapeDcsmom_transformer_reranker_117/dense_2775/Tensordot/MatMul:product:0Ecsmom_transformer_reranker_117/dense_2775/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
@csmom_transformer_reranker_117/dense_2775/BiasAdd/ReadVariableOpReadVariableOpIcsmom_transformer_reranker_117_dense_2775_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1csmom_transformer_reranker_117/dense_2775/BiasAddBiasAdd<csmom_transformer_reranker_117/dense_2775/Tensordot:output:0Hcsmom_transformer_reranker_117/dense_2775/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpReadVariableOp{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ShapeShape:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0*
T0*
_output_shapes
::���
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/free:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ncsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0|csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ProdProducsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Const:output:0*
T0*
_output_shapes
: �
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Prod_1Prodwcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concatConcatV2qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/free:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/stackPackqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Prod:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/transpose	Transpose:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ReshapeReshapeqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/transpose:y:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/MatMulMatMultcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Reshape:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concat_1ConcatV2ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/Const_2:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
ccsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/TensordotReshapetcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/MatMul:product:0ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpReadVariableOpycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2776_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
acsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAddBiasAddlcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpReadVariableOp{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ShapeShape:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0*
T0*
_output_shapes
::���
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/free:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ncsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0|csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ProdProducsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Const:output:0*
T0*
_output_shapes
: �
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Prod_1Prodwcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concatConcatV2qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/free:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/stackPackqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Prod:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/transpose	Transpose:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ReshapeReshapeqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/transpose:y:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/MatMulMatMultcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Reshape:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concat_1ConcatV2ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/Const_2:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
ccsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/TensordotReshapetcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/MatMul:product:0ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpReadVariableOpycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2777_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
acsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAddBiasAddlcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpReadVariableOp{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ShapeShape:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0*
T0*
_output_shapes
::���
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/free:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ncsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0|csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ProdProducsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Const:output:0*
T0*
_output_shapes
: �
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Prod_1Prodwcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concatConcatV2qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/free:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/stackPackqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Prod:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/transpose	Transpose:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ReshapeReshapeqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/transpose:y:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/MatMulMatMultcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Reshape:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concat_1ConcatV2ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/Const_2:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
ccsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/TensordotReshapetcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/MatMul:product:0ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpReadVariableOpycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2778_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
acsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAddBiasAddlcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/ShapeShapejcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd:output:0*
T0*
_output_shapes
::���
bcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
dcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
dcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
\csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_sliceStridedSlice]csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Shape:output:0kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice/stack:output:0mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice/stack_1:output:0mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
����������
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
\csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shapePackecsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice:output:0gcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shape/1:output:0gcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shape/2:output:0gcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
Vcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/ReshapeReshapejcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd:output:0ecsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
]csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
Xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose	Transpose_csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape:output:0fcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
Vcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Shape_1Shapejcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd:output:0*
T0*
_output_shapes
::���
dcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
fcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
fcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1StridedSlice_csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Shape_1:output:0mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1/stack:output:0ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1/stack_1:output:0ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
����������
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shapePackgcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_1:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shape/1:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shape/2:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:�
Xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1Reshapejcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd:output:0gcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
_csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
Zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_1	Transposeacsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_1:output:0hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_1/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
Vcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Shape_2Shapejcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd:output:0*
T0*
_output_shapes
::���
dcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
fcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
fcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2StridedSlice_csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Shape_2:output:0mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2/stack:output:0ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2/stack_1:output:0ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
����������
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shapePackgcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_2:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shape/1:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shape/2:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:�
Xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2Reshapejcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd:output:0gcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
_csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
Zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_2	Transposeacsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_2:output:0hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_2/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
Ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/MatMulBatchMatMulV2\csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose:y:0^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_1:y:0*
T0*A
_output_shapes/
-:+���������������������������*
adj_y(�
Ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
Scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/SqrtSqrt^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Sqrt/x:output:0*
T0*
_output_shapes
: �
Vcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/truedivRealDiv^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/MatMul:output:0Wcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
Vcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/SoftmaxSoftmaxZcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
Wcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/MatMul_1BatchMatMulV2`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Softmax:softmax:0^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_2:y:0*
T0*9
_output_shapes'
%:#��������������������
_csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
Zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_3	Transpose`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/MatMul_1:output:0hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_3/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
Vcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Shape_3Shape:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0*
T0*
_output_shapes
::���
dcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: �
fcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
fcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3StridedSlice_csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Shape_3:output:0mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3/stack:output:0ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3/stack_1:output:0ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
����������
`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3/shapePackgcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/strided_slice_3:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3/shape/1:output:0icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:�
Xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3Reshape^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/transpose_3:y:0gcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpReadVariableOp{csmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ShapeShapeacsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3:output:0*
T0*
_output_shapes
::���
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/free:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ncsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1GatherV2rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Shape:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0|csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
hcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ProdProducsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Const:output:0*
T0*
_output_shapes
: �
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Prod_1Prodwcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
ocsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concatConcatV2qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/free:output:0qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
icsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/stackPackqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Prod:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
mcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/transpose	Transposeacsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/Reshape_3:output:0scsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ReshapeReshapeqcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/transpose:y:0rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
jcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/MatMulMatMultcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Reshape:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
kcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
qcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
lcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concat_1ConcatV2ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0tcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/Const_2:output:0zcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
ccsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/TensordotReshapetcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/MatMul:product:0ucsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpReadVariableOpycsmom_transformer_reranker_117_encoder_layer_370_multi_head_self_attention_370_dense_2779_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
acsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAddBiasAddlcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot:output:0xcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Ecsmom_transformer_reranker_117/encoder_layer_370/dropout_874/IdentityIdentityjcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
4csmom_transformer_reranker_117/encoder_layer_370/addAddV2:csmom_transformer_reranker_117/dense_2775/BiasAdd:output:0Ncsmom_transformer_reranker_117/encoder_layer_370/dropout_874/Identity:output:0*
T0*,
_output_shapes
:���������
��
gcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Ucsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/meanMean8csmom_transformer_reranker_117/encoder_layer_370/add:z:0pcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
]csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/StopGradientStopGradient^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
bcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/SquaredDifferenceSquaredDifference8csmom_transformer_reranker_117/encoder_layer_370/add:z:0fcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
kcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Ycsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/varianceMeanfcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/SquaredDifference:z:0tcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
Vcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/addAddV2bcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/variance:output:0acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/RsqrtRsqrtZcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
ecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul/ReadVariableOpReadVariableOpncsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Vcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mulMul\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/Rsqrt:y:0mcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul_1Mul8csmom_transformer_reranker_117/encoder_layer_370/add:z:0Zcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul_2Mul^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/moments/mean:output:0Zcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/ReadVariableOpReadVariableOpjcsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_740_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Vcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/subSubicsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/ReadVariableOp:value:0\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/add_1AddV2\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul_1:z:0Zcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
��
ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ReadVariableOpReadVariableOplcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2780_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Ycsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Ycsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Zcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ShapeShape\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2GatherV2ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Shape:output:0bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/free:output:0kcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
dcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
_csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2_1GatherV2ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Shape:output:0bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/axes:output:0mcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Zcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Ycsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ProdProdfcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2:output:0ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Const:output:0*
T0*
_output_shapes
: �
\csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Prod_1Prodhcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2_1:output:0ecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
`csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concatConcatV2bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/free:output:0bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/axes:output:0icsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Zcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/stackPackbcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Prod:output:0dcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
^csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/transpose	Transpose\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/add_1:z:0dcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
\csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ReshapeReshapebcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/transpose:y:0ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/MatMulMatMulecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Reshape:output:0kcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
\csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concat_1ConcatV2fcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/GatherV2:output:0ecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/Const_2:output:0kcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/TensordotReshapeecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/MatMul:product:0fcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
�
acsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAdd/ReadVariableOpReadVariableOpjcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2780_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Rcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAddBiasAdd]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot:output:0icsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
�
Ocsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/ReluRelu[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAdd:output:0*
T0*+
_output_shapes
:���������
�
ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ReadVariableOpReadVariableOplcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2781_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Ycsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Ycsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Zcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ShapeShape]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Relu:activations:0*
T0*
_output_shapes
::���
bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2GatherV2ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Shape:output:0bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/free:output:0kcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
dcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
_csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2_1GatherV2ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Shape:output:0bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/axes:output:0mcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Zcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Ycsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ProdProdfcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2:output:0ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Const:output:0*
T0*
_output_shapes
: �
\csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Prod_1Prodhcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2_1:output:0ecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
`csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concatConcatV2bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/free:output:0bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/axes:output:0icsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Zcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/stackPackbcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Prod:output:0dcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
^csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/transpose	Transpose]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Relu:activations:0dcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
\csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ReshapeReshapebcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/transpose:y:0ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/MatMulMatMulecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Reshape:output:0kcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
\csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
bcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concat_1ConcatV2fcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/GatherV2:output:0ecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/Const_2:output:0kcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/TensordotReshapeecsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/MatMul:product:0fcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
acsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAdd/ReadVariableOpReadVariableOpjcsmom_transformer_reranker_117_encoder_layer_370_sequential_370_dense_2781_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Rcsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAddBiasAdd]csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot:output:0icsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Ecsmom_transformer_reranker_117/encoder_layer_370/dropout_875/IdentityIdentity[csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAdd:output:0*
T0*,
_output_shapes
:���������
��
6csmom_transformer_reranker_117/encoder_layer_370/add_1AddV2\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/add_1:z:0Ncsmom_transformer_reranker_117/encoder_layer_370/dropout_875/Identity:output:0*
T0*,
_output_shapes
:���������
��
gcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Ucsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/meanMean:csmom_transformer_reranker_117/encoder_layer_370/add_1:z:0pcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
]csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/StopGradientStopGradient^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
bcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/SquaredDifferenceSquaredDifference:csmom_transformer_reranker_117/encoder_layer_370/add_1:z:0fcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
kcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Ycsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/varianceMeanfcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/SquaredDifference:z:0tcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
Vcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/addAddV2bcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/variance:output:0acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/RsqrtRsqrtZcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
ecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul/ReadVariableOpReadVariableOpncsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Vcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mulMul\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/Rsqrt:y:0mcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul_1Mul:csmom_transformer_reranker_117/encoder_layer_370/add_1:z:0Zcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul_2Mul^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/moments/mean:output:0Zcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/ReadVariableOpReadVariableOpjcsmom_transformer_reranker_117_encoder_layer_370_layer_normalization_741_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Vcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/subSubicsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/ReadVariableOp:value:0\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
Xcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/add_1AddV2\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul_1:z:0Zcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
��
Bcsmom_transformer_reranker_117/dense_2782/Tensordot/ReadVariableOpReadVariableOpKcsmom_transformer_reranker_117_dense_2782_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
8csmom_transformer_reranker_117/dense_2782/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
8csmom_transformer_reranker_117/dense_2782/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
9csmom_transformer_reranker_117/dense_2782/Tensordot/ShapeShape\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/add_1:z:0*
T0*
_output_shapes
::���
Acsmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<csmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2GatherV2Bcsmom_transformer_reranker_117/dense_2782/Tensordot/Shape:output:0Acsmom_transformer_reranker_117/dense_2782/Tensordot/free:output:0Jcsmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ccsmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>csmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2_1GatherV2Bcsmom_transformer_reranker_117/dense_2782/Tensordot/Shape:output:0Acsmom_transformer_reranker_117/dense_2782/Tensordot/axes:output:0Lcsmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
9csmom_transformer_reranker_117/dense_2782/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
8csmom_transformer_reranker_117/dense_2782/Tensordot/ProdProdEcsmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2:output:0Bcsmom_transformer_reranker_117/dense_2782/Tensordot/Const:output:0*
T0*
_output_shapes
: �
;csmom_transformer_reranker_117/dense_2782/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
:csmom_transformer_reranker_117/dense_2782/Tensordot/Prod_1ProdGcsmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2_1:output:0Dcsmom_transformer_reranker_117/dense_2782/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
?csmom_transformer_reranker_117/dense_2782/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:csmom_transformer_reranker_117/dense_2782/Tensordot/concatConcatV2Acsmom_transformer_reranker_117/dense_2782/Tensordot/free:output:0Acsmom_transformer_reranker_117/dense_2782/Tensordot/axes:output:0Hcsmom_transformer_reranker_117/dense_2782/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
9csmom_transformer_reranker_117/dense_2782/Tensordot/stackPackAcsmom_transformer_reranker_117/dense_2782/Tensordot/Prod:output:0Ccsmom_transformer_reranker_117/dense_2782/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
=csmom_transformer_reranker_117/dense_2782/Tensordot/transpose	Transpose\csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/add_1:z:0Ccsmom_transformer_reranker_117/dense_2782/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
;csmom_transformer_reranker_117/dense_2782/Tensordot/ReshapeReshapeAcsmom_transformer_reranker_117/dense_2782/Tensordot/transpose:y:0Bcsmom_transformer_reranker_117/dense_2782/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
:csmom_transformer_reranker_117/dense_2782/Tensordot/MatMulMatMulDcsmom_transformer_reranker_117/dense_2782/Tensordot/Reshape:output:0Jcsmom_transformer_reranker_117/dense_2782/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;csmom_transformer_reranker_117/dense_2782/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
Acsmom_transformer_reranker_117/dense_2782/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<csmom_transformer_reranker_117/dense_2782/Tensordot/concat_1ConcatV2Ecsmom_transformer_reranker_117/dense_2782/Tensordot/GatherV2:output:0Dcsmom_transformer_reranker_117/dense_2782/Tensordot/Const_2:output:0Jcsmom_transformer_reranker_117/dense_2782/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
3csmom_transformer_reranker_117/dense_2782/TensordotReshapeDcsmom_transformer_reranker_117/dense_2782/Tensordot/MatMul:product:0Ecsmom_transformer_reranker_117/dense_2782/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
�
@csmom_transformer_reranker_117/dense_2782/BiasAdd/ReadVariableOpReadVariableOpIcsmom_transformer_reranker_117_dense_2782_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
1csmom_transformer_reranker_117/dense_2782/BiasAddBiasAdd<csmom_transformer_reranker_117/dense_2782/Tensordot:output:0Hcsmom_transformer_reranker_117/dense_2782/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
�
IdentityIdentity:csmom_transformer_reranker_117/dense_2782/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������
�
NoOpNoOpA^csmom_transformer_reranker_117/dense_2775/BiasAdd/ReadVariableOpC^csmom_transformer_reranker_117/dense_2775/Tensordot/ReadVariableOpA^csmom_transformer_reranker_117/dense_2782/BiasAdd/ReadVariableOpC^csmom_transformer_reranker_117/dense_2782/Tensordot/ReadVariableOpb^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/ReadVariableOpf^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul/ReadVariableOpb^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/ReadVariableOpf^csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul/ReadVariableOpq^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOps^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpq^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOps^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpq^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOps^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpq^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOps^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpb^csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAdd/ReadVariableOpd^csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ReadVariableOpb^csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAdd/ReadVariableOpd^csmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������
: : : : : : : : : : : : : : : : : : : : 2�
@csmom_transformer_reranker_117/dense_2775/BiasAdd/ReadVariableOp@csmom_transformer_reranker_117/dense_2775/BiasAdd/ReadVariableOp2�
Bcsmom_transformer_reranker_117/dense_2775/Tensordot/ReadVariableOpBcsmom_transformer_reranker_117/dense_2775/Tensordot/ReadVariableOp2�
@csmom_transformer_reranker_117/dense_2782/BiasAdd/ReadVariableOp@csmom_transformer_reranker_117/dense_2782/BiasAdd/ReadVariableOp2�
Bcsmom_transformer_reranker_117/dense_2782/Tensordot/ReadVariableOpBcsmom_transformer_reranker_117/dense_2782/Tensordot/ReadVariableOp2�
acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/ReadVariableOpacsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/ReadVariableOp2�
ecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul/ReadVariableOpecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/batchnorm/mul/ReadVariableOp2�
acsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/ReadVariableOpacsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/ReadVariableOp2�
ecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul/ReadVariableOpecsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/batchnorm/mul/ReadVariableOp2�
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOppcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp2�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOprcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp2�
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOppcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp2�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOprcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp2�
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOppcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp2�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOprcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp2�
pcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOppcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp2�
rcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOprcsmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp2�
acsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAdd/ReadVariableOpacsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/BiasAdd/ReadVariableOp2�
ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ReadVariableOpccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2780/Tensordot/ReadVariableOp2�
acsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAdd/ReadVariableOpacsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/BiasAdd/ReadVariableOp2�
ccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ReadVariableOpccsmom_transformer_reranker_117/encoder_layer_370/sequential_370/dense_2781/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
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
resource:T P
+
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
%__inference_signature_wrapper_5177186
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_5176147s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177182:'#
!
_user_specified_name	5177180:'#
!
_user_specified_name	5177178:'#
!
_user_specified_name	5177176:'#
!
_user_specified_name	5177174:'#
!
_user_specified_name	5177172:'#
!
_user_specified_name	5177170:'#
!
_user_specified_name	5177168:'#
!
_user_specified_name	5177166:'#
!
_user_specified_name	5177164:'
#
!
_user_specified_name	5177162:'	#
!
_user_specified_name	5177160:'#
!
_user_specified_name	5177158:'#
!
_user_specified_name	5177156:'#
!
_user_specified_name	5177154:'#
!
_user_specified_name	5177152:'#
!
_user_specified_name	5177150:'#
!
_user_specified_name	5177148:'#
!
_user_specified_name	5177146:'#
!
_user_specified_name	5177144:T P
+
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
,__inference_dense_2780_layer_call_fn_5177869

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2780_layer_call_and_return_conditional_losses_5176180s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177865:'#
!
_user_specified_name	5177863:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
G__inference_dense_2782_layer_call_and_return_conditional_losses_5177264

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������
V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
G__inference_dense_2775_layer_call_and_return_conditional_losses_5176312

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������
�V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5177860

inputs^
Jmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource:	�L
=layer_normalization_740_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_740_batchnorm_readvariableop_resource:	�N
;sequential_370_dense_2780_tensordot_readvariableop_resource:	�G
9sequential_370_dense_2780_biasadd_readvariableop_resource:N
;sequential_370_dense_2781_tensordot_readvariableop_resource:	�H
9sequential_370_dense_2781_biasadd_readvariableop_resource:	�L
=layer_normalization_741_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_741_batchnorm_readvariableop_resource:	�
identity��0layer_normalization_740/batchnorm/ReadVariableOp�4layer_normalization_740/batchnorm/mul/ReadVariableOp�0layer_normalization_741/batchnorm/ReadVariableOp�4layer_normalization_741/batchnorm/mul/ReadVariableOp�?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp�0sequential_370/dense_2780/BiasAdd/ReadVariableOp�2sequential_370/dense_2780/Tensordot/ReadVariableOp�0sequential_370/dense_2781/BiasAdd/ReadVariableOp�2sequential_370/dense_2781/Tensordot/ReadVariableOp�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2776/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2776/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2776/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2776/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2776/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2776/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2776/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2776/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2776/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2776/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2776/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/stackPack@multi_head_self_attention_370/dense_2776/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2776/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2776/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2776/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2776/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2776/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2776/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2776/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2776/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2776/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2776/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2776/TensordotReshapeCmulti_head_self_attention_370/dense_2776/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2776/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2776/BiasAddBiasAdd;multi_head_self_attention_370/dense_2776/Tensordot:output:0Gmulti_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2777/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2777/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2777/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2777/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2777/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2777/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2777/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2777/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2777/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2777/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2777/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/stackPack@multi_head_self_attention_370/dense_2777/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2777/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2777/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2777/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2777/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2777/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2777/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2777/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2777/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2777/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2777/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2777/TensordotReshapeCmulti_head_self_attention_370/dense_2777/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2777/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2777/BiasAddBiasAdd;multi_head_self_attention_370/dense_2777/Tensordot:output:0Gmulti_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2778/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2778/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2778/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2778/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2778/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2778/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2778/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2778/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2778/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2778/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2778/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/stackPack@multi_head_self_attention_370/dense_2778/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2778/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2778/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2778/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2778/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2778/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2778/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2778/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2778/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2778/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2778/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2778/TensordotReshapeCmulti_head_self_attention_370/dense_2778/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2778/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2778/BiasAddBiasAdd;multi_head_self_attention_370/dense_2778/Tensordot:output:0Gmulti_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
#multi_head_self_attention_370/ShapeShape9multi_head_self_attention_370/dense_2776/BiasAdd:output:0*
T0*
_output_shapes
::��{
1multi_head_self_attention_370/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_370/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3multi_head_self_attention_370/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+multi_head_self_attention_370/strided_sliceStridedSlice,multi_head_self_attention_370/Shape:output:0:multi_head_self_attention_370/strided_slice/stack:output:0<multi_head_self_attention_370/strided_slice/stack_1:output:0<multi_head_self_attention_370/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-multi_head_self_attention_370/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������o
-multi_head_self_attention_370/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p
-multi_head_self_attention_370/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
+multi_head_self_attention_370/Reshape/shapePack4multi_head_self_attention_370/strided_slice:output:06multi_head_self_attention_370/Reshape/shape/1:output:06multi_head_self_attention_370/Reshape/shape/2:output:06multi_head_self_attention_370/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
%multi_head_self_attention_370/ReshapeReshape9multi_head_self_attention_370/dense_2776/BiasAdd:output:04multi_head_self_attention_370/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
,multi_head_self_attention_370/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
'multi_head_self_attention_370/transpose	Transpose.multi_head_self_attention_370/Reshape:output:05multi_head_self_attention_370/transpose/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_1Shape9multi_head_self_attention_370/dense_2777/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_1StridedSlice.multi_head_self_attention_370/Shape_1:output:0<multi_head_self_attention_370/strided_slice_1/stack:output:0>multi_head_self_attention_370/strided_slice_1/stack_1:output:0>multi_head_self_attention_370/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_1/shapePack6multi_head_self_attention_370/strided_slice_1:output:08multi_head_self_attention_370/Reshape_1/shape/1:output:08multi_head_self_attention_370/Reshape_1/shape/2:output:08multi_head_self_attention_370/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_1Reshape9multi_head_self_attention_370/dense_2777/BiasAdd:output:06multi_head_self_attention_370/Reshape_1/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_1	Transpose0multi_head_self_attention_370/Reshape_1:output:07multi_head_self_attention_370/transpose_1/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_2Shape9multi_head_self_attention_370/dense_2778/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_2StridedSlice.multi_head_self_attention_370/Shape_2:output:0<multi_head_self_attention_370/strided_slice_2/stack:output:0>multi_head_self_attention_370/strided_slice_2/stack_1:output:0>multi_head_self_attention_370/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_2/shapePack6multi_head_self_attention_370/strided_slice_2:output:08multi_head_self_attention_370/Reshape_2/shape/1:output:08multi_head_self_attention_370/Reshape_2/shape/2:output:08multi_head_self_attention_370/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_2Reshape9multi_head_self_attention_370/dense_2778/BiasAdd:output:06multi_head_self_attention_370/Reshape_2/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_2	Transpose0multi_head_self_attention_370/Reshape_2:output:07multi_head_self_attention_370/transpose_2/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
$multi_head_self_attention_370/MatMulBatchMatMulV2+multi_head_self_attention_370/transpose:y:0-multi_head_self_attention_370/transpose_1:y:0*
T0*A
_output_shapes/
-:+���������������������������*
adj_y(i
$multi_head_self_attention_370/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *   Cz
"multi_head_self_attention_370/SqrtSqrt-multi_head_self_attention_370/Sqrt/x:output:0*
T0*
_output_shapes
: �
%multi_head_self_attention_370/truedivRealDiv-multi_head_self_attention_370/MatMul:output:0&multi_head_self_attention_370/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
%multi_head_self_attention_370/SoftmaxSoftmax)multi_head_self_attention_370/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
&multi_head_self_attention_370/MatMul_1BatchMatMulV2/multi_head_self_attention_370/Softmax:softmax:0-multi_head_self_attention_370/transpose_2:y:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_3	Transpose/multi_head_self_attention_370/MatMul_1:output:07multi_head_self_attention_370/transpose_3/perm:output:0*
T0*9
_output_shapes'
%:#�������������������i
%multi_head_self_attention_370/Shape_3Shapeinputs*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_3StridedSlice.multi_head_self_attention_370/Shape_3:output:0<multi_head_self_attention_370/strided_slice_3/stack:output:0>multi_head_self_attention_370/strided_slice_3/stack_1:output:0>multi_head_self_attention_370/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/multi_head_self_attention_370/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_3/shapePack6multi_head_self_attention_370/strided_slice_3:output:08multi_head_self_attention_370/Reshape_3/shape/1:output:08multi_head_self_attention_370/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_3Reshape-multi_head_self_attention_370/transpose_3:y:06multi_head_self_attention_370/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2779/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2779/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8multi_head_self_attention_370/dense_2779/Tensordot/ShapeShape0multi_head_self_attention_370/Reshape_3:output:0*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2779/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2779/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2779/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2779/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2779/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2779/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2779/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2779/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/stackPack@multi_head_self_attention_370/dense_2779/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2779/Tensordot/transpose	Transpose0multi_head_self_attention_370/Reshape_3:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
:multi_head_self_attention_370/dense_2779/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2779/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2779/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2779/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2779/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2779/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2779/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2779/TensordotReshapeCmulti_head_self_attention_370/dense_2779/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2779/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2779/BiasAddBiasAdd;multi_head_self_attention_370/dense_2779/Tensordot:output:0Gmulti_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
dropout_874/IdentityIdentity9multi_head_self_attention_370/dense_2779/BiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������j
addAddV2inputsdropout_874/Identity:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_740/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_740/moments/meanMeanadd:z:0?layer_normalization_740/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_740/moments/StopGradientStopGradient-layer_normalization_740/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_740/moments/SquaredDifferenceSquaredDifferenceadd:z:05layer_normalization_740/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_740/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_740/moments/varianceMean5layer_normalization_740/moments/SquaredDifference:z:0Clayer_normalization_740/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_740/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_740/batchnorm/addAddV21layer_normalization_740/moments/variance:output:00layer_normalization_740/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_740/batchnorm/RsqrtRsqrt)layer_normalization_740/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_740/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_740_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/mulMul+layer_normalization_740/batchnorm/Rsqrt:y:0<layer_normalization_740/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_1Muladd:z:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_2Mul-layer_normalization_740/moments/mean:output:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_740/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_740_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/subSub8layer_normalization_740/batchnorm/ReadVariableOp:value:0+layer_normalization_740/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/add_1AddV2+layer_normalization_740/batchnorm/mul_1:z:0)layer_normalization_740/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
��
2sequential_370/dense_2780/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2780_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2780/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2780/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2780/Tensordot/ShapeShape+layer_normalization_740/batchnorm/add_1:z:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2780/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/GatherV2GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/free:output:0:sequential_370/dense_2780/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2780/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2780/Tensordot/GatherV2_1GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/axes:output:0<sequential_370/dense_2780/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2780/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2780/Tensordot/ProdProd5sequential_370/dense_2780/Tensordot/GatherV2:output:02sequential_370/dense_2780/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2780/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2780/Tensordot/Prod_1Prod7sequential_370/dense_2780/Tensordot/GatherV2_1:output:04sequential_370/dense_2780/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2780/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2780/Tensordot/concatConcatV21sequential_370/dense_2780/Tensordot/free:output:01sequential_370/dense_2780/Tensordot/axes:output:08sequential_370/dense_2780/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2780/Tensordot/stackPack1sequential_370/dense_2780/Tensordot/Prod:output:03sequential_370/dense_2780/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2780/Tensordot/transpose	Transpose+layer_normalization_740/batchnorm/add_1:z:03sequential_370/dense_2780/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
+sequential_370/dense_2780/Tensordot/ReshapeReshape1sequential_370/dense_2780/Tensordot/transpose:y:02sequential_370/dense_2780/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2780/Tensordot/MatMulMatMul4sequential_370/dense_2780/Tensordot/Reshape:output:0:sequential_370/dense_2780/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
+sequential_370/dense_2780/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:s
1sequential_370/dense_2780/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/concat_1ConcatV25sequential_370/dense_2780/Tensordot/GatherV2:output:04sequential_370/dense_2780/Tensordot/Const_2:output:0:sequential_370/dense_2780/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2780/TensordotReshape4sequential_370/dense_2780/Tensordot/MatMul:product:05sequential_370/dense_2780/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
�
0sequential_370/dense_2780/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2780_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_370/dense_2780/BiasAddBiasAdd,sequential_370/dense_2780/Tensordot:output:08sequential_370/dense_2780/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
�
sequential_370/dense_2780/ReluRelu*sequential_370/dense_2780/BiasAdd:output:0*
T0*+
_output_shapes
:���������
�
2sequential_370/dense_2781/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2781_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2781/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2781/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2781/Tensordot/ShapeShape,sequential_370/dense_2780/Relu:activations:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2781/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/GatherV2GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/free:output:0:sequential_370/dense_2781/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2781/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2781/Tensordot/GatherV2_1GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/axes:output:0<sequential_370/dense_2781/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2781/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2781/Tensordot/ProdProd5sequential_370/dense_2781/Tensordot/GatherV2:output:02sequential_370/dense_2781/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2781/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2781/Tensordot/Prod_1Prod7sequential_370/dense_2781/Tensordot/GatherV2_1:output:04sequential_370/dense_2781/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2781/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2781/Tensordot/concatConcatV21sequential_370/dense_2781/Tensordot/free:output:01sequential_370/dense_2781/Tensordot/axes:output:08sequential_370/dense_2781/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2781/Tensordot/stackPack1sequential_370/dense_2781/Tensordot/Prod:output:03sequential_370/dense_2781/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2781/Tensordot/transpose	Transpose,sequential_370/dense_2780/Relu:activations:03sequential_370/dense_2781/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
+sequential_370/dense_2781/Tensordot/ReshapeReshape1sequential_370/dense_2781/Tensordot/transpose:y:02sequential_370/dense_2781/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2781/Tensordot/MatMulMatMul4sequential_370/dense_2781/Tensordot/Reshape:output:0:sequential_370/dense_2781/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
+sequential_370/dense_2781/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�s
1sequential_370/dense_2781/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/concat_1ConcatV25sequential_370/dense_2781/Tensordot/GatherV2:output:04sequential_370/dense_2781/Tensordot/Const_2:output:0:sequential_370/dense_2781/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2781/TensordotReshape4sequential_370/dense_2781/Tensordot/MatMul:product:05sequential_370/dense_2781/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
0sequential_370/dense_2781/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2781_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!sequential_370/dense_2781/BiasAddBiasAdd,sequential_370/dense_2781/Tensordot:output:08sequential_370/dense_2781/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
dropout_875/IdentityIdentity*sequential_370/dense_2781/BiasAdd:output:0*
T0*,
_output_shapes
:���������
��
add_1AddV2+layer_normalization_740/batchnorm/add_1:z:0dropout_875/Identity:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_741/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_741/moments/meanMean	add_1:z:0?layer_normalization_741/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_741/moments/StopGradientStopGradient-layer_normalization_741/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_741/moments/SquaredDifferenceSquaredDifference	add_1:z:05layer_normalization_741/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_741/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_741/moments/varianceMean5layer_normalization_741/moments/SquaredDifference:z:0Clayer_normalization_741/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_741/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_741/batchnorm/addAddV21layer_normalization_741/moments/variance:output:00layer_normalization_741/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_741/batchnorm/RsqrtRsqrt)layer_normalization_741/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_741/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_741_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/mulMul+layer_normalization_741/batchnorm/Rsqrt:y:0<layer_normalization_741/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_1Mul	add_1:z:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_2Mul-layer_normalization_741/moments/mean:output:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_741/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_741_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/subSub8layer_normalization_741/batchnorm/ReadVariableOp:value:0+layer_normalization_741/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/add_1AddV2+layer_normalization_741/batchnorm/mul_1:z:0)layer_normalization_741/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
�
IdentityIdentity+layer_normalization_741/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������
��
NoOpNoOp1^layer_normalization_740/batchnorm/ReadVariableOp5^layer_normalization_740/batchnorm/mul/ReadVariableOp1^layer_normalization_741/batchnorm/ReadVariableOp5^layer_normalization_741/batchnorm/mul/ReadVariableOp@^multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp1^sequential_370/dense_2780/BiasAdd/ReadVariableOp3^sequential_370/dense_2780/Tensordot/ReadVariableOp1^sequential_370/dense_2781/BiasAdd/ReadVariableOp3^sequential_370/dense_2781/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������
�: : : : : : : : : : : : : : : : 2d
0layer_normalization_740/batchnorm/ReadVariableOp0layer_normalization_740/batchnorm/ReadVariableOp2l
4layer_normalization_740/batchnorm/mul/ReadVariableOp4layer_normalization_740/batchnorm/mul/ReadVariableOp2d
0layer_normalization_741/batchnorm/ReadVariableOp0layer_normalization_741/batchnorm/ReadVariableOp2l
4layer_normalization_741/batchnorm/mul/ReadVariableOp4layer_normalization_741/batchnorm/mul/ReadVariableOp2�
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp2d
0sequential_370/dense_2780/BiasAdd/ReadVariableOp0sequential_370/dense_2780/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2780/Tensordot/ReadVariableOp2sequential_370/dense_2780/Tensordot/ReadVariableOp2d
0sequential_370/dense_2781/BiasAdd/ReadVariableOp0sequential_370/dense_2781/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2781/Tensordot/ReadVariableOp2sequential_370/dense_2781/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
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
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
G__inference_dense_2781_layer_call_and_return_conditional_losses_5176215

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������
�V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
3__inference_encoder_layer_370_layer_call_fn_5177301

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5176585t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������
�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177297:'#
!
_user_specified_name	5177295:'#
!
_user_specified_name	5177293:'#
!
_user_specified_name	5177291:'#
!
_user_specified_name	5177289:'#
!
_user_specified_name	5177287:'
#
!
_user_specified_name	5177285:'	#
!
_user_specified_name	5177283:'#
!
_user_specified_name	5177281:'#
!
_user_specified_name	5177279:'#
!
_user_specified_name	5177277:'#
!
_user_specified_name	5177275:'#
!
_user_specified_name	5177273:'#
!
_user_specified_name	5177271:'#
!
_user_specified_name	5177269:'#
!
_user_specified_name	5177267:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176222
dense_2780_input%
dense_2780_5176181:	� 
dense_2780_5176183:%
dense_2781_5176216:	�!
dense_2781_5176218:	�
identity��"dense_2780/StatefulPartitionedCall�"dense_2781/StatefulPartitionedCall�
"dense_2780/StatefulPartitionedCallStatefulPartitionedCalldense_2780_inputdense_2780_5176181dense_2780_5176183*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2780_layer_call_and_return_conditional_losses_5176180�
"dense_2781/StatefulPartitionedCallStatefulPartitionedCall+dense_2780/StatefulPartitionedCall:output:0dense_2781_5176216dense_2781_5176218*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2781_layer_call_and_return_conditional_losses_5176215
IdentityIdentity+dense_2781/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�l
NoOpNoOp#^dense_2780/StatefulPartitionedCall#^dense_2781/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������
�: : : : 2H
"dense_2780/StatefulPartitionedCall"dense_2780/StatefulPartitionedCall2H
"dense_2781/StatefulPartitionedCall"dense_2781/StatefulPartitionedCall:'#
!
_user_specified_name	5176218:'#
!
_user_specified_name	5176216:'#
!
_user_specified_name	5176183:'#
!
_user_specified_name	5176181:^ Z
,
_output_shapes
:���������
�
*
_user_specified_namedense_2780_input
�
�
,__inference_dense_2775_layer_call_fn_5177195

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2775_layer_call_and_return_conditional_losses_5176312t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177191:'#
!
_user_specified_name	5177189:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
,__inference_dense_2781_layer_call_fn_5177909

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_2781_layer_call_and_return_conditional_losses_5176215t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������
�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5177905:'#
!
_user_specified_name	5177903:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5176916

inputs^
Jmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource:	�^
Jmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource:
��W
Hmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource:	�L
=layer_normalization_740_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_740_batchnorm_readvariableop_resource:	�N
;sequential_370_dense_2780_tensordot_readvariableop_resource:	�G
9sequential_370_dense_2780_biasadd_readvariableop_resource:N
;sequential_370_dense_2781_tensordot_readvariableop_resource:	�H
9sequential_370_dense_2781_biasadd_readvariableop_resource:	�L
=layer_normalization_741_batchnorm_mul_readvariableop_resource:	�H
9layer_normalization_741_batchnorm_readvariableop_resource:	�
identity��0layer_normalization_740/batchnorm/ReadVariableOp�4layer_normalization_740/batchnorm/mul/ReadVariableOp�0layer_normalization_741/batchnorm/ReadVariableOp�4layer_normalization_741/batchnorm/mul/ReadVariableOp�?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp�?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp�Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp�0sequential_370/dense_2780/BiasAdd/ReadVariableOp�2sequential_370/dense_2780/Tensordot/ReadVariableOp�0sequential_370/dense_2781/BiasAdd/ReadVariableOp�2sequential_370/dense_2781/Tensordot/ReadVariableOp�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2776_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2776/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2776/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2776/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2776/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2776/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2776/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2776/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2776/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2776/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2776/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2776/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2776/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2776/Tensordot/free:output:0@multi_head_self_attention_370/dense_2776/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2776/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2776/Tensordot/stackPack@multi_head_self_attention_370/dense_2776/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2776/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2776/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2776/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2776/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2776/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2776/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2776/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2776/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2776/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2776/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2776/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2776/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2776/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2776/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2776/TensordotReshapeCmulti_head_self_attention_370/dense_2776/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2776/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2776_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2776/BiasAddBiasAdd;multi_head_self_attention_370/dense_2776/Tensordot:output:0Gmulti_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2777_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2777/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2777/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2777/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2777/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2777/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2777/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2777/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2777/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2777/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2777/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2777/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2777/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2777/Tensordot/free:output:0@multi_head_self_attention_370/dense_2777/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2777/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2777/Tensordot/stackPack@multi_head_self_attention_370/dense_2777/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2777/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2777/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2777/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2777/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2777/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2777/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2777/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2777/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2777/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2777/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2777/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2777/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2777/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2777/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2777/TensordotReshapeCmulti_head_self_attention_370/dense_2777/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2777/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2777_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2777/BiasAddBiasAdd;multi_head_self_attention_370/dense_2777/Tensordot:output:0Gmulti_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2778_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2778/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2778/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
8multi_head_self_attention_370/dense_2778/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2778/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2778/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2778/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2778/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2778/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2778/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2778/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2778/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2778/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2778/Tensordot/free:output:0@multi_head_self_attention_370/dense_2778/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2778/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2778/Tensordot/stackPack@multi_head_self_attention_370/dense_2778/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2778/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2778/Tensordot/transpose	TransposeinputsBmulti_head_self_attention_370/dense_2778/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
:multi_head_self_attention_370/dense_2778/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2778/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2778/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2778/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2778/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2778/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2778/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2778/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2778/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2778/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2778/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2778/TensordotReshapeCmulti_head_self_attention_370/dense_2778/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2778/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2778_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2778/BiasAddBiasAdd;multi_head_self_attention_370/dense_2778/Tensordot:output:0Gmulti_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
#multi_head_self_attention_370/ShapeShape9multi_head_self_attention_370/dense_2776/BiasAdd:output:0*
T0*
_output_shapes
::��{
1multi_head_self_attention_370/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3multi_head_self_attention_370/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3multi_head_self_attention_370/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+multi_head_self_attention_370/strided_sliceStridedSlice,multi_head_self_attention_370/Shape:output:0:multi_head_self_attention_370/strided_slice/stack:output:0<multi_head_self_attention_370/strided_slice/stack_1:output:0<multi_head_self_attention_370/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
-multi_head_self_attention_370/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������o
-multi_head_self_attention_370/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :p
-multi_head_self_attention_370/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
+multi_head_self_attention_370/Reshape/shapePack4multi_head_self_attention_370/strided_slice:output:06multi_head_self_attention_370/Reshape/shape/1:output:06multi_head_self_attention_370/Reshape/shape/2:output:06multi_head_self_attention_370/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
%multi_head_self_attention_370/ReshapeReshape9multi_head_self_attention_370/dense_2776/BiasAdd:output:04multi_head_self_attention_370/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
,multi_head_self_attention_370/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
'multi_head_self_attention_370/transpose	Transpose.multi_head_self_attention_370/Reshape:output:05multi_head_self_attention_370/transpose/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_1Shape9multi_head_self_attention_370/dense_2777/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_1StridedSlice.multi_head_self_attention_370/Shape_1:output:0<multi_head_self_attention_370/strided_slice_1/stack:output:0>multi_head_self_attention_370/strided_slice_1/stack_1:output:0>multi_head_self_attention_370/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_1/shapePack6multi_head_self_attention_370/strided_slice_1:output:08multi_head_self_attention_370/Reshape_1/shape/1:output:08multi_head_self_attention_370/Reshape_1/shape/2:output:08multi_head_self_attention_370/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_1Reshape9multi_head_self_attention_370/dense_2777/BiasAdd:output:06multi_head_self_attention_370/Reshape_1/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_1	Transpose0multi_head_self_attention_370/Reshape_1:output:07multi_head_self_attention_370/transpose_1/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
%multi_head_self_attention_370/Shape_2Shape9multi_head_self_attention_370/dense_2778/BiasAdd:output:0*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_2StridedSlice.multi_head_self_attention_370/Shape_2:output:0<multi_head_self_attention_370/strided_slice_2/stack:output:0>multi_head_self_attention_370/strided_slice_2/stack_1:output:0>multi_head_self_attention_370/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������q
/multi_head_self_attention_370/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
/multi_head_self_attention_370/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_2/shapePack6multi_head_self_attention_370/strided_slice_2:output:08multi_head_self_attention_370/Reshape_2/shape/1:output:08multi_head_self_attention_370/Reshape_2/shape/2:output:08multi_head_self_attention_370/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_2Reshape9multi_head_self_attention_370/dense_2778/BiasAdd:output:06multi_head_self_attention_370/Reshape_2/shape:output:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_2	Transpose0multi_head_self_attention_370/Reshape_2:output:07multi_head_self_attention_370/transpose_2/perm:output:0*
T0*9
_output_shapes'
%:#��������������������
$multi_head_self_attention_370/MatMulBatchMatMulV2+multi_head_self_attention_370/transpose:y:0-multi_head_self_attention_370/transpose_1:y:0*
T0*A
_output_shapes/
-:+���������������������������*
adj_y(i
$multi_head_self_attention_370/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *   Cz
"multi_head_self_attention_370/SqrtSqrt-multi_head_self_attention_370/Sqrt/x:output:0*
T0*
_output_shapes
: �
%multi_head_self_attention_370/truedivRealDiv-multi_head_self_attention_370/MatMul:output:0&multi_head_self_attention_370/Sqrt:y:0*
T0*A
_output_shapes/
-:+����������������������������
%multi_head_self_attention_370/SoftmaxSoftmax)multi_head_self_attention_370/truediv:z:0*
T0*A
_output_shapes/
-:+����������������������������
&multi_head_self_attention_370/MatMul_1BatchMatMulV2/multi_head_self_attention_370/Softmax:softmax:0-multi_head_self_attention_370/transpose_2:y:0*
T0*9
_output_shapes'
%:#��������������������
.multi_head_self_attention_370/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
)multi_head_self_attention_370/transpose_3	Transpose/multi_head_self_attention_370/MatMul_1:output:07multi_head_self_attention_370/transpose_3/perm:output:0*
T0*9
_output_shapes'
%:#�������������������i
%multi_head_self_attention_370/Shape_3Shapeinputs*
T0*
_output_shapes
::��}
3multi_head_self_attention_370/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5multi_head_self_attention_370/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5multi_head_self_attention_370/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-multi_head_self_attention_370/strided_slice_3StridedSlice.multi_head_self_attention_370/Shape_3:output:0<multi_head_self_attention_370/strided_slice_3/stack:output:0>multi_head_self_attention_370/strided_slice_3/stack_1:output:0>multi_head_self_attention_370/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/multi_head_self_attention_370/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������r
/multi_head_self_attention_370/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
-multi_head_self_attention_370/Reshape_3/shapePack6multi_head_self_attention_370/strided_slice_3:output:08multi_head_self_attention_370/Reshape_3/shape/1:output:08multi_head_self_attention_370/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:�
'multi_head_self_attention_370/Reshape_3Reshape-multi_head_self_attention_370/transpose_3:y:06multi_head_self_attention_370/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:��������������������
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpReadVariableOpJmulti_head_self_attention_370_dense_2779_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7multi_head_self_attention_370/dense_2779/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
7multi_head_self_attention_370/dense_2779/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
8multi_head_self_attention_370/dense_2779/Tensordot/ShapeShape0multi_head_self_attention_370/Reshape_3:output:0*
T0*
_output_shapes
::���
@multi_head_self_attention_370/dense_2779/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/GatherV2GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=multi_head_self_attention_370/dense_2779/Tensordot/GatherV2_1GatherV2Amulti_head_self_attention_370/dense_2779/Tensordot/Shape:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Kmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
7multi_head_self_attention_370/dense_2779/Tensordot/ProdProdDmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Amulti_head_self_attention_370/dense_2779/Tensordot/Const:output:0*
T0*
_output_shapes
: �
:multi_head_self_attention_370/dense_2779/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
9multi_head_self_attention_370/dense_2779/Tensordot/Prod_1ProdFmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2_1:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
>multi_head_self_attention_370/dense_2779/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
9multi_head_self_attention_370/dense_2779/Tensordot/concatConcatV2@multi_head_self_attention_370/dense_2779/Tensordot/free:output:0@multi_head_self_attention_370/dense_2779/Tensordot/axes:output:0Gmulti_head_self_attention_370/dense_2779/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
8multi_head_self_attention_370/dense_2779/Tensordot/stackPack@multi_head_self_attention_370/dense_2779/Tensordot/Prod:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
<multi_head_self_attention_370/dense_2779/Tensordot/transpose	Transpose0multi_head_self_attention_370/Reshape_3:output:0Bmulti_head_self_attention_370/dense_2779/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
:multi_head_self_attention_370/dense_2779/Tensordot/ReshapeReshape@multi_head_self_attention_370/dense_2779/Tensordot/transpose:y:0Amulti_head_self_attention_370/dense_2779/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
9multi_head_self_attention_370/dense_2779/Tensordot/MatMulMatMulCmulti_head_self_attention_370/dense_2779/Tensordot/Reshape:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:multi_head_self_attention_370/dense_2779/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
@multi_head_self_attention_370/dense_2779/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
;multi_head_self_attention_370/dense_2779/Tensordot/concat_1ConcatV2Dmulti_head_self_attention_370/dense_2779/Tensordot/GatherV2:output:0Cmulti_head_self_attention_370/dense_2779/Tensordot/Const_2:output:0Imulti_head_self_attention_370/dense_2779/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
2multi_head_self_attention_370/dense_2779/TensordotReshapeCmulti_head_self_attention_370/dense_2779/Tensordot/MatMul:product:0Dmulti_head_self_attention_370/dense_2779/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpReadVariableOpHmulti_head_self_attention_370_dense_2779_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
0multi_head_self_attention_370/dense_2779/BiasAddBiasAdd;multi_head_self_attention_370/dense_2779/Tensordot:output:0Gmulti_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
dropout_874/IdentityIdentity9multi_head_self_attention_370/dense_2779/BiasAdd:output:0*
T0*5
_output_shapes#
!:�������������������j
addAddV2inputsdropout_874/Identity:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_740/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_740/moments/meanMeanadd:z:0?layer_normalization_740/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_740/moments/StopGradientStopGradient-layer_normalization_740/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_740/moments/SquaredDifferenceSquaredDifferenceadd:z:05layer_normalization_740/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_740/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_740/moments/varianceMean5layer_normalization_740/moments/SquaredDifference:z:0Clayer_normalization_740/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_740/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_740/batchnorm/addAddV21layer_normalization_740/moments/variance:output:00layer_normalization_740/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_740/batchnorm/RsqrtRsqrt)layer_normalization_740/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_740/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_740_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/mulMul+layer_normalization_740/batchnorm/Rsqrt:y:0<layer_normalization_740/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_1Muladd:z:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/mul_2Mul-layer_normalization_740/moments/mean:output:0)layer_normalization_740/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_740/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_740_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_740/batchnorm/subSub8layer_normalization_740/batchnorm/ReadVariableOp:value:0+layer_normalization_740/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_740/batchnorm/add_1AddV2+layer_normalization_740/batchnorm/mul_1:z:0)layer_normalization_740/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
��
2sequential_370/dense_2780/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2780_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2780/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2780/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2780/Tensordot/ShapeShape+layer_normalization_740/batchnorm/add_1:z:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2780/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/GatherV2GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/free:output:0:sequential_370/dense_2780/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2780/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2780/Tensordot/GatherV2_1GatherV22sequential_370/dense_2780/Tensordot/Shape:output:01sequential_370/dense_2780/Tensordot/axes:output:0<sequential_370/dense_2780/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2780/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2780/Tensordot/ProdProd5sequential_370/dense_2780/Tensordot/GatherV2:output:02sequential_370/dense_2780/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2780/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2780/Tensordot/Prod_1Prod7sequential_370/dense_2780/Tensordot/GatherV2_1:output:04sequential_370/dense_2780/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2780/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2780/Tensordot/concatConcatV21sequential_370/dense_2780/Tensordot/free:output:01sequential_370/dense_2780/Tensordot/axes:output:08sequential_370/dense_2780/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2780/Tensordot/stackPack1sequential_370/dense_2780/Tensordot/Prod:output:03sequential_370/dense_2780/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2780/Tensordot/transpose	Transpose+layer_normalization_740/batchnorm/add_1:z:03sequential_370/dense_2780/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������
��
+sequential_370/dense_2780/Tensordot/ReshapeReshape1sequential_370/dense_2780/Tensordot/transpose:y:02sequential_370/dense_2780/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2780/Tensordot/MatMulMatMul4sequential_370/dense_2780/Tensordot/Reshape:output:0:sequential_370/dense_2780/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
+sequential_370/dense_2780/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:s
1sequential_370/dense_2780/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2780/Tensordot/concat_1ConcatV25sequential_370/dense_2780/Tensordot/GatherV2:output:04sequential_370/dense_2780/Tensordot/Const_2:output:0:sequential_370/dense_2780/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2780/TensordotReshape4sequential_370/dense_2780/Tensordot/MatMul:product:05sequential_370/dense_2780/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������
�
0sequential_370/dense_2780/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2780_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_370/dense_2780/BiasAddBiasAdd,sequential_370/dense_2780/Tensordot:output:08sequential_370/dense_2780/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������
�
sequential_370/dense_2780/ReluRelu*sequential_370/dense_2780/BiasAdd:output:0*
T0*+
_output_shapes
:���������
�
2sequential_370/dense_2781/Tensordot/ReadVariableOpReadVariableOp;sequential_370_dense_2781_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_370/dense_2781/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_370/dense_2781/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_370/dense_2781/Tensordot/ShapeShape,sequential_370/dense_2780/Relu:activations:0*
T0*
_output_shapes
::��s
1sequential_370/dense_2781/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/GatherV2GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/free:output:0:sequential_370/dense_2781/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_370/dense_2781/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_370/dense_2781/Tensordot/GatherV2_1GatherV22sequential_370/dense_2781/Tensordot/Shape:output:01sequential_370/dense_2781/Tensordot/axes:output:0<sequential_370/dense_2781/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_370/dense_2781/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_370/dense_2781/Tensordot/ProdProd5sequential_370/dense_2781/Tensordot/GatherV2:output:02sequential_370/dense_2781/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_370/dense_2781/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_370/dense_2781/Tensordot/Prod_1Prod7sequential_370/dense_2781/Tensordot/GatherV2_1:output:04sequential_370/dense_2781/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_370/dense_2781/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_370/dense_2781/Tensordot/concatConcatV21sequential_370/dense_2781/Tensordot/free:output:01sequential_370/dense_2781/Tensordot/axes:output:08sequential_370/dense_2781/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_370/dense_2781/Tensordot/stackPack1sequential_370/dense_2781/Tensordot/Prod:output:03sequential_370/dense_2781/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_370/dense_2781/Tensordot/transpose	Transpose,sequential_370/dense_2780/Relu:activations:03sequential_370/dense_2781/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
+sequential_370/dense_2781/Tensordot/ReshapeReshape1sequential_370/dense_2781/Tensordot/transpose:y:02sequential_370/dense_2781/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_370/dense_2781/Tensordot/MatMulMatMul4sequential_370/dense_2781/Tensordot/Reshape:output:0:sequential_370/dense_2781/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
+sequential_370/dense_2781/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�s
1sequential_370/dense_2781/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_370/dense_2781/Tensordot/concat_1ConcatV25sequential_370/dense_2781/Tensordot/GatherV2:output:04sequential_370/dense_2781/Tensordot/Const_2:output:0:sequential_370/dense_2781/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_370/dense_2781/TensordotReshape4sequential_370/dense_2781/Tensordot/MatMul:product:05sequential_370/dense_2781/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
��
0sequential_370/dense_2781/BiasAdd/ReadVariableOpReadVariableOp9sequential_370_dense_2781_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!sequential_370/dense_2781/BiasAddBiasAdd,sequential_370/dense_2781/Tensordot:output:08sequential_370/dense_2781/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
dropout_875/IdentityIdentity*sequential_370/dense_2781/BiasAdd:output:0*
T0*,
_output_shapes
:���������
��
add_1AddV2+layer_normalization_740/batchnorm/add_1:z:0dropout_875/Identity:output:0*
T0*,
_output_shapes
:���������
��
6layer_normalization_741/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
$layer_normalization_741/moments/meanMean	add_1:z:0?layer_normalization_741/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(�
,layer_normalization_741/moments/StopGradientStopGradient-layer_normalization_741/moments/mean:output:0*
T0*+
_output_shapes
:���������
�
1layer_normalization_741/moments/SquaredDifferenceSquaredDifference	add_1:z:05layer_normalization_741/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������
��
:layer_normalization_741/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
(layer_normalization_741/moments/varianceMean5layer_normalization_741/moments/SquaredDifference:z:0Clayer_normalization_741/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������
*
	keep_dims(l
'layer_normalization_741/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
%layer_normalization_741/batchnorm/addAddV21layer_normalization_741/moments/variance:output:00layer_normalization_741/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������
�
'layer_normalization_741/batchnorm/RsqrtRsqrt)layer_normalization_741/batchnorm/add:z:0*
T0*+
_output_shapes
:���������
�
4layer_normalization_741/batchnorm/mul/ReadVariableOpReadVariableOp=layer_normalization_741_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/mulMul+layer_normalization_741/batchnorm/Rsqrt:y:0<layer_normalization_741/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_1Mul	add_1:z:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/mul_2Mul-layer_normalization_741/moments/mean:output:0)layer_normalization_741/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������
��
0layer_normalization_741/batchnorm/ReadVariableOpReadVariableOp9layer_normalization_741_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%layer_normalization_741/batchnorm/subSub8layer_normalization_741/batchnorm/ReadVariableOp:value:0+layer_normalization_741/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:���������
��
'layer_normalization_741/batchnorm/add_1AddV2+layer_normalization_741/batchnorm/mul_1:z:0)layer_normalization_741/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������
�
IdentityIdentity+layer_normalization_741/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������
��
NoOpNoOp1^layer_normalization_740/batchnorm/ReadVariableOp5^layer_normalization_740/batchnorm/mul/ReadVariableOp1^layer_normalization_741/batchnorm/ReadVariableOp5^layer_normalization_741/batchnorm/mul/ReadVariableOp@^multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp@^multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOpB^multi_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp1^sequential_370/dense_2780/BiasAdd/ReadVariableOp3^sequential_370/dense_2780/Tensordot/ReadVariableOp1^sequential_370/dense_2781/BiasAdd/ReadVariableOp3^sequential_370/dense_2781/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������
�: : : : : : : : : : : : : : : : 2d
0layer_normalization_740/batchnorm/ReadVariableOp0layer_normalization_740/batchnorm/ReadVariableOp2l
4layer_normalization_740/batchnorm/mul/ReadVariableOp4layer_normalization_740/batchnorm/mul/ReadVariableOp2d
0layer_normalization_741/batchnorm/ReadVariableOp0layer_normalization_741/batchnorm/ReadVariableOp2l
4layer_normalization_741/batchnorm/mul/ReadVariableOp4layer_normalization_741/batchnorm/mul/ReadVariableOp2�
?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2776/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2776/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2777/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2777/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2778/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2778/Tensordot/ReadVariableOp2�
?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp?multi_head_self_attention_370/dense_2779/BiasAdd/ReadVariableOp2�
Amulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOpAmulti_head_self_attention_370/dense_2779/Tensordot/ReadVariableOp2d
0sequential_370/dense_2780/BiasAdd/ReadVariableOp0sequential_370/dense_2780/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2780/Tensordot/ReadVariableOp2sequential_370/dense_2780/Tensordot/ReadVariableOp2d
0sequential_370/dense_2781/BiasAdd/ReadVariableOp0sequential_370/dense_2781/BiasAdd/ReadVariableOp2h
2sequential_370/dense_2781/Tensordot/ReadVariableOp2sequential_370/dense_2781/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
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
resource:T P
,
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
G__inference_dense_2775_layer_call_and_return_conditional_losses_5177225

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������
�V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
G__inference_dense_2781_layer_call_and_return_conditional_losses_5177939

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������
�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������
�s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������
�d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������
�V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177001
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *d
f_R]
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176655s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	5176997:'#
!
_user_specified_name	5176995:'#
!
_user_specified_name	5176993:'#
!
_user_specified_name	5176991:'#
!
_user_specified_name	5176989:'#
!
_user_specified_name	5176987:'#
!
_user_specified_name	5176985:'#
!
_user_specified_name	5176983:'#
!
_user_specified_name	5176981:'#
!
_user_specified_name	5176979:'
#
!
_user_specified_name	5176977:'	#
!
_user_specified_name	5176975:'#
!
_user_specified_name	5176973:'#
!
_user_specified_name	5176971:'#
!
_user_specified_name	5176969:'#
!
_user_specified_name	5176967:'#
!
_user_specified_name	5176965:'#
!
_user_specified_name	5176963:'#
!
_user_specified_name	5176961:'#
!
_user_specified_name	5176959:T P
+
_output_shapes
:���������

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
?
input_14
serving_default_input_1:0���������
@
output_14
StatefulPartitionedCall:0���������
tensorflow/serving/predict:�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
input_layer
	
enc_layers


ffn_output
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
&trace_0
'trace_12�
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177001
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177046�
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
 z&trace_0z'trace_1
�
(trace_0
)trace_12�
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176655
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176956�
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
 z(trace_0z)trace_1
�B�
"__inference__wrapped_model_5176147input_1"�
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
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
'
00"
trackable_list_wrapper
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
7
_variables
8_iterations
9_learning_rate
:_index_dict
;
_momentums
<_velocities
=_update_step_xla"
experimentalOptimizer
,
>serving_default"
signature_map
C:A	�20csmom_transformer_reranker_117/dense_2775/kernel
=:;�2.csmom_transformer_reranker_117/dense_2775/bias
t:r
��2`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel
m:k�2^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias
t:r
��2`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel
m:k�2^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias
t:r
��2`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel
m:k�2^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias
t:r
��2`csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel
m:k�2^csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias
$:"	�2dense_2780/kernel
:2dense_2780/bias
$:"	�2dense_2781/kernel
:�2dense_2781/bias
]:[�2Ncsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma
\:Z�2Mcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta
]:[�2Ncsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma
\:Z�2Mcsmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta
C:A	�20csmom_transformer_reranker_117/dense_2782/kernel
<::2.csmom_transformer_reranker_117/dense_2782/bias
 "
trackable_list_wrapper
5
0
01

2"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177001input_1"�
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
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177046input_1"�
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
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176655input_1"�
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
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176956input_1"�
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
Etrace_02�
,__inference_dense_2775_layer_call_fn_5177195�
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
 zEtrace_0
�
Ftrace_02�
G__inference_dense_2775_layer_call_and_return_conditional_losses_5177225�
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
 zFtrace_0
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Matt
Nffn
O
layernorm1
P
layernorm2
Qdropout1
Rdropout2"
_tf_keras_layer
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_02�
,__inference_dense_2782_layer_call_fn_5177234�
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
 zXtrace_0
�
Ytrace_02�
G__inference_dense_2782_layer_call_and_return_conditional_losses_5177264�
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
 zYtrace_0
�
80
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15
i16
j17
k18
l19
m20
n21
o22
p23
q24
r25
s26
t27
u28
v29
w30
x31
y32
z33
{34
|35
}36
~37
38
�39
�40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
Z0
\1
^2
`3
b4
d5
f6
h7
j8
l9
n10
p11
r12
t13
v14
x15
z16
|17
~18
�19"
trackable_list_wrapper
�
[0
]1
_2
a3
c4
e5
g6
i7
k8
m9
o10
q11
s12
u13
w14
y15
{16
}17
18
�19"
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
%__inference_signature_wrapper_5177186input_1"�
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
R
�	variables
�	keras_api

�total

�count"
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
,__inference_dense_2775_layer_call_fn_5177195inputs"�
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
G__inference_dense_2775_layer_call_and_return_conditional_losses_5177225inputs"�
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
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_encoder_layer_370_layer_call_fn_5177301
3__inference_encoder_layer_370_layer_call_fn_5177338�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5177606
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5177860�
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
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�query_dense
�	key_dense
�value_dense
�combine_heads"
_tf_keras_layer
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
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
,__inference_dense_2782_layer_call_fn_5177234inputs"�
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
G__inference_dense_2782_layer_call_and_return_conditional_losses_5177264inputs"�
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
H:F	�27Adam/m/csmom_transformer_reranker_117/dense_2775/kernel
H:F	�27Adam/v/csmom_transformer_reranker_117/dense_2775/kernel
B:@�25Adam/m/csmom_transformer_reranker_117/dense_2775/bias
B:@�25Adam/v/csmom_transformer_reranker_117/dense_2775/bias
y:w
��2gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel
y:w
��2gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/kernel
r:p�2eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias
r:p�2eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2776/bias
y:w
��2gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel
y:w
��2gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/kernel
r:p�2eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias
r:p�2eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2777/bias
y:w
��2gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel
y:w
��2gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/kernel
r:p�2eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias
r:p�2eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2778/bias
y:w
��2gAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel
y:w
��2gAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/kernel
r:p�2eAdam/m/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias
r:p�2eAdam/v/csmom_transformer_reranker_117/encoder_layer_370/multi_head_self_attention_370/dense_2779/bias
):'	�2Adam/m/dense_2780/kernel
):'	�2Adam/v/dense_2780/kernel
": 2Adam/m/dense_2780/bias
": 2Adam/v/dense_2780/bias
):'	�2Adam/m/dense_2781/kernel
):'	�2Adam/v/dense_2781/kernel
#:!�2Adam/m/dense_2781/bias
#:!�2Adam/v/dense_2781/bias
b:`�2UAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma
b:`�2UAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/gamma
a:_�2TAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta
a:_�2TAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_740/beta
b:`�2UAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma
b:`�2UAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/gamma
a:_�2TAdam/m/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta
a:_�2TAdam/v/csmom_transformer_reranker_117/encoder_layer_370/layer_normalization_741/beta
H:F	�27Adam/m/csmom_transformer_reranker_117/dense_2782/kernel
H:F	�27Adam/v/csmom_transformer_reranker_117/dense_2782/kernel
A:?25Adam/m/csmom_transformer_reranker_117/dense_2782/bias
A:?25Adam/v/csmom_transformer_reranker_117/dense_2782/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
J
M0
N1
O2
P3
Q4
R5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_encoder_layer_370_layer_call_fn_5177301inputs"�
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
3__inference_encoder_layer_370_layer_call_fn_5177338inputs"�
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
�B�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5177606inputs"�
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
�B�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_5177860inputs"�
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
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_sequential_370_layer_call_fn_5176249
0__inference_sequential_370_layer_call_fn_5176262�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176222
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176236�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
 
�2��
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
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
 
�2��
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
 
"
_generic_user_object
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_2780_layer_call_fn_5177869�
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
 z�trace_0
�
�trace_02�
G__inference_dense_2780_layer_call_and_return_conditional_losses_5177900�
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
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_2781_layer_call_fn_5177909�
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
 z�trace_0
�
�trace_02�
G__inference_dense_2781_layer_call_and_return_conditional_losses_5177939�
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
 z�trace_0
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_sequential_370_layer_call_fn_5176249dense_2780_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
0__inference_sequential_370_layer_call_fn_5176262dense_2780_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
�B�
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176222dense_2780_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
�B�
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176236dense_2780_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
,__inference_dense_2780_layer_call_fn_5177869inputs"�
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
G__inference_dense_2780_layer_call_and_return_conditional_losses_5177900inputs"�
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
,__inference_dense_2781_layer_call_fn_5177909inputs"�
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
G__inference_dense_2781_layer_call_and_return_conditional_losses_5177939inputs"�
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
 �
"__inference__wrapped_model_5176147� 4�1
*�'
%�"
input_1���������

� "7�4
2
output_1&�#
output_1���������
�
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176655� D�A
*�'
%�"
input_1���������

�

trainingp"0�-
&�#
tensor_0���������

� �
[__inference_csmom_transformer_reranker_117_layer_call_and_return_conditional_losses_5176956� D�A
*�'
%�"
input_1���������

�

trainingp "0�-
&�#
tensor_0���������

� �
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177001� D�A
*�'
%�"
input_1���������

�

trainingp"%�"
unknown���������
�
@__inference_csmom_transformer_reranker_117_layer_call_fn_5177046� D�A
*�'
%�"
input_1���������

�

trainingp "%�"
unknown���������
�
G__inference_dense_2775_layer_call_and_return_conditional_losses_5177225l3�0
)�&
$�!
inputs���������

� "1�.
'�$
tensor_0���������
�
� �
,__inference_dense_2775_layer_call_fn_5177195a3�0
)�&
$�!
inputs���������

� "&�#
unknown���������
��
G__inference_dense_2780_layer_call_and_return_conditional_losses_5177900l4�1
*�'
%�"
inputs���������
�
� "0�-
&�#
tensor_0���������

� �
,__inference_dense_2780_layer_call_fn_5177869a4�1
*�'
%�"
inputs���������
�
� "%�"
unknown���������
�
G__inference_dense_2781_layer_call_and_return_conditional_losses_5177939l3�0
)�&
$�!
inputs���������

� "1�.
'�$
tensor_0���������
�
� �
,__inference_dense_2781_layer_call_fn_5177909a3�0
)�&
$�!
inputs���������

� "&�#
unknown���������
��
G__inference_dense_2782_layer_call_and_return_conditional_losses_5177264l 4�1
*�'
%�"
inputs���������
�
� "0�-
&�#
tensor_0���������

� �
,__inference_dense_2782_layer_call_fn_5177234a 4�1
*�'
%�"
inputs���������
�
� "%�"
unknown���������
�
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_51776068�5
.�+
%�"
inputs���������
�
p
� "1�.
'�$
tensor_0���������
�
� �
N__inference_encoder_layer_370_layer_call_and_return_conditional_losses_51778608�5
.�+
%�"
inputs���������
�
p 
� "1�.
'�$
tensor_0���������
�
� �
3__inference_encoder_layer_370_layer_call_fn_5177301t8�5
.�+
%�"
inputs���������
�
p
� "&�#
unknown���������
��
3__inference_encoder_layer_370_layer_call_fn_5177338t8�5
.�+
%�"
inputs���������
�
p 
� "&�#
unknown���������
��
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176222�F�C
<�9
/�,
dense_2780_input���������
�
p

 
� "1�.
'�$
tensor_0���������
�
� �
K__inference_sequential_370_layer_call_and_return_conditional_losses_5176236�F�C
<�9
/�,
dense_2780_input���������
�
p 

 
� "1�.
'�$
tensor_0���������
�
� �
0__inference_sequential_370_layer_call_fn_5176249vF�C
<�9
/�,
dense_2780_input���������
�
p

 
� "&�#
unknown���������
��
0__inference_sequential_370_layer_call_fn_5176262vF�C
<�9
/�,
dense_2780_input���������
�
p 

 
� "&�#
unknown���������
��
%__inference_signature_wrapper_5177186� ?�<
� 
5�2
0
input_1%�"
input_1���������
"7�4
2
output_1&�#
output_1���������
