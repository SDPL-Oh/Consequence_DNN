??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
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
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:

*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:

*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:

*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?.
value?.B?. B?.
|
	logic
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer_with_weights-2

layer-2
layer_with_weights-3
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
?
iter

beta_1

beta_2
	decaym[m\m]m^m_m`mambvcvdvevfvgvhvivj
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
?
	variables

layers
regularization_losses
layer_metrics
non_trainable_variables
layer_regularization_losses
trainable_variables
 metrics
 
h

kernel
bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h

kernel
bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
h

kernel
bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

kernel
bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
?
	variables

1layers
regularization_losses
2layer_metrics
3non_trainable_variables
4layer_regularization_losses
trainable_variables
5metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
 

60
71
82

0
1
 

0
1
?
!	variables

9layers
"regularization_losses
:layer_metrics
;non_trainable_variables
<layer_regularization_losses
#trainable_variables
=metrics

0
1
 

0
1
?
%	variables

>layers
&regularization_losses
?layer_metrics
@non_trainable_variables
Alayer_regularization_losses
'trainable_variables
Bmetrics

0
1
 

0
1
?
)	variables

Clayers
*regularization_losses
Dlayer_metrics
Enon_trainable_variables
Flayer_regularization_losses
+trainable_variables
Gmetrics

0
1
 

0
1
?
-	variables

Hlayers
.regularization_losses
Ilayer_metrics
Jnon_trainable_variables
Klayer_regularization_losses
/trainable_variables
Lmetrics

0
	1

2
3
 
 
 
 
4
	Mtotal
	Ncount
O	variables
P	keras_api
D
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api
D
	Vtotal
	Wcount
X
_fn_kwargs
Y	variables
Z	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

O	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

T	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

Y	variables
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_227454
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *(
f#R!
__inference__traced_save_227970
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaydense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotalcounttotal_1count_1total_2count_2Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *+
f&R$
"__inference__traced_restore_228082??
?

?
)__inference_Densenet_layer_call_fn_227082
dense_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_Densenet_layer_call_and_return_conditional_losses_2270632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?
?
(__inference_dense_1_layer_call_fn_227806

inputs
unknown:


	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2270232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_227454
input_1
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_2269882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
)__inference_Densenet_layer_call_fn_227745

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_Densenet_layer_call_and_return_conditional_losses_2270632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_Densenet_layer_call_and_return_conditional_losses_227169

inputs
dense_227148:

dense_227150:
 
dense_1_227153:


dense_1_227155:
 
dense_2_227158:

dense_2_227160: 
dense_3_227163:
dense_3_227165:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_227148dense_227150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2270062
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_227153dense_1_227155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2270232!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_227158dense_2_227160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2270402!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_227163dense_3_227165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2270562!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_Densenet_layer_call_and_return_conditional_losses_227233
dense_input
dense_227212:

dense_227214:
 
dense_1_227217:


dense_1_227219:
 
dense_2_227222:

dense_2_227224: 
dense_3_227227:
dense_3_227229:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_227212dense_227214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2270062
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_227217dense_1_227219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2270232!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_227222dense_2_227224*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2270402!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_227227dense_3_227229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2270562!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?	
?
.__inference_network_model_layer_call_fn_227620

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_network_model_layer_call_and_return_conditional_losses_2272822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_2_layer_call_fn_227826

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2270402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
D__inference_Densenet_layer_call_and_return_conditional_losses_227063

inputs
dense_227007:

dense_227009:
 
dense_1_227024:


dense_1_227026:
 
dense_2_227041:

dense_2_227043: 
dense_3_227057:
dense_3_227059:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_227007dense_227009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2270062
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_227024dense_1_227026*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2270232!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_227041dense_2_227043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2270402!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_227057dense_3_227059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2270562!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
D__inference_Densenet_layer_call_and_return_conditional_losses_227724

inputs6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:

5
'dense_1_biasadd_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
I__inference_network_model_layer_call_and_return_conditional_losses_227485

inputs?
-densenet_dense_matmul_readvariableop_resource:
<
.densenet_dense_biasadd_readvariableop_resource:
A
/densenet_dense_1_matmul_readvariableop_resource:

>
0densenet_dense_1_biasadd_readvariableop_resource:
A
/densenet_dense_2_matmul_readvariableop_resource:
>
0densenet_dense_2_biasadd_readvariableop_resource:A
/densenet_dense_3_matmul_readvariableop_resource:>
0densenet_dense_3_biasadd_readvariableop_resource:
identity??%Densenet/dense/BiasAdd/ReadVariableOp?$Densenet/dense/MatMul/ReadVariableOp?'Densenet/dense_1/BiasAdd/ReadVariableOp?&Densenet/dense_1/MatMul/ReadVariableOp?'Densenet/dense_2/BiasAdd/ReadVariableOp?&Densenet/dense_2/MatMul/ReadVariableOp?'Densenet/dense_3/BiasAdd/ReadVariableOp?&Densenet/dense_3/MatMul/ReadVariableOp?
$Densenet/dense/MatMul/ReadVariableOpReadVariableOp-densenet_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02&
$Densenet/dense/MatMul/ReadVariableOp?
Densenet/dense/MatMulMatMulinputs,Densenet/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/MatMul?
%Densenet/dense/BiasAdd/ReadVariableOpReadVariableOp.densenet_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%Densenet/dense/BiasAdd/ReadVariableOp?
Densenet/dense/BiasAddBiasAddDensenet/dense/MatMul:product:0-Densenet/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/BiasAdd?
Densenet/dense/ReluReluDensenet/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/Relu?
&Densenet/dense_1/MatMul/ReadVariableOpReadVariableOp/densenet_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02(
&Densenet/dense_1/MatMul/ReadVariableOp?
Densenet/dense_1/MatMulMatMul!Densenet/dense/Relu:activations:0.Densenet/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/MatMul?
'Densenet/dense_1/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Densenet/dense_1/BiasAdd/ReadVariableOp?
Densenet/dense_1/BiasAddBiasAdd!Densenet/dense_1/MatMul:product:0/Densenet/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/BiasAdd?
Densenet/dense_1/ReluRelu!Densenet/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/Relu?
&Densenet/dense_2/MatMul/ReadVariableOpReadVariableOp/densenet_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Densenet/dense_2/MatMul/ReadVariableOp?
Densenet/dense_2/MatMulMatMul#Densenet/dense_1/Relu:activations:0.Densenet/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/MatMul?
'Densenet/dense_2/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_2/BiasAdd/ReadVariableOp?
Densenet/dense_2/BiasAddBiasAdd!Densenet/dense_2/MatMul:product:0/Densenet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/BiasAdd?
Densenet/dense_2/ReluRelu!Densenet/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/Relu?
&Densenet/dense_3/MatMul/ReadVariableOpReadVariableOp/densenet_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Densenet/dense_3/MatMul/ReadVariableOp?
Densenet/dense_3/MatMulMatMul#Densenet/dense_2/Relu:activations:0.Densenet/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/MatMul?
'Densenet/dense_3/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_3/BiasAdd/ReadVariableOp?
Densenet/dense_3/BiasAddBiasAdd!Densenet/dense_3/MatMul:product:0/Densenet/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/BiasAdd|
IdentityIdentity!Densenet/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^Densenet/dense/BiasAdd/ReadVariableOp%^Densenet/dense/MatMul/ReadVariableOp(^Densenet/dense_1/BiasAdd/ReadVariableOp'^Densenet/dense_1/MatMul/ReadVariableOp(^Densenet/dense_2/BiasAdd/ReadVariableOp'^Densenet/dense_2/MatMul/ReadVariableOp(^Densenet/dense_3/BiasAdd/ReadVariableOp'^Densenet/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2N
%Densenet/dense/BiasAdd/ReadVariableOp%Densenet/dense/BiasAdd/ReadVariableOp2L
$Densenet/dense/MatMul/ReadVariableOp$Densenet/dense/MatMul/ReadVariableOp2R
'Densenet/dense_1/BiasAdd/ReadVariableOp'Densenet/dense_1/BiasAdd/ReadVariableOp2P
&Densenet/dense_1/MatMul/ReadVariableOp&Densenet/dense_1/MatMul/ReadVariableOp2R
'Densenet/dense_2/BiasAdd/ReadVariableOp'Densenet/dense_2/BiasAdd/ReadVariableOp2P
&Densenet/dense_2/MatMul/ReadVariableOp&Densenet/dense_2/MatMul/ReadVariableOp2R
'Densenet/dense_3/BiasAdd/ReadVariableOp'Densenet/dense_3/BiasAdd/ReadVariableOp2P
&Densenet/dense_3/MatMul/ReadVariableOp&Densenet/dense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_Densenet_layer_call_and_return_conditional_losses_227257
dense_input
dense_227236:

dense_227238:
 
dense_1_227241:


dense_1_227243:
 
dense_2_227246:

dense_2_227248: 
dense_3_227251:
dense_3_227253:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_227236dense_227238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2270062
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_227241dense_1_227243*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2270232!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_227246dense_2_227248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2270402!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_227251dense_3_227253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2270562!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?&
?
D__inference_Densenet_layer_call_and_return_conditional_losses_227693

inputs6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:

5
'dense_1_biasadd_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
.__inference_network_model_layer_call_fn_227662
input_1
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_network_model_layer_call_and_return_conditional_losses_2273452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_227040

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
)__inference_Densenet_layer_call_fn_227766

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_Densenet_layer_call_and_return_conditional_losses_2271692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_227836

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
.__inference_network_model_layer_call_fn_227599
input_1
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_network_model_layer_call_and_return_conditional_losses_2272822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_227056

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
.__inference_network_model_layer_call_fn_227641

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_network_model_layer_call_and_return_conditional_losses_2273452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
I__inference_network_model_layer_call_and_return_conditional_losses_227547
input_1?
-densenet_dense_matmul_readvariableop_resource:
<
.densenet_dense_biasadd_readvariableop_resource:
A
/densenet_dense_1_matmul_readvariableop_resource:

>
0densenet_dense_1_biasadd_readvariableop_resource:
A
/densenet_dense_2_matmul_readvariableop_resource:
>
0densenet_dense_2_biasadd_readvariableop_resource:A
/densenet_dense_3_matmul_readvariableop_resource:>
0densenet_dense_3_biasadd_readvariableop_resource:
identity??%Densenet/dense/BiasAdd/ReadVariableOp?$Densenet/dense/MatMul/ReadVariableOp?'Densenet/dense_1/BiasAdd/ReadVariableOp?&Densenet/dense_1/MatMul/ReadVariableOp?'Densenet/dense_2/BiasAdd/ReadVariableOp?&Densenet/dense_2/MatMul/ReadVariableOp?'Densenet/dense_3/BiasAdd/ReadVariableOp?&Densenet/dense_3/MatMul/ReadVariableOp?
$Densenet/dense/MatMul/ReadVariableOpReadVariableOp-densenet_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02&
$Densenet/dense/MatMul/ReadVariableOp?
Densenet/dense/MatMulMatMulinput_1,Densenet/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/MatMul?
%Densenet/dense/BiasAdd/ReadVariableOpReadVariableOp.densenet_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%Densenet/dense/BiasAdd/ReadVariableOp?
Densenet/dense/BiasAddBiasAddDensenet/dense/MatMul:product:0-Densenet/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/BiasAdd?
Densenet/dense/ReluReluDensenet/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/Relu?
&Densenet/dense_1/MatMul/ReadVariableOpReadVariableOp/densenet_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02(
&Densenet/dense_1/MatMul/ReadVariableOp?
Densenet/dense_1/MatMulMatMul!Densenet/dense/Relu:activations:0.Densenet/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/MatMul?
'Densenet/dense_1/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Densenet/dense_1/BiasAdd/ReadVariableOp?
Densenet/dense_1/BiasAddBiasAdd!Densenet/dense_1/MatMul:product:0/Densenet/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/BiasAdd?
Densenet/dense_1/ReluRelu!Densenet/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/Relu?
&Densenet/dense_2/MatMul/ReadVariableOpReadVariableOp/densenet_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Densenet/dense_2/MatMul/ReadVariableOp?
Densenet/dense_2/MatMulMatMul#Densenet/dense_1/Relu:activations:0.Densenet/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/MatMul?
'Densenet/dense_2/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_2/BiasAdd/ReadVariableOp?
Densenet/dense_2/BiasAddBiasAdd!Densenet/dense_2/MatMul:product:0/Densenet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/BiasAdd?
Densenet/dense_2/ReluRelu!Densenet/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/Relu?
&Densenet/dense_3/MatMul/ReadVariableOpReadVariableOp/densenet_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Densenet/dense_3/MatMul/ReadVariableOp?
Densenet/dense_3/MatMulMatMul#Densenet/dense_2/Relu:activations:0.Densenet/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/MatMul?
'Densenet/dense_3/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_3/BiasAdd/ReadVariableOp?
Densenet/dense_3/BiasAddBiasAdd!Densenet/dense_3/MatMul:product:0/Densenet/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/BiasAdd|
IdentityIdentity!Densenet/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^Densenet/dense/BiasAdd/ReadVariableOp%^Densenet/dense/MatMul/ReadVariableOp(^Densenet/dense_1/BiasAdd/ReadVariableOp'^Densenet/dense_1/MatMul/ReadVariableOp(^Densenet/dense_2/BiasAdd/ReadVariableOp'^Densenet/dense_2/MatMul/ReadVariableOp(^Densenet/dense_3/BiasAdd/ReadVariableOp'^Densenet/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2N
%Densenet/dense/BiasAdd/ReadVariableOp%Densenet/dense/BiasAdd/ReadVariableOp2L
$Densenet/dense/MatMul/ReadVariableOp$Densenet/dense/MatMul/ReadVariableOp2R
'Densenet/dense_1/BiasAdd/ReadVariableOp'Densenet/dense_1/BiasAdd/ReadVariableOp2P
&Densenet/dense_1/MatMul/ReadVariableOp&Densenet/dense_1/MatMul/ReadVariableOp2R
'Densenet/dense_2/BiasAdd/ReadVariableOp'Densenet/dense_2/BiasAdd/ReadVariableOp2P
&Densenet/dense_2/MatMul/ReadVariableOp&Densenet/dense_2/MatMul/ReadVariableOp2R
'Densenet/dense_3/BiasAdd/ReadVariableOp'Densenet/dense_3/BiasAdd/ReadVariableOp2P
&Densenet/dense_3/MatMul/ReadVariableOp&Densenet/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
(__inference_dense_3_layer_call_fn_227845

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2270562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_227786

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2270062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_227006

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
!__inference__wrapped_model_226988
input_1M
;network_model_densenet_dense_matmul_readvariableop_resource:
J
<network_model_densenet_dense_biasadd_readvariableop_resource:
O
=network_model_densenet_dense_1_matmul_readvariableop_resource:

L
>network_model_densenet_dense_1_biasadd_readvariableop_resource:
O
=network_model_densenet_dense_2_matmul_readvariableop_resource:
L
>network_model_densenet_dense_2_biasadd_readvariableop_resource:O
=network_model_densenet_dense_3_matmul_readvariableop_resource:L
>network_model_densenet_dense_3_biasadd_readvariableop_resource:
identity??3network_model/Densenet/dense/BiasAdd/ReadVariableOp?2network_model/Densenet/dense/MatMul/ReadVariableOp?5network_model/Densenet/dense_1/BiasAdd/ReadVariableOp?4network_model/Densenet/dense_1/MatMul/ReadVariableOp?5network_model/Densenet/dense_2/BiasAdd/ReadVariableOp?4network_model/Densenet/dense_2/MatMul/ReadVariableOp?5network_model/Densenet/dense_3/BiasAdd/ReadVariableOp?4network_model/Densenet/dense_3/MatMul/ReadVariableOp?
2network_model/Densenet/dense/MatMul/ReadVariableOpReadVariableOp;network_model_densenet_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype024
2network_model/Densenet/dense/MatMul/ReadVariableOp?
#network_model/Densenet/dense/MatMulMatMulinput_1:network_model/Densenet/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2%
#network_model/Densenet/dense/MatMul?
3network_model/Densenet/dense/BiasAdd/ReadVariableOpReadVariableOp<network_model_densenet_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3network_model/Densenet/dense/BiasAdd/ReadVariableOp?
$network_model/Densenet/dense/BiasAddBiasAdd-network_model/Densenet/dense/MatMul:product:0;network_model/Densenet/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2&
$network_model/Densenet/dense/BiasAdd?
!network_model/Densenet/dense/ReluRelu-network_model/Densenet/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2#
!network_model/Densenet/dense/Relu?
4network_model/Densenet/dense_1/MatMul/ReadVariableOpReadVariableOp=network_model_densenet_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype026
4network_model/Densenet/dense_1/MatMul/ReadVariableOp?
%network_model/Densenet/dense_1/MatMulMatMul/network_model/Densenet/dense/Relu:activations:0<network_model/Densenet/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2'
%network_model/Densenet/dense_1/MatMul?
5network_model/Densenet/dense_1/BiasAdd/ReadVariableOpReadVariableOp>network_model_densenet_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype027
5network_model/Densenet/dense_1/BiasAdd/ReadVariableOp?
&network_model/Densenet/dense_1/BiasAddBiasAdd/network_model/Densenet/dense_1/MatMul:product:0=network_model/Densenet/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2(
&network_model/Densenet/dense_1/BiasAdd?
#network_model/Densenet/dense_1/ReluRelu/network_model/Densenet/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2%
#network_model/Densenet/dense_1/Relu?
4network_model/Densenet/dense_2/MatMul/ReadVariableOpReadVariableOp=network_model_densenet_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype026
4network_model/Densenet/dense_2/MatMul/ReadVariableOp?
%network_model/Densenet/dense_2/MatMulMatMul1network_model/Densenet/dense_1/Relu:activations:0<network_model/Densenet/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%network_model/Densenet/dense_2/MatMul?
5network_model/Densenet/dense_2/BiasAdd/ReadVariableOpReadVariableOp>network_model_densenet_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5network_model/Densenet/dense_2/BiasAdd/ReadVariableOp?
&network_model/Densenet/dense_2/BiasAddBiasAdd/network_model/Densenet/dense_2/MatMul:product:0=network_model/Densenet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&network_model/Densenet/dense_2/BiasAdd?
#network_model/Densenet/dense_2/ReluRelu/network_model/Densenet/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2%
#network_model/Densenet/dense_2/Relu?
4network_model/Densenet/dense_3/MatMul/ReadVariableOpReadVariableOp=network_model_densenet_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4network_model/Densenet/dense_3/MatMul/ReadVariableOp?
%network_model/Densenet/dense_3/MatMulMatMul1network_model/Densenet/dense_2/Relu:activations:0<network_model/Densenet/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%network_model/Densenet/dense_3/MatMul?
5network_model/Densenet/dense_3/BiasAdd/ReadVariableOpReadVariableOp>network_model_densenet_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5network_model/Densenet/dense_3/BiasAdd/ReadVariableOp?
&network_model/Densenet/dense_3/BiasAddBiasAdd/network_model/Densenet/dense_3/MatMul:product:0=network_model/Densenet/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&network_model/Densenet/dense_3/BiasAdd?
IdentityIdentity/network_model/Densenet/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp4^network_model/Densenet/dense/BiasAdd/ReadVariableOp3^network_model/Densenet/dense/MatMul/ReadVariableOp6^network_model/Densenet/dense_1/BiasAdd/ReadVariableOp5^network_model/Densenet/dense_1/MatMul/ReadVariableOp6^network_model/Densenet/dense_2/BiasAdd/ReadVariableOp5^network_model/Densenet/dense_2/MatMul/ReadVariableOp6^network_model/Densenet/dense_3/BiasAdd/ReadVariableOp5^network_model/Densenet/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2j
3network_model/Densenet/dense/BiasAdd/ReadVariableOp3network_model/Densenet/dense/BiasAdd/ReadVariableOp2h
2network_model/Densenet/dense/MatMul/ReadVariableOp2network_model/Densenet/dense/MatMul/ReadVariableOp2n
5network_model/Densenet/dense_1/BiasAdd/ReadVariableOp5network_model/Densenet/dense_1/BiasAdd/ReadVariableOp2l
4network_model/Densenet/dense_1/MatMul/ReadVariableOp4network_model/Densenet/dense_1/MatMul/ReadVariableOp2n
5network_model/Densenet/dense_2/BiasAdd/ReadVariableOp5network_model/Densenet/dense_2/BiasAdd/ReadVariableOp2l
4network_model/Densenet/dense_2/MatMul/ReadVariableOp4network_model/Densenet/dense_2/MatMul/ReadVariableOp2n
5network_model/Densenet/dense_3/BiasAdd/ReadVariableOp5network_model/Densenet/dense_3/BiasAdd/ReadVariableOp2l
4network_model/Densenet/dense_3/MatMul/ReadVariableOp4network_model/Densenet/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_227023

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?-
?
I__inference_network_model_layer_call_and_return_conditional_losses_227516

inputs?
-densenet_dense_matmul_readvariableop_resource:
<
.densenet_dense_biasadd_readvariableop_resource:
A
/densenet_dense_1_matmul_readvariableop_resource:

>
0densenet_dense_1_biasadd_readvariableop_resource:
A
/densenet_dense_2_matmul_readvariableop_resource:
>
0densenet_dense_2_biasadd_readvariableop_resource:A
/densenet_dense_3_matmul_readvariableop_resource:>
0densenet_dense_3_biasadd_readvariableop_resource:
identity??%Densenet/dense/BiasAdd/ReadVariableOp?$Densenet/dense/MatMul/ReadVariableOp?'Densenet/dense_1/BiasAdd/ReadVariableOp?&Densenet/dense_1/MatMul/ReadVariableOp?'Densenet/dense_2/BiasAdd/ReadVariableOp?&Densenet/dense_2/MatMul/ReadVariableOp?'Densenet/dense_3/BiasAdd/ReadVariableOp?&Densenet/dense_3/MatMul/ReadVariableOp?
$Densenet/dense/MatMul/ReadVariableOpReadVariableOp-densenet_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02&
$Densenet/dense/MatMul/ReadVariableOp?
Densenet/dense/MatMulMatMulinputs,Densenet/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/MatMul?
%Densenet/dense/BiasAdd/ReadVariableOpReadVariableOp.densenet_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%Densenet/dense/BiasAdd/ReadVariableOp?
Densenet/dense/BiasAddBiasAddDensenet/dense/MatMul:product:0-Densenet/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/BiasAdd?
Densenet/dense/ReluReluDensenet/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/Relu?
&Densenet/dense_1/MatMul/ReadVariableOpReadVariableOp/densenet_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02(
&Densenet/dense_1/MatMul/ReadVariableOp?
Densenet/dense_1/MatMulMatMul!Densenet/dense/Relu:activations:0.Densenet/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/MatMul?
'Densenet/dense_1/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Densenet/dense_1/BiasAdd/ReadVariableOp?
Densenet/dense_1/BiasAddBiasAdd!Densenet/dense_1/MatMul:product:0/Densenet/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/BiasAdd?
Densenet/dense_1/ReluRelu!Densenet/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/Relu?
&Densenet/dense_2/MatMul/ReadVariableOpReadVariableOp/densenet_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Densenet/dense_2/MatMul/ReadVariableOp?
Densenet/dense_2/MatMulMatMul#Densenet/dense_1/Relu:activations:0.Densenet/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/MatMul?
'Densenet/dense_2/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_2/BiasAdd/ReadVariableOp?
Densenet/dense_2/BiasAddBiasAdd!Densenet/dense_2/MatMul:product:0/Densenet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/BiasAdd?
Densenet/dense_2/ReluRelu!Densenet/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/Relu?
&Densenet/dense_3/MatMul/ReadVariableOpReadVariableOp/densenet_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Densenet/dense_3/MatMul/ReadVariableOp?
Densenet/dense_3/MatMulMatMul#Densenet/dense_2/Relu:activations:0.Densenet/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/MatMul?
'Densenet/dense_3/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_3/BiasAdd/ReadVariableOp?
Densenet/dense_3/BiasAddBiasAdd!Densenet/dense_3/MatMul:product:0/Densenet/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/BiasAdd|
IdentityIdentity!Densenet/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^Densenet/dense/BiasAdd/ReadVariableOp%^Densenet/dense/MatMul/ReadVariableOp(^Densenet/dense_1/BiasAdd/ReadVariableOp'^Densenet/dense_1/MatMul/ReadVariableOp(^Densenet/dense_2/BiasAdd/ReadVariableOp'^Densenet/dense_2/MatMul/ReadVariableOp(^Densenet/dense_3/BiasAdd/ReadVariableOp'^Densenet/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2N
%Densenet/dense/BiasAdd/ReadVariableOp%Densenet/dense/BiasAdd/ReadVariableOp2L
$Densenet/dense/MatMul/ReadVariableOp$Densenet/dense/MatMul/ReadVariableOp2R
'Densenet/dense_1/BiasAdd/ReadVariableOp'Densenet/dense_1/BiasAdd/ReadVariableOp2P
&Densenet/dense_1/MatMul/ReadVariableOp&Densenet/dense_1/MatMul/ReadVariableOp2R
'Densenet/dense_2/BiasAdd/ReadVariableOp'Densenet/dense_2/BiasAdd/ReadVariableOp2P
&Densenet/dense_2/MatMul/ReadVariableOp&Densenet/dense_2/MatMul/ReadVariableOp2R
'Densenet/dense_3/BiasAdd/ReadVariableOp'Densenet/dense_3/BiasAdd/ReadVariableOp2P
&Densenet/dense_3/MatMul/ReadVariableOp&Densenet/dense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_2_layer_call_and_return_conditional_losses_227817

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
)__inference_Densenet_layer_call_fn_227209
dense_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_Densenet_layer_call_and_return_conditional_losses_2271692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namedense_input
?

?
A__inference_dense_layer_call_and_return_conditional_losses_227777

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_227797

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?-
?
I__inference_network_model_layer_call_and_return_conditional_losses_227578
input_1?
-densenet_dense_matmul_readvariableop_resource:
<
.densenet_dense_biasadd_readvariableop_resource:
A
/densenet_dense_1_matmul_readvariableop_resource:

>
0densenet_dense_1_biasadd_readvariableop_resource:
A
/densenet_dense_2_matmul_readvariableop_resource:
>
0densenet_dense_2_biasadd_readvariableop_resource:A
/densenet_dense_3_matmul_readvariableop_resource:>
0densenet_dense_3_biasadd_readvariableop_resource:
identity??%Densenet/dense/BiasAdd/ReadVariableOp?$Densenet/dense/MatMul/ReadVariableOp?'Densenet/dense_1/BiasAdd/ReadVariableOp?&Densenet/dense_1/MatMul/ReadVariableOp?'Densenet/dense_2/BiasAdd/ReadVariableOp?&Densenet/dense_2/MatMul/ReadVariableOp?'Densenet/dense_3/BiasAdd/ReadVariableOp?&Densenet/dense_3/MatMul/ReadVariableOp?
$Densenet/dense/MatMul/ReadVariableOpReadVariableOp-densenet_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02&
$Densenet/dense/MatMul/ReadVariableOp?
Densenet/dense/MatMulMatMulinput_1,Densenet/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/MatMul?
%Densenet/dense/BiasAdd/ReadVariableOpReadVariableOp.densenet_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%Densenet/dense/BiasAdd/ReadVariableOp?
Densenet/dense/BiasAddBiasAddDensenet/dense/MatMul:product:0-Densenet/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/BiasAdd?
Densenet/dense/ReluReluDensenet/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense/Relu?
&Densenet/dense_1/MatMul/ReadVariableOpReadVariableOp/densenet_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02(
&Densenet/dense_1/MatMul/ReadVariableOp?
Densenet/dense_1/MatMulMatMul!Densenet/dense/Relu:activations:0.Densenet/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/MatMul?
'Densenet/dense_1/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'Densenet/dense_1/BiasAdd/ReadVariableOp?
Densenet/dense_1/BiasAddBiasAdd!Densenet/dense_1/MatMul:product:0/Densenet/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/BiasAdd?
Densenet/dense_1/ReluRelu!Densenet/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Densenet/dense_1/Relu?
&Densenet/dense_2/MatMul/ReadVariableOpReadVariableOp/densenet_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&Densenet/dense_2/MatMul/ReadVariableOp?
Densenet/dense_2/MatMulMatMul#Densenet/dense_1/Relu:activations:0.Densenet/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/MatMul?
'Densenet/dense_2/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_2/BiasAdd/ReadVariableOp?
Densenet/dense_2/BiasAddBiasAdd!Densenet/dense_2/MatMul:product:0/Densenet/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/BiasAdd?
Densenet/dense_2/ReluRelu!Densenet/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_2/Relu?
&Densenet/dense_3/MatMul/ReadVariableOpReadVariableOp/densenet_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&Densenet/dense_3/MatMul/ReadVariableOp?
Densenet/dense_3/MatMulMatMul#Densenet/dense_2/Relu:activations:0.Densenet/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/MatMul?
'Densenet/dense_3/BiasAdd/ReadVariableOpReadVariableOp0densenet_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Densenet/dense_3/BiasAdd/ReadVariableOp?
Densenet/dense_3/BiasAddBiasAdd!Densenet/dense_3/MatMul:product:0/Densenet/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Densenet/dense_3/BiasAdd|
IdentityIdentity!Densenet/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^Densenet/dense/BiasAdd/ReadVariableOp%^Densenet/dense/MatMul/ReadVariableOp(^Densenet/dense_1/BiasAdd/ReadVariableOp'^Densenet/dense_1/MatMul/ReadVariableOp(^Densenet/dense_2/BiasAdd/ReadVariableOp'^Densenet/dense_2/MatMul/ReadVariableOp(^Densenet/dense_3/BiasAdd/ReadVariableOp'^Densenet/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2N
%Densenet/dense/BiasAdd/ReadVariableOp%Densenet/dense/BiasAdd/ReadVariableOp2L
$Densenet/dense/MatMul/ReadVariableOp$Densenet/dense/MatMul/ReadVariableOp2R
'Densenet/dense_1/BiasAdd/ReadVariableOp'Densenet/dense_1/BiasAdd/ReadVariableOp2P
&Densenet/dense_1/MatMul/ReadVariableOp&Densenet/dense_1/MatMul/ReadVariableOp2R
'Densenet/dense_2/BiasAdd/ReadVariableOp'Densenet/dense_2/BiasAdd/ReadVariableOp2P
&Densenet/dense_2/MatMul/ReadVariableOp&Densenet/dense_2/MatMul/ReadVariableOp2R
'Densenet/dense_3/BiasAdd/ReadVariableOp'Densenet/dense_3/BiasAdd/ReadVariableOp2P
&Densenet/dense_3/MatMul/ReadVariableOp&Densenet/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
I__inference_network_model_layer_call_and_return_conditional_losses_227282

inputs!
densenet_227264:

densenet_227266:
!
densenet_227268:


densenet_227270:
!
densenet_227272:

densenet_227274:!
densenet_227276:
densenet_227278:
identity?? Densenet/StatefulPartitionedCall?
 Densenet/StatefulPartitionedCallStatefulPartitionedCallinputsdensenet_227264densenet_227266densenet_227268densenet_227270densenet_227272densenet_227274densenet_227276densenet_227278*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_Densenet_layer_call_and_return_conditional_losses_2270632"
 Densenet/StatefulPartitionedCall?
IdentityIdentity)Densenet/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityq
NoOpNoOp!^Densenet/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 Densenet/StatefulPartitionedCall Densenet/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
__inference__traced_save_227970
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : :
:
:

:
:
:::: : : : : : :
:
:

:
:
::::
:
:

:
:
:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$	 

_output_shapes

:
: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
:  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::#

_output_shapes
: 
?
?
I__inference_network_model_layer_call_and_return_conditional_losses_227345

inputs!
densenet_227327:

densenet_227329:
!
densenet_227331:


densenet_227333:
!
densenet_227335:

densenet_227337:!
densenet_227339:
densenet_227341:
identity?? Densenet/StatefulPartitionedCall?
 Densenet/StatefulPartitionedCallStatefulPartitionedCallinputsdensenet_227327densenet_227329densenet_227331densenet_227333densenet_227335densenet_227337densenet_227339densenet_227341*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_Densenet_layer_call_and_return_conditional_losses_2271692"
 Densenet/StatefulPartitionedCall?
IdentityIdentity)Densenet/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityq
NoOpNoOp!^Densenet/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2D
 Densenet/StatefulPartitionedCall Densenet/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_228082
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: 1
assignvariableop_4_dense_kernel:
+
assignvariableop_5_dense_bias:
3
!assignvariableop_6_dense_1_kernel:

-
assignvariableop_7_dense_1_bias:
3
!assignvariableop_8_dense_2_kernel:
-
assignvariableop_9_dense_2_bias:4
"assignvariableop_10_dense_3_kernel:.
 assignvariableop_11_dense_3_bias:#
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: %
assignvariableop_16_total_2: %
assignvariableop_17_count_2: 9
'assignvariableop_18_adam_dense_kernel_m:
3
%assignvariableop_19_adam_dense_bias_m:
;
)assignvariableop_20_adam_dense_1_kernel_m:

5
'assignvariableop_21_adam_dense_1_bias_m:
;
)assignvariableop_22_adam_dense_2_kernel_m:
5
'assignvariableop_23_adam_dense_2_bias_m:;
)assignvariableop_24_adam_dense_3_kernel_m:5
'assignvariableop_25_adam_dense_3_bias_m:9
'assignvariableop_26_adam_dense_kernel_v:
3
%assignvariableop_27_adam_dense_bias_v:
;
)assignvariableop_28_adam_dense_1_kernel_v:

5
'assignvariableop_29_adam_dense_1_bias_v:
;
)assignvariableop_30_adam_dense_2_kernel_v:
5
'assignvariableop_31_adam_dense_2_bias_v:;
)assignvariableop_32_adam_dense_3_kernel_v:5
'assignvariableop_33_adam_dense_3_bias_v:
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_dense_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_2_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_2_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_3_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_1_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_1_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_2_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_2_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_3_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_3_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34f
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_35?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?v
?
	logic
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
k_default_save_signature
*l&call_and_return_all_conditional_losses
m__call__"
_tf_keras_model
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer_with_weights-2

layer-2
layer_with_weights-3
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"
_tf_keras_sequential
?
iter

beta_1

beta_2
	decaym[m\m]m^m_m`mambvcvdvevfvgvhvivj"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?
	variables

layers
regularization_losses
layer_metrics
non_trainable_variables
layer_regularization_losses
trainable_variables
 metrics
m__call__
k_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
pserving_default"
signature_map
?

kernel
bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
*q&call_and_return_all_conditional_losses
r__call__"
_tf_keras_layer
?

kernel
bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
*s&call_and_return_all_conditional_losses
t__call__"
_tf_keras_layer
?

kernel
bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
*u&call_and_return_all_conditional_losses
v__call__"
_tf_keras_layer
?

kernel
bias
-	variables
.regularization_losses
/trainable_variables
0	keras_api
*w&call_and_return_all_conditional_losses
x__call__"
_tf_keras_layer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
?
	variables

1layers
regularization_losses
2layer_metrics
3non_trainable_variables
4layer_regularization_losses
trainable_variables
5metrics
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
:	  (2	Adam/iter
:  (2Adam/beta_1
:  (2Adam/beta_2
:  (2
Adam/decay
 :
 2dense/kernel
:
 2
dense/bias
": 

 2dense_1/kernel
:
 2dense_1/bias
": 
 2dense_2/kernel
: 2dense_2/bias
":  2dense_3/kernel
: 2dense_3/bias
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
!	variables

9layers
"regularization_losses
:layer_metrics
;non_trainable_variables
<layer_regularization_losses
#trainable_variables
=metrics
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
%	variables

>layers
&regularization_losses
?layer_metrics
@non_trainable_variables
Alayer_regularization_losses
'trainable_variables
Bmetrics
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
)	variables

Clayers
*regularization_losses
Dlayer_metrics
Enon_trainable_variables
Flayer_regularization_losses
+trainable_variables
Gmetrics
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
-	variables

Hlayers
.regularization_losses
Ilayer_metrics
Jnon_trainable_variables
Klayer_regularization_losses
/trainable_variables
Lmetrics
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Mtotal
	Ncount
O	variables
P	keras_api"
_tf_keras_metric
^
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api"
_tf_keras_metric
^
	Vtotal
	Wcount
X
_fn_kwargs
Y	variables
Z	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
.
M0
N1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
%:#
 2Adam/dense/kernel/m
:
 2Adam/dense/bias/m
':%

 2Adam/dense_1/kernel/m
!:
 2Adam/dense_1/bias/m
':%
 2Adam/dense_2/kernel/m
!: 2Adam/dense_2/bias/m
':% 2Adam/dense_3/kernel/m
!: 2Adam/dense_3/bias/m
%:#
 2Adam/dense/kernel/v
:
 2Adam/dense/bias/v
':%

 2Adam/dense_1/kernel/v
!:
 2Adam/dense_1/bias/v
':%
 2Adam/dense_2/kernel/v
!: 2Adam/dense_2/bias/v
':% 2Adam/dense_3/kernel/v
!: 2Adam/dense_3/bias/v
?B?
!__inference__wrapped_model_226988input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_network_model_layer_call_and_return_conditional_losses_227485
I__inference_network_model_layer_call_and_return_conditional_losses_227516
I__inference_network_model_layer_call_and_return_conditional_losses_227547
I__inference_network_model_layer_call_and_return_conditional_losses_227578?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_network_model_layer_call_fn_227599
.__inference_network_model_layer_call_fn_227620
.__inference_network_model_layer_call_fn_227641
.__inference_network_model_layer_call_fn_227662?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Densenet_layer_call_and_return_conditional_losses_227693
D__inference_Densenet_layer_call_and_return_conditional_losses_227724
D__inference_Densenet_layer_call_and_return_conditional_losses_227233
D__inference_Densenet_layer_call_and_return_conditional_losses_227257?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_Densenet_layer_call_fn_227082
)__inference_Densenet_layer_call_fn_227745
)__inference_Densenet_layer_call_fn_227766
)__inference_Densenet_layer_call_fn_227209?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_227454input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_227777?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_227786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_227797?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_227806?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_227817?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_2_layer_call_fn_227826?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_227836?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_3_layer_call_fn_227845?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
D__inference_Densenet_layer_call_and_return_conditional_losses_227233o<?9
2?/
%?"
dense_input?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_Densenet_layer_call_and_return_conditional_losses_227257o<?9
2?/
%?"
dense_input?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_Densenet_layer_call_and_return_conditional_losses_227693j7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_Densenet_layer_call_and_return_conditional_losses_227724j7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
)__inference_Densenet_layer_call_fn_227082b<?9
2?/
%?"
dense_input?????????
p 

 
? "???????????
)__inference_Densenet_layer_call_fn_227209b<?9
2?/
%?"
dense_input?????????
p

 
? "???????????
)__inference_Densenet_layer_call_fn_227745]7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
)__inference_Densenet_layer_call_fn_227766]7?4
-?*
 ?
inputs?????????
p

 
? "???????????
!__inference__wrapped_model_226988q0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
C__inference_dense_1_layer_call_and_return_conditional_losses_227797\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????

? {
(__inference_dense_1_layer_call_fn_227806O/?,
%?"
 ?
inputs?????????

? "??????????
?
C__inference_dense_2_layer_call_and_return_conditional_losses_227817\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? {
(__inference_dense_2_layer_call_fn_227826O/?,
%?"
 ?
inputs?????????

? "???????????
C__inference_dense_3_layer_call_and_return_conditional_losses_227836\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_3_layer_call_fn_227845O/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_227777\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? y
&__inference_dense_layer_call_fn_227786O/?,
%?"
 ?
inputs?????????
? "??????????
?
I__inference_network_model_layer_call_and_return_conditional_losses_227485f3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
I__inference_network_model_layer_call_and_return_conditional_losses_227516f3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
I__inference_network_model_layer_call_and_return_conditional_losses_227547g4?1
*?'
!?
input_1?????????
p 
? "%?"
?
0?????????
? ?
I__inference_network_model_layer_call_and_return_conditional_losses_227578g4?1
*?'
!?
input_1?????????
p
? "%?"
?
0?????????
? ?
.__inference_network_model_layer_call_fn_227599Z4?1
*?'
!?
input_1?????????
p 
? "???????????
.__inference_network_model_layer_call_fn_227620Y3?0
)?&
 ?
inputs?????????
p 
? "???????????
.__inference_network_model_layer_call_fn_227641Y3?0
)?&
 ?
inputs?????????
p
? "???????????
.__inference_network_model_layer_call_fn_227662Z4?1
*?'
!?
input_1?????????
p
? "???????????
$__inference_signature_wrapper_227454|;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????