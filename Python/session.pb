
F
x_inputPlaceholder*!
shape:??????????*
dtype0
J
Reshape/shapeConst*
dtype0*%
valueB"????   ?      
A
ReshapeReshapex_inputReshape/shape*
T0*
Tshape0
E
PlaceholderPlaceholder*
shape:?????????*
dtype0
S
truncated_normal/shapeConst*
dtype0*%
valueB"            
B
truncated_normal/meanConst*
dtype0*
valueB
 *    
D
truncated_normal/stddevConst*
dtype0*
valueB
 *???=
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
T0*
seed2 *
dtype0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
d
Variable
VariableV2*
shared_name *
shape:*
dtype0*
	container 
?
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
I
Variable/readIdentityVariable*
_class
loc:@Variable*
T0
6
ConstConst*
dtype0*
valueB<*    
Z

Variable_1
VariableV2*
shared_name *
shape:<*
dtype0*
	container 

Variable_1/AssignAssign
Variable_1Const*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(
O
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0
L
depthwise/ShapeConst*
dtype0*%
valueB"            
L
depthwise/dilation_rateConst*
dtype0*
valueB"      
?
	depthwiseDepthwiseConv2dNativeReshapeVariable/read*
strides
*
T0*
paddingVALID*
data_formatNHWC
/
AddAdd	depthwiseVariable_1/read*
T0

ReluReluAdd*
T0
t
MaxPoolMaxPoolRelu*
strides
*
T0*
ksize
*
paddingVALID*
data_formatNHWC
U
truncated_normal_1/shapeConst*
dtype0*%
valueB"      <      
D
truncated_normal_1/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_1/stddevConst*
dtype0*
valueB
 *???=
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
seed2 *
dtype0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
f

Variable_2
VariableV2*
shared_name *
shape:<*
dtype0*
	container 
?
Variable_2/AssignAssign
Variable_2truncated_normal_1*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(
O
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0
8
Const_1Const*
dtype0*
valueBx*    
Z

Variable_3
VariableV2*
shared_name *
shape:x*
dtype0*
	container 
?
Variable_3/AssignAssign
Variable_3Const_1*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(
O
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0
N
depthwise_1/ShapeConst*
dtype0*%
valueB"      <      
N
depthwise_1/dilation_rateConst*
dtype0*
valueB"      
?
depthwise_1DepthwiseConv2dNativeMaxPoolVariable_2/read*
strides
*
T0*
paddingVALID*
data_formatNHWC
3
Add_1Adddepthwise_1Variable_3/read*
T0

Relu_1ReluAdd_1*
T0
D
Reshape_1/shapeConst*
dtype0*
valueB"????h  
D
	Reshape_1ReshapeRelu_1Reshape_1/shape*
T0*
Tshape0
M
truncated_normal_2/shapeConst*
dtype0*
valueB"h  d   
D
truncated_normal_2/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_2/stddevConst*
dtype0*
valueB
 *???=
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
T0*
seed2 *
dtype0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
_

Variable_4
VariableV2*
shared_name *
shape:	? d*
dtype0*
	container 
?
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(
O
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0
8
Const_2Const*
dtype0*
valueBd*    
Z

Variable_5
VariableV2*
shared_name *
shape:d*
dtype0*
	container 
?
Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(
O
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0
[
MatMulMatMul	Reshape_1Variable_4/read*
transpose_b( *
T0*
transpose_a( 
.
Add_2AddMatMulVariable_5/read*
T0

TanhTanhAdd_2*
T0
M
truncated_normal_3/shapeConst*
dtype0*
valueB"d      
D
truncated_normal_3/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_3/stddevConst*
dtype0*
valueB
 *???=
~
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
T0*
seed2 *
dtype0
e
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0
S
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0
^

Variable_6
VariableV2*
shared_name *
shape
:d*
dtype0*
	container 
?
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(
O
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0
8
Const_3Const*
dtype0*
valueB*    
Z

Variable_7
VariableV2*
shared_name *
shape:*
dtype0*
	container 
?
Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(
O
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0
X
MatMul_1MatMulTanhVariable_6/read*
transpose_b( *
T0*
transpose_a( 
.
addAddMatMul_1Variable_7/read*
T0
&
labels_outputSoftmaxadd*
T0
"
LogLoglabels_output*
T0
%
mulMulPlaceholderLog*
T0
<
Const_4Const*
dtype0*
valueB"       
>
SumSummulConst_4*
	keep_dims( *
T0*

Tidx0

NegNegSum*
T0
8
gradients/ShapeConst*
dtype0*
valueB 
<
gradients/ConstConst*
dtype0*
valueB
 *  ??
A
gradients/FillFillgradients/Shapegradients/Const*
T0
6
gradients/Neg_grad/NegNeggradients/Fill*
T0
U
 gradients/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
v
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
T0*
Tshape0
?
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0
p
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*

Tmultiples0*
T0
G
gradients/mul_grad/ShapeShapePlaceholder*
T0*
out_type0
A
gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0
?
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0
D
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0
?
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
n
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0
N
gradients/mul_grad/mul_1MulPlaceholdergradients/Sum_grad/Tile*
T0
?
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
t
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
?
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0
?
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0
s
gradients/Log_grad/Reciprocal
Reciprocallabels_output.^gradients/mul_grad/tuple/control_dependency_1*
T0
t
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0
W
 gradients/labels_output_grad/mulMulgradients/Log_grad/mullabels_output*
T0
`
2gradients/labels_output_grad/Sum/reduction_indicesConst*
dtype0*
valueB:
?
 gradients/labels_output_grad/SumSum gradients/labels_output_grad/mul2gradients/labels_output_grad/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0
_
*gradients/labels_output_grad/Reshape/shapeConst*
dtype0*
valueB"????   
?
$gradients/labels_output_grad/ReshapeReshape gradients/labels_output_grad/Sum*gradients/labels_output_grad/Reshape/shape*
T0*
Tshape0
n
 gradients/labels_output_grad/subSubgradients/Log_grad/mul$gradients/labels_output_grad/Reshape*
T0
c
"gradients/labels_output_grad/mul_1Mul gradients/labels_output_grad/sublabels_output*
T0
D
gradients/add_grad/ShapeShapeMatMul_1*
T0*
out_type0
H
gradients/add_grad/Shape_1Const*
dtype0*
valueB:
?
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0
?
gradients/add_grad/SumSum"gradients/labels_output_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0
?
gradients/add_grad/Sum_1Sum"gradients/labels_output_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
?
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0
?
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0
?
gradients/MatMul_1_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
T0*
transpose_a( 
?
 gradients/MatMul_1_grad/MatMul_1MatMulTanh+gradients/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
?
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
?
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0
i
gradients/Tanh_grad/TanhGradTanhGradTanh0gradients/MatMul_1_grad/tuple/control_dependency*
T0
D
gradients/Add_2_grad/ShapeShapeMatMul*
T0*
out_type0
J
gradients/Add_2_grad/Shape_1Const*
dtype0*
valueB:d
?
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0
?
gradients/Add_2_grad/SumSumgradients/Tanh_grad/TanhGrad*gradients/Add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
t
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*
Tshape0
?
gradients/Add_2_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
z
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
?
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_2_grad/Reshape*
T0
?
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1*
T0
?
gradients/MatMul_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
T0*
transpose_a( 
?
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/Add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
?
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
?
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
H
gradients/Reshape_1_grad/ShapeShapeRelu_1*
T0*
out_type0
?
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0
]
gradients/Relu_1_grad/ReluGradReluGrad gradients/Reshape_1_grad/ReshapeRelu_1*
T0
I
gradients/Add_1_grad/ShapeShapedepthwise_1*
T0*
out_type0
J
gradients/Add_1_grad/Shape_1Const*
dtype0*
valueB:x
?
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0
?
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
t
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0
?
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
z
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
?
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*
T0
?
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
T0
K
 gradients/depthwise_1_grad/ShapeShapeMaxPool*
T0*
out_type0
?
=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInput gradients/depthwise_1_grad/ShapeVariable_2/read-gradients/Add_1_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*
data_formatNHWC
_
"gradients/depthwise_1_grad/Shape_1Const*
dtype0*%
valueB"      <      
?
>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterMaxPool"gradients/depthwise_1_grad/Shape_1-gradients/Add_1_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*
data_formatNHWC
?
+gradients/depthwise_1_grad/tuple/group_depsNoOp>^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput?^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter
?
3gradients/depthwise_1_grad/tuple/control_dependencyIdentity=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput,^gradients/depthwise_1_grad/tuple/group_deps*P
_classF
DBloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput*
T0
?
5gradients/depthwise_1_grad/tuple/control_dependency_1Identity>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter,^gradients/depthwise_1_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter*
T0
?
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/depthwise_1_grad/tuple/control_dependency*
strides
*
T0*
ksize
*
paddingVALID*
data_formatNHWC
[
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0
E
gradients/Add_grad/ShapeShape	depthwise*
T0*
out_type0
H
gradients/Add_grad/Shape_1Const*
dtype0*
valueB:<
?
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0
?
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
n
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0
?
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
t
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
?
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*-
_class#
!loc:@gradients/Add_grad/Reshape*
T0
?
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
T0
I
gradients/depthwise_grad/ShapeShapeReshape*
T0*
out_type0
?
;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInputgradients/depthwise_grad/ShapeVariable/read+gradients/Add_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*
data_formatNHWC
]
 gradients/depthwise_grad/Shape_1Const*
dtype0*%
valueB"            
?
<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterReshape gradients/depthwise_grad/Shape_1+gradients/Add_grad/tuple/control_dependency*
strides
*
T0*
paddingVALID*
data_formatNHWC
?
)gradients/depthwise_grad/tuple/group_depsNoOp<^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput=^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter
?
1gradients/depthwise_grad/tuple/control_dependencyIdentity;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*^gradients/depthwise_grad/tuple/group_deps*N
_classD
B@loc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*
T0
?
3gradients/depthwise_grad/tuple/control_dependency_1Identity<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*^gradients/depthwise_grad/tuple/group_deps*O
_classE
CAloc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*
T0
c
beta1_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *fff?
t
beta1_power
VariableV2*
shared_name *
_class
loc:@Variable*
shape: *
dtype0*
	container 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
O
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
T0
c
beta2_power/initial_valueConst*
_class
loc:@Variable*
dtype0*
valueB
 *w??
t
beta2_power
VariableV2*
shared_name *
_class
loc:@Variable*
shape: *
dtype0*
	container 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
O
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0
y
Variable/Adam/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueB*    
?
Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
shape:*
dtype0*
	container 
?
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
S
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0
{
!Variable/Adam_1/Initializer/zerosConst*
_class
loc:@Variable*
dtype0*%
valueB*    
?
Variable/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable*
shape:*
dtype0*
	container 
?
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
W
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0
q
!Variable_1/Adam/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueB<*    
~
Variable_1/Adam
VariableV2*
shared_name *
_class
loc:@Variable_1*
shape:<*
dtype0*
	container 
?
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(
Y
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0
s
#Variable_1/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_1*
dtype0*
valueB<*    
?
Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
shape:<*
dtype0*
	container 
?
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(
]
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0
}
!Variable_2/Adam/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*%
valueB<*    
?
Variable_2/Adam
VariableV2*
shared_name *
_class
loc:@Variable_2*
shape:<*
dtype0*
	container 
?
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(
Y
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0

#Variable_2/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_2*
dtype0*%
valueB<*    
?
Variable_2/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_2*
shape:<*
dtype0*
	container 
?
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(
]
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0
q
!Variable_3/Adam/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBx*    
~
Variable_3/Adam
VariableV2*
shared_name *
_class
loc:@Variable_3*
shape:x*
dtype0*
	container 
?
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(
Y
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0
s
#Variable_3/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_3*
dtype0*
valueBx*    
?
Variable_3/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_3*
shape:x*
dtype0*
	container 
?
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(
]
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0
v
!Variable_4/Adam/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0*
valueB	? d*    
?
Variable_4/Adam
VariableV2*
shared_name *
_class
loc:@Variable_4*
shape:	? d*
dtype0*
	container 
?
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(
Y
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0
x
#Variable_4/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_4*
dtype0*
valueB	? d*    
?
Variable_4/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
shape:	? d*
dtype0*
	container 
?
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(
]
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0
q
!Variable_5/Adam/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBd*    
~
Variable_5/Adam
VariableV2*
shared_name *
_class
loc:@Variable_5*
shape:d*
dtype0*
	container 
?
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(
Y
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0
s
#Variable_5/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_5*
dtype0*
valueBd*    
?
Variable_5/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_5*
shape:d*
dtype0*
	container 
?
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(
]
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0
u
!Variable_6/Adam/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueBd*    
?
Variable_6/Adam
VariableV2*
shared_name *
_class
loc:@Variable_6*
shape
:d*
dtype0*
	container 
?
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(
Y
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
T0
w
#Variable_6/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_6*
dtype0*
valueBd*    
?
Variable_6/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_6*
shape
:d*
dtype0*
	container 
?
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(
]
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_class
loc:@Variable_6*
T0
q
!Variable_7/Adam/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    
~
Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
shape:*
dtype0*
	container 
?
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(
Y
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
T0
s
#Variable_7/Adam_1/Initializer/zerosConst*
_class
loc:@Variable_7*
dtype0*
valueB*    
?
Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
shape:*
dtype0*
	container 
?
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(
]
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_class
loc:@Variable_7*
T0
?
Adam/learning_rateConst*
dtype0*
valueB
 *??8
7

Adam/beta1Const*
dtype0*
valueB
 *fff?
7

Adam/beta2Const*
dtype0*
valueB
 *w??
9
Adam/epsilonConst*
dtype0*
valueB
 *w?+2
?
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/depthwise_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable*
T0*
use_nesterov( 
?
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_1*
T0*
use_nesterov( 
?
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon5gradients/depthwise_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_2*
T0*
use_nesterov( 
?
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_3*
T0*
use_nesterov( 
?
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_4*
T0*
use_nesterov( 
?
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_5*
T0*
use_nesterov( 
?
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_6*
T0*
use_nesterov( 
?
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_7*
T0*
use_nesterov( 
?
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0
{
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
_class
loc:@Variable*
T0*
validate_shape(
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
_class
loc:@Variable*
T0*
validate_shape(
?
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
:
ArgMax/dimensionConst*
dtype0*
value	B :
Y
ArgMaxArgMaxlabels_outputArgMax/dimension*
T0*
output_type0	*

Tidx0
<
ArgMax_1/dimensionConst*
dtype0*
value	B :
[
ArgMax_1ArgMaxPlaceholderArgMax_1/dimension*
T0*
output_type0	*

Tidx0
)
EqualEqualArgMaxArgMax_1*
T0	
+
CastCastEqual*

DstT0*

SrcT0

5
Const_5Const*
dtype0*
valueB: 
A
MeanMeanCastConst_5*
	keep_dims( *
T0*

Tidx0
8

save/ConstConst*
dtype0*
valueB Bmodel
?
save/SaveV2/tensor_namesConst*
dtype0*?
value?B?BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power
{
save/SaveV2/shape_and_slicesConst*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_powerbeta2_power*(
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0
P
save/RestoreV2/tensor_namesConst*
dtype0*
valueBBVariable
L
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B 
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2
~
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
W
save/RestoreV2_1/tensor_namesConst*
dtype0*"
valueBBVariable/Adam
N
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2
?
save/Assign_1AssignVariable/Adamsave/RestoreV2_1*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
Y
save/RestoreV2_2/tensor_namesConst*
dtype0*$
valueBBVariable/Adam_1
N
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2
?
save/Assign_2AssignVariable/Adam_1save/RestoreV2_2*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
T
save/RestoreV2_3/tensor_namesConst*
dtype0*
valueBB
Variable_1
N
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2
?
save/Assign_3Assign
Variable_1save/RestoreV2_3*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(
Y
save/RestoreV2_4/tensor_namesConst*
dtype0*$
valueBBVariable_1/Adam
N
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2
?
save/Assign_4AssignVariable_1/Adamsave/RestoreV2_4*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(
[
save/RestoreV2_5/tensor_namesConst*
dtype0*&
valueBBVariable_1/Adam_1
N
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2
?
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2_5*
use_locking(*
_class
loc:@Variable_1*
T0*
validate_shape(
T
save/RestoreV2_6/tensor_namesConst*
dtype0*
valueBB
Variable_2
N
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2
?
save/Assign_6Assign
Variable_2save/RestoreV2_6*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(
Y
save/RestoreV2_7/tensor_namesConst*
dtype0*$
valueBBVariable_2/Adam
N
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2
?
save/Assign_7AssignVariable_2/Adamsave/RestoreV2_7*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(
[
save/RestoreV2_8/tensor_namesConst*
dtype0*&
valueBBVariable_2/Adam_1
N
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2
?
save/Assign_8AssignVariable_2/Adam_1save/RestoreV2_8*
use_locking(*
_class
loc:@Variable_2*
T0*
validate_shape(
T
save/RestoreV2_9/tensor_namesConst*
dtype0*
valueBB
Variable_3
N
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2
?
save/Assign_9Assign
Variable_3save/RestoreV2_9*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(
Z
save/RestoreV2_10/tensor_namesConst*
dtype0*$
valueBBVariable_3/Adam
O
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2
?
save/Assign_10AssignVariable_3/Adamsave/RestoreV2_10*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(
\
save/RestoreV2_11/tensor_namesConst*
dtype0*&
valueBBVariable_3/Adam_1
O
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2
?
save/Assign_11AssignVariable_3/Adam_1save/RestoreV2_11*
use_locking(*
_class
loc:@Variable_3*
T0*
validate_shape(
U
save/RestoreV2_12/tensor_namesConst*
dtype0*
valueBB
Variable_4
O
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2
?
save/Assign_12Assign
Variable_4save/RestoreV2_12*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(
Z
save/RestoreV2_13/tensor_namesConst*
dtype0*$
valueBBVariable_4/Adam
O
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2
?
save/Assign_13AssignVariable_4/Adamsave/RestoreV2_13*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(
\
save/RestoreV2_14/tensor_namesConst*
dtype0*&
valueBBVariable_4/Adam_1
O
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2
?
save/Assign_14AssignVariable_4/Adam_1save/RestoreV2_14*
use_locking(*
_class
loc:@Variable_4*
T0*
validate_shape(
U
save/RestoreV2_15/tensor_namesConst*
dtype0*
valueBB
Variable_5
O
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2
?
save/Assign_15Assign
Variable_5save/RestoreV2_15*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(
Z
save/RestoreV2_16/tensor_namesConst*
dtype0*$
valueBBVariable_5/Adam
O
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2
?
save/Assign_16AssignVariable_5/Adamsave/RestoreV2_16*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(
\
save/RestoreV2_17/tensor_namesConst*
dtype0*&
valueBBVariable_5/Adam_1
O
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2
?
save/Assign_17AssignVariable_5/Adam_1save/RestoreV2_17*
use_locking(*
_class
loc:@Variable_5*
T0*
validate_shape(
U
save/RestoreV2_18/tensor_namesConst*
dtype0*
valueBB
Variable_6
O
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2
?
save/Assign_18Assign
Variable_6save/RestoreV2_18*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(
Z
save/RestoreV2_19/tensor_namesConst*
dtype0*$
valueBBVariable_6/Adam
O
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2
?
save/Assign_19AssignVariable_6/Adamsave/RestoreV2_19*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(
\
save/RestoreV2_20/tensor_namesConst*
dtype0*&
valueBBVariable_6/Adam_1
O
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2
?
save/Assign_20AssignVariable_6/Adam_1save/RestoreV2_20*
use_locking(*
_class
loc:@Variable_6*
T0*
validate_shape(
U
save/RestoreV2_21/tensor_namesConst*
dtype0*
valueBB
Variable_7
O
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2
?
save/Assign_21Assign
Variable_7save/RestoreV2_21*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(
Z
save/RestoreV2_22/tensor_namesConst*
dtype0*$
valueBBVariable_7/Adam
O
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2
?
save/Assign_22AssignVariable_7/Adamsave/RestoreV2_22*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(
\
save/RestoreV2_23/tensor_namesConst*
dtype0*&
valueBBVariable_7/Adam_1
O
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2
?
save/Assign_23AssignVariable_7/Adam_1save/RestoreV2_23*
use_locking(*
_class
loc:@Variable_7*
T0*
validate_shape(
V
save/RestoreV2_24/tensor_namesConst*
dtype0* 
valueBBbeta1_power
O
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2
?
save/Assign_24Assignbeta1_powersave/RestoreV2_24*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
V
save/RestoreV2_25/tensor_namesConst*
dtype0* 
valueBBbeta2_power
O
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2
?
save/Assign_25Assignbeta2_powersave/RestoreV2_25*
use_locking(*
_class
loc:@Variable*
T0*
validate_shape(
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25
?
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign"