
Ã
z_Z13gemv2N_kernelIiiffffLi128ELi2ELi4ELi4ELi1ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_8*28H@HHHXb#network_model/Densenet/dense/MatMulhu  aB
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28@@@H@bAdam/Powhu  ÈB
Å
z_Z13gemv2N_kernelIiiffffLi128ELi4ELi4ELi4ELi1ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_8*28@@@H@Xb%network_model/Densenet/dense_1/MatMulhu  aB

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28@@@H@b-Adam/Adam/update_7/update_0/ResourceApplyAdamhu  ÈB

§_Z13gemmk1_kernelIfLi256ELi5ELb0ELb0ELb1ELb0E30cublasGemvTensorStridedBatchedIKfES2_S0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_T9_N8biasTypeINS8_10value_typeES9_E4typeEE*288@8H8b5gradient_tape/network_model/Densenet/dense_3/MatMul_1hu  ÈB

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*288@8H8b-Adam/Adam/update_2/update_0/ResourceApplyAdamhu  ÈB

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*288@8H8b-Adam/Adam/update_4/update_0/ResourceApplyAdamhu  ÈB

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*288@8H8b-Adam/Adam/update_6/update_0/ResourceApplyAdamhu  ÈB
ê
¢_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*288@8H8b&mean_squared_error/weighted_loss/valuehu  ÈB
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel
*288@8H8bAdam/addhu  ÈB
Å
z_Z13gemv2N_kernelIiiffffLi128ELi4ELi4ELi4ELi1ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_8*288@8H8Xb%network_model/Densenet/dense_2/MatMulhu  aB
ß
_Z17gemv2T_kernel_valIiiffffLi128ELi16ELi2ELi2ELb0ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_T4_S7_$*288@8H8Xb3gradient_tape/network_model/Densenet/dense_3/MatMulhu  B
ß
_Z17gemv2T_kernel_valIiiffffLi128ELi16ELi2ELi2ELb0ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_T4_S7_$*288@8H8Xb3gradient_tape/network_model/Densenet/dense_2/MatMulhu  B
Î
¦_ZN10tensorflow7functor17BlockReduceKernelIPfNS_23TransformOutputIteratorIffNS0_9DividesByIffEExEELi256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS9_E10value_typeE0*288@8H8bMeanhu  ÈB

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*280@0H0b+Adam/Adam/update/update_0/ResourceApplyAdamhu  ÈB
Ý
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE!* 280@0H0b@gradient_tape/network_model/Densenet/dense_3/BiasAdd/BiasAddGradhu 	B
^
"Floor_GPU_DT_FLOAT_DT_FLOAT_kernel*280@0H0bAdam/ExponentialDecay/Floorhu  ÈB
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*280@0H0bAdam/ExponentialDecay/truedivhu  ÈB
Å
z_Z13gemv2N_kernelIiiffffLi128ELi2ELi4ELi4ELi1ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_8*280@0H0Xb%network_model/Densenet/dense_3/MatMulhu  aB
ß
_Z17gemv2T_kernel_valIiiffffLi128ELi16ELi2ELi2ELb0ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES3_S1_IfEfEEvT11_T4_S7_$*280@0H0Xb3gradient_tape/network_model/Densenet/dense_1/MatMulhu  B

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*280@0H0b-Adam/Adam/update_1/update_0/ResourceApplyAdamhu  ÈB

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*280@0H0b-Adam/Adam/update_3/update_0/ResourceApplyAdamhu  ÈB

O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*280@0H0b-Adam/Adam/update_5/update_0/ResourceApplyAdamhu  ÈB
D
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*28(@(H(bAbshu  ÈB
e
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28(@(H(b$gradient_tape/mean_squared_error/subhu  ÈB

§_Z13gemmk1_kernelIfLi256ELi5ELb0ELb0ELb1ELb0E30cublasGemvTensorStridedBatchedIKfES2_S0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_T9_N8biasTypeINS8_10value_typeES9_E4typeEE*28(@(H(b5gradient_tape/network_model/Densenet/dense_2/MatMul_1hu  ÈB
ÿ
§_Z13gemmk1_kernelIfLi256ELi5ELb0ELb0ELb1ELb0E30cublasGemvTensorStridedBatchedIKfES2_S0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_T9_N8biasTypeINS8_10value_typeES9_E4typeEE*28(@(H(Xb1gradient_tape/network_model/Densenet/dense/MatMulhu  ÈB
v
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28(@(H(b$network_model/Densenet/dense/BiasAddhu  ÈB
Ð
¢_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(bdiv_no_nan_1hu  ÈB

Â_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(b!network_model/Densenet/dense/Reluhu  ÈB
»
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(bAssignAddVariableOp_1hu  ÈB
»
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(bAssignAddVariableOp_2hu  ÈB
ò
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(b3gradient_tape/network_model/Densenet/dense/ReluGradhu  ÈB
ô
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(b5gradient_tape/network_model/Densenet/dense_2/ReluGradhu  ÈB
Ì
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28(@(H(b&Adam/Adam/update_0/AssignAddVariableOphu  ÈB
Ý
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE!* 28à'@à'Hà'b@gradient_tape/network_model/Densenet/dense_1/BiasAdd/BiasAddGradhu 	B

Â_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28à'@à'Hà'b#network_model/Densenet/dense_1/Reluhu  ÈB
Û
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE!* 28¡ @¡ H¡ b>gradient_tape/network_model/Densenet/dense/BiasAdd/BiasAddGradhu 	B
x
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28 @ H b&network_model/Densenet/dense_3/BiasAddhu  ÈB
I
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*28 @ H bCast_2hu  ÈB
U
"Cast_GPU_DT_DOUBLE_DT_FLOAT_kernel*28 @ H bnetwork_model/Casthu  ÈB
K
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*28 @ H b	Adam/Casthu  ÈB
M
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*28 @ H bAdam/Cast_1hu  ÈB
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*28 @ H b
LogicalAndhu  ÈB
V
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/ExponentialDecayhu  ÈB
Z
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/ExponentialDecay/Powhu  ÈB
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H b
Adam/Pow_1hu  ÈB
`
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bSquaredDifferencehu  ÈB

§_Z13gemmk1_kernelIfLi256ELi5ELb0ELb0ELb1ELb0E30cublasGemvTensorStridedBatchedIKfES2_S0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_T9_N8biasTypeINS8_10value_typeES9_E4typeEE*28 @ H b5gradient_tape/network_model/Densenet/dense_1/MatMul_1hu  ÈB
x
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28 @ H b&network_model/Densenet/dense_1/BiasAddhu  ÈB
x
1_ZN10tensorflow14BiasNHWCKernelIfEEviPKT_S3_PS1_i*28 @ H b&network_model/Densenet/dense_2/BiasAddhu  ÈB
Ð
¦_ZN10tensorflow7functor17BlockReduceKernelIPfNS_23TransformOutputIteratorIffNS0_9DividesByIffEExEELi256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS9_E10value_typeE0*28 @ H bMean_1hu  ÈB
Ý
z_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPKfPfN3cub3SumEEEvT_T0_iiT1_NSt15iterator_traitsIS7_E10value_typeE!* 28 @ H b@gradient_tape/network_model/Densenet/dense_2/BiasAdd/BiasAddGradhu 	B
Î
¢_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ H b
div_no_nanhu  ÈB
Ð
¢_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfLb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ H bdiv_no_nan_2hu  ÈB

Â_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_Li1EEEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ H b#network_model/Densenet/dense_2/Reluhu  ÈB
¹
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ H bAssignAddVariableOphu  ÈB
»
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ H bAssignAddVariableOp_3hu  ÈB
Â
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 @ H bupdate_0/AssignAddVariableOphu  ÈB
»
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbAssignAddVariableOp_4hu  ÈB
»
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28@HbAssignAddVariableOp_5hu  ÈB
ô
_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*28@Hb5gradient_tape/network_model/Densenet/dense_1/ReluGradhu  ÈB