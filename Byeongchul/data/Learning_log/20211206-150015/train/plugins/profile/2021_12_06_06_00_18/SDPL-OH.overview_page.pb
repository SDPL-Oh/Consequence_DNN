?	??ם?@??ם?@!??ם?@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??ם?@??V?c#`?1?[[%X??AE???V	V?IE?e?????r0*D???Ԅ?@)      P=2T
Iterator::Prefetch::Generatorf?ʉv??!$??kQW@)f?ʉv??1$??kQW@:Preprocessing2O
Iterator::Root::Prefetchiq?0'h??!r?;?@%??)iq?0'h??1r?;?@%??:Preprocessing2I
Iterator::Prefetch)A?G???!k??v????))A?G???1k??v????:Preprocessing2?
SIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeat?f+/????!!U}?????)W	?3???1??J?????:Preprocessing2V
Iterator::Root::Prefetch::ShardK"? ˂??!?w4_{??)???$???1?~,N???:Preprocessing2E
Iterator::Root6<?R?!??!g?4?/G??)???C?r??1?G?M?C??:Preprocessing2?
MIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap?d?????!{??;????)?????^??1????./??:Preprocessing2s
<Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2???uR_v?!yiށQ9??)???uR_v?1yiށQ9??:Preprocessing2d
-Iterator::Root::Prefetch::Shard::Rebatch::Map?	L?u??!?⃄b???)?y?ؘ?q?1u\)?s???:Preprocessing2?
]Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice]N	?I?p?!CD?4[??)]N	?I?p?1CD?4[??:Preprocessing2x
AIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip?=?#d??!??w??@)K?8???l?1??˙????:Preprocessing2_
(Iterator::Root::Prefetch::Shard::Rebatch?ۼqR??!?N?;p???)??>V??`?1Ю?6???:Preprocessing2?
_Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??U?ZN?!?,?$1??)??U?ZN?1?,?$1??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?84.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI:،%U@Q.???.@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??V?c#`???V?c#`?!??V?c#`?      ??!       "	?[[%X???[[%X??!?[[%X??*      ??!       2	E???V	V?E???V	V?!E???V	V?:	E?e?????E?e?????!E?e?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q:،%U@y.???.@?"A
#network_model/Densenet/dense/MatMulMatMul?w??????!?w??????0"T
-Adam/Adam/update_7/update_0/ResourceApplyAdamResourceApplyAdam??e?P???!?,٥c??"!
Adam/PowPow??e?P???!?}?'???"C
%network_model/Densenet/dense_1/MatMulMatMul??e?P???!???R{???0"T
-Adam/Adam/update_2/update_0/ResourceApplyAdamResourceApplyAdamk_?e???!?? ??޽?"T
-Adam/Adam/update_4/update_0/ResourceApplyAdamResourceApplyAdamk_?e???!?Q? ????"T
-Adam/Adam/update_6/update_0/ResourceApplyAdamResourceApplyAdamk_?e???!?h??)??"Q
5gradient_tape/network_model/Densenet/dense_3/MatMul_1MatMulk_?e???!x,?????"D
&mean_squared_error/weighted_loss/valueDivNoNank_?e???!??Sd??"#
Adam/addAddV2?ٲ????!?:*2??Q      Y@Y?E]t??@a?.???Q@q??*?a*X@y?}?'???"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?84.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 