	??ם?@??ם?@!??ם?@      ??!       "q
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
	??V?c#`???V?c#`?!??V?c#`?      ??!       "	?[[%X???[[%X??!?[[%X??*      ??!       2	E???V	V?E???V	V?!E???V	V?:	E?e?????E?e?????!E?e?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q:،%U@y.???.@