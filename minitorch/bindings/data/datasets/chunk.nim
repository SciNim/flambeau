{.push header: "data/datasets/chunk.h".}


# Constructors and methods
proc constructor_BatchDataBuffer<UnwrappedBatch, ExampleSampler>*(batch_size: cint, example_sampler: var ExampleSampler, queue_capacity: cint): BatchDataBuffer {.constructor,importcpp: "BatchDataBuffer<UnwrappedBatch, ExampleSampler>(@)".}

proc constructor_UnwrappedBatchData*(data: torch::data::datasets::detail::BatchDataBuffer::UnwrappedBatchType): UnwrappedBatchData {.constructor,importcpp: "UnwrappedBatchData(@)".}

proc constructor_UnwrappedBatchData*(e: std::exception_ptr): UnwrappedBatchData {.constructor,importcpp: "UnwrappedBatchData(@)".}

proc constructor_ChunkDatasetOptions*(): ChunkDatasetOptions {.constructor,importcpp: "ChunkDatasetOptions".}

proc constructor_ChunkDatasetOptions*(preloader_count: cint, batch_size: cint, cache_size: cint, cross_chunk_shuffle_count: cint): ChunkDatasetOptions {.constructor,importcpp: "ChunkDatasetOptions(@)".}

proc constructor_ChunkDataset<ChunkReader, ChunkSampler, ExampleSampler>*(chunk_reader: ChunkReader, chunk_sampler: ChunkSampler, example_sampler: ExampleSampler, options: torch::data::datasets::ChunkDatasetOptions, preprocessing_policy: cint): ChunkDataset {.constructor,importcpp: "ChunkDataset<ChunkReader, ChunkSampler, ExampleSampler>(@)".}

proc read_chunk*(this: var ChunkDataReader, chunk_index: cint): torch::data::datasets::ChunkDataReader::ChunkType  {.importcpp: "read_chunk".}
  ## Read an entire chunk.

proc chunk_count*(this: var ChunkDataReader): int  {.importcpp: "chunk_count".}
  ## Returns the number of chunks available in this reader.

proc reset*(this: var ChunkDataReader)  {.importcpp: "reset".}
  ## This will clear any internal state associate with this reader.

proc get_batch*(this: var BatchDataBuffer): int  {.importcpp: "get_batch".}
  ## Return batch data from the queue. Called from the ChunkDataset main
  ## thread.

proc add_chunk_data*(this: var BatchDataBuffer, data: torch::data::datasets::detail::BatchDataBuffer::UnwrappedBatchType)  {.importcpp: "add_chunk_data".}
  ## Push preloaded chunks to batch queue. Called from the ChunkDataset
  ## worker threads.

proc add_chunk_data*(this: var BatchDataBuffer, e_ptr: std::exception_ptr)  {.importcpp: "add_chunk_data".}
  ## Push exceptions thrown during preloading into batch queue. Called from
  ## the ChunkDataset worker threads.

proc stop*(this: var BatchDataBuffer)  {.importcpp: "stop".}

proc TORCH_ARG*(this: var ChunkDatasetOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of worker thread to preload chunk data.

proc TORCH_ARG*(this: var ChunkDatasetOptions): int  {.importcpp: "TORCH_ARG".}
  ## The size of each batch.

proc TORCH_ARG*(this: var ChunkDatasetOptions): int  {.importcpp: "TORCH_ARG".}
  ## The capacity of the queue for batch caching.

proc TORCH_ARG*(this: var ChunkDatasetOptions): int  {.importcpp: "TORCH_ARG".}

proc get_batch*(this: var ChunkDataset): int  {.importcpp: "get_batch".}
  ## Default get_batch method of BatchDataset. This method returns Example
  ## batches created from the preloaded chunks. The implemenation is
  ## dataset agnostic and does not need overriding in different chunk
  ## datasets.

proc get_batch*(this: var ChunkDataset): int  {.importcpp: "get_batch".}
  ## Helper method around get_batch as `batch_size` is not strictly
  ## necessary

proc reset*(this: var ChunkDataset)  {.importcpp: "reset".}
  ## This will clear any internal state and starts the internal prefetching
  ## mechanism for the chunk dataset.

proc size*(this: ChunkDataset): int  {.importcpp: "size".}
  ## size is not used for chunk dataset.

proc chunk_sampler*(this: var ChunkDataset): torch::data::datasets::ChunkDataset::ChunkSamplerType  {.importcpp: "chunk_sampler".}

proc save*(this: ChunkDataset, archive: cint)  {.importcpp: "save".}

proc load*(this: var ChunkDataset, archive: cint)  {.importcpp: "load".}

proc preloader*(this: var ChunkDataset, id: cint)  {.importcpp: "preloader".}
  ## running on worker thread to preload chunk data.

proc free_workers*(this: var ChunkDataset)  {.importcpp: "free_workers".}
  ## Block the current thread until the workers finish execution and exit.

{.pop.} # header: "data/datasets/chunk.h
