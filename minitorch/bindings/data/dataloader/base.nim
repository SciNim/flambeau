{.push header: "data/dataloader/base.h".}


# Constructors and methods
proc constructor_DataLoaderBase<Dataset, Batch, BatchRequest>*(options: cint, main_thread_dataset: cint): DataLoaderBase {.constructor,importcpp: "DataLoaderBase<Dataset, Batch, BatchRequest>(@)".}
  ## Constructs a new DataLoader from a `dataset` to sample from, `options`
  ## to configure the DataLoader with, and a `sampler` that specifies the
  ## sampling strategy.

proc constructor_Sequenced*(): Sequenced {.constructor,importcpp: "Sequenced".}

proc constructor_Sequenced*(sqn: cint): Sequenced {.constructor,importcpp: "Sequenced(@)".}

proc constructor_Job*(): Job {.constructor,importcpp: "Job".}

proc constructor_Job*(q: torch::data::DataLoaderBase::QuitWorker, sqn: cint): Job {.constructor,importcpp: "Job(@)".}

proc constructor_Job*(i: var BatchRequest &, sqn: cint): Job {.constructor,importcpp: "Job(@)".}

proc constructor_Result*(): Result {.constructor,importcpp: "Result".}

proc begin*(this: var DataLoaderBase): int  {.importcpp: "begin".}
  ## Returns an iterator into the DataLoader. The lifetime of the iterator
  ## is bound to the DataLoader. In C++ standards language, the category of
  ## the iterator is `OutputIterator`. See
  ## https://en.cppreference.com/w/cpp/named_req/OutputIterator for what
  ## this means. In short: you may increment the iterator and dereference
  ## it, but cannot go back, or step forward more than one position at a
  ## time. When the DataLoader is exhausted, it will compare equal with the
  ## special "sentinel" iterator returned by `DataLoader::end()`. Most of
  ## the time, you should only use range-for loops to loop over the
  ## DataLoader, but standard algorithms like
  ## `std::copy(dataloader.begin(), dataloader.end(), output_iterator)` are
  ## supported too.

proc end*(this: var DataLoaderBase): int  {.importcpp: "end".}
  ## Returns a special "sentinel" iterator that compares equal with a non-
  ## sentinel iterator once the DataLoader is exhausted.

proc join*(this: var DataLoaderBase)  {.importcpp: "join".}
  ## Joins the DataLoader's worker threads and drains internal queues. This
  ## function may only be invoked from the main thread (in which the
  ## DataLoader lives).

proc options*(this: DataLoaderBase): int  {.importcpp: "options".}
  ## Returns the options with which the DataLoader was configured.

proc batch*(this: var Result, b: cint): torch::data::DataLoaderBase::Result  {.importcpp: "batch".}

proc get_batch_request*(this: var DataLoaderBase): int  {.importcpp: "get_batch_request".}
  ## Subclass hook for getting the next batch request. The stateless case
  ## will ask the sampler for a new batch request (e.g. a vector of
  ## indices), while the stateful one will simply return the batch size.

proc reset*(this: var DataLoaderBase)  {.importcpp: "reset".}
  ## Resets the internal state of the DataLoader, optionally pre-fetching
  ## new jobs.

proc prefetch*(this: var DataLoaderBase, requested_jobs: cint)  {.importcpp: "prefetch".}
  ## Schedules `requested_jobs` many new batches to be fetched. The actual
  ## number of jobs scheduled may be less if the DataLoader exhausts.

proc prefetch*(this: var DataLoaderBase)  {.importcpp: "prefetch".}
  ## Schedules the maximum number of jobs (based on the `max_jobs` option).

proc next*(this: var DataLoaderBase): int  {.importcpp: "next".}
  ## Returns the next batch of data, or an empty `optional` if the
  ## DataLoader is exhausted. This operation will block until a batch is
  ## available if one is still expected.

proc worker_thread*(this: var DataLoaderBase, dataset: var Dataset)  {.importcpp: "worker_thread".}
  ## The function that worker threads run.

proc pop_result*(this: var DataLoaderBase): int  {.importcpp: "pop_result".}
  ## Convenience method that gets the next result from the sequencer.

{.pop.} # header: "data/dataloader/base.h
