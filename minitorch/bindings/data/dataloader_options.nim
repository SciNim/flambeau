{.push header: "data/dataloader_options.h".}


# Constructors and methods
proc constructor_DataLoaderOptions*(): DataLoaderOptions {.constructor,importcpp: "DataLoaderOptions".}

proc constructor_DataLoaderOptions*(batch_size: cint): DataLoaderOptions {.constructor,importcpp: "DataLoaderOptions(@)".}

proc constructor_FullDataLoaderOptions*(options: torch::data::DataLoaderOptions): FullDataLoaderOptions {.constructor,importcpp: "FullDataLoaderOptions(@)".}

proc TORCH_ARG*(this: var DataLoaderOptions): int  {.importcpp: "TORCH_ARG".}
  ## The size of each batch to fetch.

proc TORCH_ARG*(this: var DataLoaderOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of worker threads to launch. If zero, the main thread will
  ## synchronously perform the data loading.

proc TORCH_ARG*(this: var DataLoaderOptions): int  {.importcpp: "TORCH_ARG".}
  ## The maximum number of jobs to enqueue for fetching by worker threads.
  ## Defaults to two times the number of worker threads.

proc TORCH_ARG*(this: var DataLoaderOptions): int  {.importcpp: "TORCH_ARG".}
  ## An optional limit on the time to wait for the next batch.

proc TORCH_ARG*(this: var DataLoaderOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to enforce ordering of batches when multiple are loaded
  ## asynchronously by worker threads. Set to `false` for better
  ## performance if you do not care about determinism.

proc TORCH_ARG*(this: var DataLoaderOptions): int  {.importcpp: "TORCH_ARG".}
  ## Whether to omit the last batch if it contains less than `batch_size`
  ## examples.

{.pop.} # header: "data/dataloader_options.h
