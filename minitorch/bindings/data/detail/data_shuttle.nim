{.push header: "data/detail/data_shuttle.h".}


# Constructors and methods
proc push_job*(this: var DataShuttle, job: Job)  {.importcpp: "push_job".}
  ## Pushes a new job. Called by the main thread.

proc push_result*(this: var DataShuttle, result: Result)  {.importcpp: "push_result".}
  ## Pushes the result of a job. Called by worker threads.

proc pop_job*(this: var DataShuttle): Job  {.importcpp: "pop_job".}
  ## Returns the next job, blocking until there is one available. Called by
  ## worker threads.

proc pop_result*(this: var DataShuttle): int  {.importcpp: "pop_result".}
  ## Returns the result of a job, or nullopt if all jobs were exhausted.
  ## Called by the main thread.

proc drain*(this: var DataShuttle)  {.importcpp: "drain".}
  ## Discards any jobs that are not yet in flight, and waits for all in-
  ## flight jobs to finish, discarding their result.

proc in_flight_jobs*(this: DataShuttle): int  {.importcpp: "in_flight_jobs".}
  ## Returns the number of jobs that are still in progress. When this
  ## number is zero, an epoch is finished.

{.pop.} # header: "data/detail/data_shuttle.h
