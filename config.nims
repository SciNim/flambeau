# TODO per file --experimental:views is needed
# Neither {.experimental: "views".}
# or a .nim.cfg with --experimental:views work,
# we need a global switch.
when not defined(torchInstaller):
  switch("experimental", "views")
include src/flambeau/libtorch
