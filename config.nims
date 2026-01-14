# TODO per file --experimental:views is needed
# Neither {.experimental: "views".}
# or a .nim.cfg with --experimental:views work,
# we need a global switch.
#when not defined(torchInstaller):
#  switch("experimental", "views")

switch("threads", "on")

# macOS Apple Silicon compatibility
# PyTorch's libtorch for macOS is x86_64 only, compile for x86_64 and run under Rosetta 2
when defined(macosx) and defined(arm64):
  switch("cpu", "amd64")
  switch("passC", "-arch x86_64")
  switch("passL", "-arch x86_64")

when defined(windows):
  switch("passC", "/DNOMINMAX")
  switch("cc", "vcc")
  switch("passC", "/W0") # The console is flooded with NOTES and WARNINGS from vcc, this disables them.
# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
# end Nimble config
