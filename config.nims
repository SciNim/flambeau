# TODO per file --experimental:views is needed
# Neither {.experimental: "views".}
# or a .nim.cfg with --experimental:views work,
# we need a global switch.
#when not defined(torchInstaller):
#  switch("experimental", "views")

switch("threads", "on")
when defined(windows):
  switch("passC", "/DNOMINMAX")
  switch("cc", "vcc")
  switch("passC", "/W0") # The console is flooded with NOTES and WARNINGS from vcc, this disables them.
