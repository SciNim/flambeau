# Special emit - https://github.com/nim-lang/Nim/blob/acf3715/compiler/ccgstmts.nim#L1489-L1495

template emitTypes*(typesection: static string): untyped =
  ## Emit a C/C++ typesection
  {.emit: "/*TYPESECTION*/\n" & typesection.}

template emitGlobals*(globals: static string): untyped =
  ## Emit a C/C++ global variable declaration
  {.emit: "/*VARSECTION*/\n" & globals.}

template emitIncludes*(includes: static string): untyped =
  ## Emit a C/C++ global variable declaration
  {.emit: "/*INCLUDESECTION*/\n" & includes.}
