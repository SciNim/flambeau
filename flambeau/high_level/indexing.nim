# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ../helpers/ast_utils,
  ../raw_bindings/tensors

# #######################################################################
#
#                      Slicing syntactic sugar
#
# #######################################################################
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/accessors_macros_syntax.nim

# Tensor of shape 5x5 of type "int" on backend "Cpu"
# |1      1       1       1       1|
# |2      4       8       16      32|
# |3      9       27      81      243|
# |4      16      64      256     1024|
# |5      25      125     625     3125|
#
#
# Slicing syntax:
#
# Basic indexing - foo[2, 3]
# Basic indexing - foo[1+1, 2*2*1]
# Basic slicing - foo[1..2, 3]
# Basic slicing - foo[1+1..4, 3-2..2]
# Span slices - foo[_, 3]
# Span slices - foo[1.._, 3]
# Span slices - foo[_..3, 3]
# Span slices - foo[_.._, 3]
# Stepping - foo[1..3|2, 3]
# Span stepping - foo[_.._|2, 3]
# Span stepping - foo[_.._|+2, 3]
# Span stepping - foo[1.._|1, 2..3]
# Span stepping - foo[_..<4|2, 3]
# Slicing until at n from the end - foo[0..^4, 3]
# Span Slicing until at n from the end - foo[_..^2, 3]
# Stepped Slicing until at n from the end - foo[1..^1|2, 3]
# Slice from the end - foo[^1..0|-1, 3]
# Slice from the end - expect non-negative step error - foo[^1..0, 3]
# Slice from the end - foo[^(2*2)..2*2, 3]
# Slice from the end - foo[^3..^2, 3]
#
# Important: Nim slices are inclusive while TorchSlice are exclusive!
#
# Note: This syntax sugar is actually never generated
#       When desugaring we directly generate the proper TorchSlice.
#       However, detecting whether we have integers, slices or tensors
#       for dispatch requires help for the type system, and so
#       all the sigils must be defined as properly typed procedures.

type Step = object
  ## Internal: Workaround to build ``TorchSlice`` without using parenthesis.
  ##
  ## Expected syntax is ``tensor[0..10|1]``.
  ##
  ## Due to operator precedence of ``|`` over ``..`` [0..10|1] is interpreted as [0..(10|1)]
  b: int
  step: int

func `|`*(s: Slice[int], step: int): TorchSlice {.inline.}=
  ## Internal: A ``TorchSlice`` constructor
  ## Input:
  ##     - a slice
  ##     - a step
  ## Returns:
  ##     - a ``TorchSlice``
  return torchSlice(s.a, s.b, step)

func `|`*(b, step: int): Step {.inline.}=
  ## Internal: A ``Step`` constructor
  ##
  ## ``Step`` is a workaround due to operator precedence.
  ##
  ## [0..10|1] is interpreted as [0..(10|1)]
  ## Input:
  ##     - the end of a slice range
  ##     - a step
  ## Returns:
  ##     - a ``Step``
  return Step(b: b, step: step)

func `|`*(ss: TorchSlice, step: int): TorchSlice {.inline.}=
  ## Internal: Modifies the step of a ``TorchSlice``
  ## Input:
  ##     - a ``TorchSlice``
  ##     - the new stepping
  ## Returns:
  ##     - a ``TorchSlice``
  return torchSlice(ss.start, ss.stop, step)

func `|+`*(s: Slice[int], step: int): TorchSlice {.inline.}=
  ## Internal: Alias for ``|``
  return `|`(s, step)

func `|+`*(b, step: int): Step {.inline.}=
  ## Internal: Alias for ``|``
  return `|`(b, step)

func `|+`*(ss: TorchSlice, step: int): TorchSlice {.inline.}=
  ## Internal: Alias for ``|``
  return `|`(ss, step)

func `|-`*(s: Slice[int], step: int): TorchSlice {.inline.}=
  ## Internal: A ``TorchSlice`` constructor
  ##
  ## Workaround to tensor[slice|-1] being interpreted as [slice `|-` 1]
  ##
  ## Properly create ``TorchSlice`` with negative stepping
  return torchSlice(s.a, s.b, -step)

func `|-`*(b, step: int): Step {.inline.}=
  ## Internal: A ``TorchSlice`` constructor
  ##
  ## Workaround to tensor[0..10|-1] being intepreted as [0 .. (10 `|-` 1)]
  ##
  ## Properly create ``TorchSlice`` with negative stepping
  return Step(b: b, step: -step)

func `|-`*(ss: TorchSlice, step: int): TorchSlice {.inline.}=
  ## Internal: Modifies the step of a ``TorchSlice``
  ##
  ## Workaround to tensor[slice|-1] being interpreted as [slice `|-` 1]
  ##
  ## Properly create ``TorchSlice`` with negative stepping
  return torchSlice(ss.start, ss.stop, -step)

func `..`*(a: int, s: Step): TorchSlice {.inline.} =
  ## Internal: Build a TorchSlice from [a .. (b|step)] (workaround to operator precedence)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``TorchSlice``, end of range will be inclusive
  return torchSlice(a, s.b+1, s.step)

func `..<`*(a: int, s: Step): TorchSlice {.inline.} =
  ## Internal: Build a TorchSlice from [a ..< (b|step)] (workaround to operator precedence)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``TorchSlice``, end of range will be exclusive.
  return torchSlice(a, s.b, s.step)

func `..^`*(a: int, s: Step): TorchSlice {.inline.} =
  ## Internal: Build a TorchSlice from [a ..^ (b|step)] (workaround to operator precedence and ..^b not being interpreted as .. ^b)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``TorchSlice``, end of range will start at "b" away from the end
  return torchSlice(a, -s.b, s.step)

func `^`*(s: TorchSlice): TorchSlice {.inline.} =
  ## Internal: Prefix to a to indicate starting the slice at "a" away from the end
  ## Note: This does not automatically inverse stepping, what if we want ^5..^1
  return torchSlice(-s.start, s.stop, s.step)

func `^`*(s: Slice): TorchSlice {.inline.} =
  ## Internal: Prefix to a to indicate starting the slice at "a" away from the end
  ## Note: This does not automatically inverse stepping, what if we want ^5..^1
  return torchSlice(-s.a, s.b, 1)

# #######################################################################
#
#                          Slicing desugaring
#
# #######################################################################
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/private/p_accessors_macros_desugar.nim

import std/macros

func sliceNone(): NimNode =
  bindSym("SliceSpan")
func indexNone(): NimNode =
  bindSym("IndexNone")
func succ(node: NimNode): NimNode =
  newCall(bindsym"succ", node)
func `-`(node: NimNode): NimNode =
  newCall(bindsym"-", node)
func Slice(nodes: varargs[NimNode]): NimNode =
  result = newCall(bindSym"torchSlice")
  for node in nodes:
    result.add node

macro desugarSlices(args: untyped): void =
  ## Transform all syntactic sugar into Slice(start, stop, step)

  # echo "\n------------------\nOriginal tree"
  # echo args.treerepr
  var r = newNimNode(nnkArglist)

  for nnk in children(args):

    ###### Traverse top tree nodes and one-hot-encode the different conditions

    # Node is "_"
    let nnk_joker = eqIdent(nnk, "_")

    # Node is of the form "* .. *"
    let nnk0_inf_dotdot = (
      nnk.kind == nnkInfix and
      eqIdent(nnk[0], "..")
    )

    # Node is of the form "* ..< *" or "* ..^ *"
    let nnk0_inf_dotdot_alt = (
      nnk.kind == nnkInfix and (
        eqIdent(nnk[0], "..<") or
        eqident(nnk[0], "..^")
      )
    )

    # Node is of the form "* .. *", "* ..< *" or "* ..^ *"
    let nnk0_inf_dotdot_all = (
      nnk0_inf_dotdot or
      nnk0_inf_dotdot_alt
    )

    # Node is of the form "^ *"
    let nnk0_pre_hat = (
      nnk.kind == nnkPrefix and
      eqIdent(nnk[0], "^")
    )

    # Node is of the form "_ `op` *"
    let nnk1_joker = (
      nnk.kind == nnkInfix and
      eqIdent(nnk[1], "_")
    )

    # Node is of the form "_ `op` *"
    let nnk10_hat = (
      nnk.kind == nnkInfix and
      nnk[1].kind == nnkPrefix and
      eqident(nnk[1][0], "^")
    )

    # Node is of the form "* `op` _"
    let nnk2_joker = (
      nnk.kind == nnkInfix and
      eqident(nnk[2], "_")
    )

    # Node is of the form "* `op` * | *" or "* `op` * |+ *"
    let nnk20_bar_pos = (
      nnk.kind == nnkInfix and
      nnk[2].kind == nnkInfix and (
        eqident(nnk[2][0], "|") or
        eqIdent(nnk[2][0], "|+")
      )
    )

    # Node is of the form "* `op` * |- *"
    let nnk20_bar_min = (
      nnk.kind == nnkInfix and
      nnk[2].kind == nnkInfix and
      eqIdent(nnk[2][0], "|-")
    )

    # Node is of the form "* `op` * | *" or "* `op` * |+ *" or "* `op` * |- *"
    let nnk20_bar_all = nnk20_bar_pos or nnk20_bar_min

    # Node is of the form "* `op1` _ `op2` *"
    let nnk21_joker = (
      nnk.kind == nnkInfix and
      nnk[2].kind == nnkInfix and
      eqIdent(nnk[2][1], "_")
    )

    ###### Core desugaring logic
    if nnk_joker:
      ## [_, 3] into [{None, 3}]
      r.add(sliceNone())
    elif nnk0_inf_dotdot and nnk1_joker and nnk2_joker:
      ## [_.._, 3] into [{None, 3}]
      r.add(sliceNone())
    elif nnk0_inf_dotdot and nnk1_joker and nnk20_bar_all and nnk21_joker:
      ## [_.._|2, 3] into [{Slice(None, None, 2), 3}]
      ## [_.._|+2, 3] into [{Slice(None, None, 2), 3}]
      ## [_.._|-2 doesn't make sense and will throw out of bounds
      r.add(Slice(indexNone(), indexNone(), nnk[2][2]))
    elif nnk0_inf_dotdot_all and nnk1_joker and nnk20_bar_all:
      ## [_..10|1, 3] into [{Slice(None, 10+1, 1), 3}] (for inclusive)
      ## [_..^10|1, 3] into [{Slice(None, -10, 1), 3}]
      ## [_..<10|1, 3] into [{Slice(None, 10, 1), 3}] (exclusive)
      if nnk[0].eqident(".."):
        r.add Slice(indexNone(), succ(nnk[2][1]), nnk[2][2])
      elif nnk[0].eqident("..^"):
        r.add Slice(indexNone(), -nnk[2][1], nnk[2][2])
      elif nnk[0].eqident("..<"):
        r.add Slice(indexNone(), nnk[2][1], nnk[2][2])
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot_all and nnk1_joker:
      ## [_..10, 3] into [{Slice(None, 10+1), 3}]
      ## [_..^10, 3] into [{Slice(None, -10), 3}]
      ## [_..<10, 3] into [{Slice(None, 10), 3}]
      if nnk[0].eqident(".."):
        r.add Slice(indexNone(), succ(nnk[2]))
      elif nnk[0].eqident("..^"):
        r.add Slice(indexNone(), -nnk[2])
      elif nnk[0].eqident("..<"):
        r.add Slice(indexNone(), nnk[2])
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot and nnk2_joker:
      ## [1.._, 3] into [{Slice(1, None, None), 3}]
      r.add Slice(nnk[1])
    elif nnk0_inf_dotdot and nnk20_bar_pos and nnk21_joker:
      ## [1.._|1, 3] into [{Slice(1, None, 1), 3}]
      ## [1.._|+1, 3] into [{Slice(1, None, 1), 3}]
      r.add Slice(nnk[1], indexNone(), nnk[2][2])
    elif nnk0_inf_dotdot and nnk20_bar_min and nnk21_joker:
      ## Raise error on [5.._|-1, 3]
      raise newException(IndexDefect, "Please use explicit end of range " &
                       "instead of `_` " &
                       "when the steps are negative")
    elif nnk0_inf_dotdot_all and nnk10_hat and nnk20_bar_all:
      ## [^1..2|-1, 3] into [{Slice(-1, 2, -1), 3}]
      r.add Slice(-nnk[1][1], nnk[2][1], -nnk[2][2])
    elif nnk0_inf_dotdot_all and nnk10_hat:
      ## [^1..2*3, 3] into [{Slice(-1, 2*3 + 1), 3}]
      ## [^1..0, 3] into [{Slice(-1, 0 + 1), 3}]
      ## [^1..<10, 3] into [{Slice(-1, 10), 3}]
      ## [^10..^1, 3] into [{Slice(-10, -1), 3}]
      ## Note: apart from the last case, the other
      ## should throw a non-negative step error
      if nnk[0].eqident(".."):
        r.add Slice(nnk[1][1], succ(nnk[2]))
      elif nnk[0].eqident("..^"):
        r.add Slice(nnk[1][1], -nnk[2])
      elif nnk[0].eqident("..<"):
        r.add Slice(nnk[1][1], nnk[2])
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot_all and nnk20_bar_all:
      ## [1..10|1] into [{Slice(1, 10 + 1, 1)}]
      ## [1..^10|1] into [{Slice(1, -10, 1)}]
      ## [1..<10|1] into [{Slice(1, 10, 1)}]
      if nnk[0].eqident(".."):
        r.add Slice(nnk[1], succ(nnk[2][0]), nnk[2][1])
      elif nnk[0].eqident("..^"):
        r.add Slice(nnk[1], -nnk[2][0], nnk[2][1])
      elif nnk[0].eqident("..<"):
        r.add Slice(nnk[1], nnk[2][0], nnk[2][1])
      else:
        error "Unreachable"
    elif nnk0_inf_dotdot_all:
      ## [1..10] into [{Slice(1, 10 + 1)}]
      ## [1..^10] into [{Slice(1, -10)}]
      ## [1..<10] into [{Slice(1, 10)}]
      if nnk[0].eqident(".."):
        r.add Slice(nnk[1], succ(nnk[2]))
      elif nnk[0].eqident("..^"):
        r.add Slice(nnk[1], -nnk[2])
      elif nnk[0].eqident("..<"):
        r.add Slice(nnk[1], nnk[2])
      else:
        error "Unreachable"
    elif nnk0_pre_hat:
      ## [^2, 3] into [^2..^2|1, 3]
      r.add(-nnk[1])
    else:
      r.add(nnk)
  # echo "\nAfter modif"
  # echo r.treerepr
  return r

# #######################################################################
#
#                          Slicing dispatch
#
# #######################################################################
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/private/p_accessors_macros_read.nim
# https://github.com/mratsim/Arraymancer/blob/bdcdfe1/src/arraymancer/tensor/private/p_accessors_macros_write.nim

type FancySelectorKind* = enum
  FancyNone
  FancyIndex
  FancyMaskFull
  FancyMaskAxis
  # Workaround needed for https://github.com/nim-lang/Nim/issues/14021
  FancyUnknownFull
  FancyUnknownAxis

proc getFancySelector(ast: NimNode, axis: var int, selector: var NimNode): FancySelectorKind =
  ## Detect indexing in the form
  ##   - "tensor[_, _, [0, 1, 4], _, _]
  ##   - "tensor[_, _, [0, 1, 4], `...`]
  ##  or with the index selector being a tensor
  result = FancyNone
  var foundNonSpanOrEllipsis = false
  var ellipsisAtStart = false

  template checkNonSpan(): untyped {.dirty.} =
    doAssert not foundNonSpanOrEllipsis,
        "Fancy indexing is only compatible with full spans `_` on non-indexed dimensions" &
        " and/or ellipsis `...`"

  var i = 0
  while i < ast.len:
    let cur = ast[i]
    # Important: sameType doesn't work for generic type like Array, Seq or Tensors ...
    #            https://github.com/nim-lang/Nim/issues/14021
    if cur.kind in {nnkIdent, nnkSym} and cur.eqIdent"SliceSpan":
      # Found a span
      discard
    elif (cur.kind == nnkCall and cur[0].eqIdent"torchSlice") or cur.isInt():
      doAssert result == FancyNone, "Internal FancyIndexing Error: Expected FancyNone, but got " & $result & " for AST: " & cur.repr()
      foundNonSpanOrEllipsis = true
    elif cur.eqIdent"IndexEllipsis":
      if i == ast.len - 1: # t[t.sum(axis = 1) >. 0.5, `...`]
        doAssert not ellipsisAtStart, "Cannot deduce the indexed/sliced dimensions due to ellipsis at the start and end of indexing."
        ellipsisAtStart = false
      elif i == 0: # t[`...`, t.sum(axis = 0) >. 0.5]
        ellipsisAtStart = true
      else:
        # t[0 ..< 10, `...`, t.sum(axis = 0) >. 0.5] is unsupported
        # so we tag as "foundNonSpanOrEllipsis"
        foundNonSpanOrEllipsis = true
    elif cur.kind == nnkBracket:
      checkNonSpan()
      axis = i
      if cur[0].kind == nnkIntLit:
        result = FancyIndex
        selector = cur
      elif cur[0].isBool():
        let full = i == 0 and ast.len == 1
        result = if full: FancyMaskFull else: FancyMaskAxis
        selector = cur
      else:
        # byte, char, enums are all represented by integers in the VM
        error "Fancy indexing is only possible with integers or booleans"
    else:
      checkNonSpan()
      axis = i
      let full = i == 0 and ast.len == 1
      result = if full: FancyUnknownFull else: FancyUnknownAxis
      selector = cur
    inc i

  # Handle ellipsis at the start
  if result != FancyNone and ellipsisAtStart:
    axis = ast.len - axis

  # replace all possible `nnkSym` by `idents` because we otherwise might get
  # type mismatches
  selector = replaceSymsByIdents(selector)

macro slice_typed_dispatch(t: typed, args: varargs[typed]): untyped =
  ## Typed macro so that isAllInt has typed context and we can dispatch.
  ## If args are all int, we dispatch to atIndex and return T
  ## Else, all ints are converted to SteppedSlices and we return a Tensor.
  ## Note, normal slices and `_` were already converted in the `[]` macro
  ## TODO in total we do 3 passes over the list of arguments :/. It is done only at compile time though

  # Point indexing
  # -----------------------------------------------------------------
  if isAllInt(args):
    result = newCall(bindSym"index", t)
    for slice in args:
      result.add(slice)
    return

  # Fancy indexing
  # -----------------------------------------------------------------
  # Cannot depend/bindSym the "selectors.nim" proc
  # Due to recursive module dependencies
  var selector: NimNode
  var axis: int
  let fancy = args.getFancySelector(axis, selector)
  if fancy == FancyIndex:
    return newCall(
        ident"index_select",
        t, newLit axis, selector
      )
  if fancy == FancyMaskFull:
    return newCall(
        ident"masked_select",
        t, selector
      )
  elif fancy == FancyMaskAxis:
    return newCall(
        ident"masked_axis_select",
        t, selector, newLit axis
      )

  # Slice indexing
  # -----------------------------------------------------------------
  if fancy == FancyNone:
    result = newCall(bindSym"index", t)
    for slice in args:
      result.add(slice)
    return

  # Fancy bug in Nim compiler
  # -----------------------------------------------------------------
  # We need to drop down to "when a is T" to infer what selector to call
  # as `getType`/`getTypeInst`/`getTypeImpl`/`sameType`
  # are buggy with generics
  # due to https://github.com/nim-lang/Nim/issues/14021
  let lateBind_masked_select = ident"masked_select"
  let lateBind_masked_axis_select = ident"masked_axis_select"
  let lateBind_index_select = ident"index_select"

  result = quote do:
    type FancyType = typeof(`selector`)
    when FancyType is (array or seq):
      type FancyTensorType = typeof(toTensor(`selector`))
    else:
      type FancyTensorType = FancyType
    when FancyTensorType is Tensor[bool]:
      when FancySelectorKind(`fancy`) == FancyUnknownFull:
        `lateBind_masked_select`(`t`, `selector`)
      elif FancySelectorKind(`fancy`) == FancyUnknownAxis:
        `lateBind_masked_axis_select`(`t`, `selector`, `axis`)
      else:
        {.error: "Unreachable".}
    else:
      `lateBind_index_select`(`t`, `axis`, `selector`)


macro slice_typed_dispatch_mut(t: typed, args: varargs[typed], val: typed): untyped =
  ## Assign `val` to Tensor T at slice/position `args`

  # Point indexing
  # -----------------------------------------------------------------
  if isAllInt(args):
    result = newCall(bindSym"index_put", t)
    for slice in args:
      result.add(slice)
    result.add(val)
    return

  # Fancy indexing
  # -----------------------------------------------------------------
  # Cannot depend/bindSym the "selectors.nim" proc
  # Due to recursive module dependencies
  var selector: NimNode
  var axis: int
  let fancy = args.getFancySelector(axis, selector)
  if fancy == FancyIndex:
    return newCall(
        ident"index_fill_mut",
        t, newLit axis, selector,
        val
      )
  if fancy == FancyMaskFull:
    return newCall(
        ident"masked_fill_mut",
        t, selector,
        val
      )
  elif fancy == FancyMaskAxis:
    return newCall(
        ident"masked_axis_fill_mut",
        t, selector, newLit axis,
        val
      )

  # Slice indexing
  # -----------------------------------------------------------------
  if fancy == FancyNone:
    result = newCall(bindSym"index_put", t)
    for slice in args:
      result.add(slice)
    result.add(val)
    return

  # Fancy bug in Nim compiler
  # -----------------------------------------------------------------
  # We need to drop down to "when a is T" to infer what selector to call
  # as `getType`/`getTypeInst`/`getTypeImpl`/`sameType`
  # are buggy with generics
  # due to https://github.com/nim-lang/Nim/issues/14021
  let lateBind_masked_fill = ident"masked_fill"
  let lateBind_masked_axis_fill = ident"masked_axis_fill"
  let lateBind_index_fill = ident"index_fill"

  result = quote do:
    type FancyType = typeof(`selector`)
    when FancyType is (array or seq):
      type FancyTensorType = typeof(toTensor(`selector`))
    else:
      type FancyTensorType = FancyType
    when FancyTensorType is Tensor[bool]:
      when FancySelectorKind(`fancy`) == FancyUnknownFull:
        `lateBind_masked_fill`(`t`, `selector`, `val`)
      elif FancySelectorKind(`fancy`) == FancyUnknownAxis:
        `lateBind_masked_axis_fill`(`t`, `selector`, `axis`, `val`)
      else:
        {.error: "Unreachable".}
    else:
      `lateBind_index_fill`(`t`, `axis`, `selector`, `val`)

# #######################################################################
#
#                        Public fancy indexers
#
# #######################################################################

macro `[]`*(t: Tensor, args: varargs[untyped]): untyped =
  ## Slice a Tensor
  ## Input:
  ##   - a Tensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a tensor corresponding to the slice
  ##
  ## Usage:
  ##    - Basic indexing - foo[2, 3]
  ##    - Basic indexing - foo[1+1, 2*2*1]
  ##    - Basic slicing - foo[1..2, 3]
  ##    - Basic slicing - foo[1+1..4, 3-2..2]
  ##    - Span slices - foo[_, 3]
  ##    - Span slices - foo[1.._, 3]
  ##    - Span slices - foo[_..3, 3]
  ##    - Span slices - foo[_.._, 3]
  ##    - Stepping - foo[1..3\|2, 3]
  ##    - Span stepping - foo[_.._\|2, 3]
  ##    - Span stepping - foo[_.._\|+2, 3]
  ##    - Span stepping - foo[1.._\|1, 2..3]
  ##    - Span stepping - foo[_..<4\|2, 3]
  ##    - Slicing until at n from the end - foo[0..^4, 3]
  ##    - Span Slicing until at n from the end - foo[_..^2, 3]
  ##    - Stepped Slicing until at n from the end - foo[1..^1\|2, 3]
  ##    - Slice from the end - foo[^1..0\|-1, 3]
  ##    - Slice from the end - expect non-negative step error - foo[^1..0, 3]
  ##    - Slice from the end - foo[^(2*2)..2*2, 3]
  ##    - Slice from the end - foo[^3..^2, 3]
  let new_args = getAST(desugarSlices(args))

  result = quote do:
    slice_typed_dispatch(`t`, `new_args`)

macro `[]=`*(t: var Tensor, args: varargs[untyped]): untyped =
  ## Modifies a tensor inplace at the corresponding location or slice
  ##
  ##
  ## Input:
  ##   - a ``var`` tensor
  ##   - a location:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ##   - a value:
  ##     - a single value that will
  ##       - replace the value at the specific coordinates
  ##       - or be applied to the whole slice
  ##     - an openarray with a shape that matches the slice
  ##     - a tensor with a shape that matches the slice
  ## Result:
  ##   - Nothing, the tensor is modified in-place
  ## Usage:
  ##   - Assign a single value - foo[1..2, 3..4] = 999
  ##   - Assign an array/seq of values - foo[0..1,0..1] = [[111, 222], [333, 444]]
  ##   - Assign values from a view/Tensor - foo[^2..^1,2..4] = bar
  ##   - Assign values from the same Tensor - foo[^2..^1,2..4] = foo[^1..^2|-1, 4..2|-1]

  # varargs[untyped] consumes all arguments so the actual value should be popped
  # https://github.com/nim-lang/Nim/issues/5855

  var tmp = args
  let val = tmp.pop
  let new_args = getAST(desugarSlices(tmp))

  result = quote do:
    slice_typed_dispatch_mut(`t`, `new_args`,`val`)
