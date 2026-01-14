# Copyright 2017-2021 the Flambeau contributors

import ../tensors
import std/macros
# Reuse same desugar syntax as RawTensor
include ../raw/sugar/indexing_macros

macro `[]`*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  let new_args = getAST(desugarSlices(args))
  result = quote:
    slice_typed_dispatch(`t`, `new_args`)

macro `[]=`*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  var tmp = args
  let val = tmp.pop
  let new_args = getAST(desugarSlices(tmp))

  result = quote:
    slice_typed_dispatch_mut(`t`, `new_args`, `val`)

# In-place operators support via code transformation macro
# -----------------------------------------------------------------

proc transformInplaceOps(n: NimNode): NimNode =
  ## Recursively transform in-place operations on indexed expressions
  ## a[i, j] += v becomes a[i, j] = a[i, j] + v
  result = n.copy()

  case n.kind
  of nnkInfix:
    if n.len >= 3 and n[0].kind == nnkIdent:
      let op = $n[0]
      if op in ["+=", "-=", "*=", "/=", "div=", "mod="] and n[1].kind == nnkBracketExpr:
        # Transform a[i] += v to a[i] = a[i] + v
        let lhs = n[1]
        let rhs = transformInplaceOps(n[2])
        let binOp =
          case op
          of "+=":
            ident"+"
          of "-=":
            ident"-"
          of "*=":
            ident"*"
          of "/=":
            ident"/"
          of "div=":
            ident"div"
          of "mod=":
            ident"mod"
          else:
            error("Unsupported operator: " & op)
            ident"+" # Unreachable, but needed for type checking

        let rhsExpr = nnkInfix.newTree(binOp, lhs.copy(), rhs)
        result = nnkAsgn.newTree(lhs, rhsExpr)
        return result
    # Transform children
    for i in 0 ..< result.len:
      result[i] = transformInplaceOps(result[i])
  of nnkAsgn, nnkCall, nnkCommand, nnkCallStrLit:
    for i in 0 ..< result.len:
      result[i] = transformInplaceOps(result[i])
  of nnkStmtList, nnkStmtListExpr:
    for i in 0 ..< result.len:
      result[i] = transformInplaceOps(result[i])
  of nnkIfStmt, nnkIfExpr, nnkWhenStmt, nnkWhileStmt, nnkForStmt:
    for i in 0 ..< result.len:
      result[i] = transformInplaceOps(result[i])
  of nnkElifBranch, nnkElse, nnkOfBranch, nnkElifExpr, nnkElseExpr:
    for i in 0 ..< result.len:
      result[i] = transformInplaceOps(result[i])
  of nnkCaseStmt, nnkTryStmt, nnkBlockStmt:
    for i in 0 ..< result.len:
      result[i] = transformInplaceOps(result[i])
  else:
    discard

macro indexedMutate*(body: untyped): untyped =
  ## Macro to enable in-place operators on indexed tensor expressions
  ##
  ## Usage:
  ##   indexedMutate:
  ##     a[1, 1] += 10
  ##     a[2, 2] *= 5
  ##
  ## This transforms to:
  ##     a[1, 1] = a[1, 1] + 10
  ##     a[2, 2] = a[2, 2] * 5
  result = transformInplaceOps(body)
