import macros, sequtils
import fusion/matching
{.experimental: "caseStmtMacros".}

type
  NetTypes = enum
    netBuiltin
    netCustom
    netParam

proc changeTypeName(n: NimNode, newName: string): NimNode =
  ## Returns a copy of a typeNameSection with changed name
  result = n.copy()
  if result.kind == nnkIdent:
    result = ident(newName)
  elif result.kind == nnkPostfix:
    result[1] = ident(newName)
  elif result.kind == nnkPragmaExpr:
    result[0] = changeTypeName(result[0], newName)
  else:
    raise newException(ValueError, "Node kind " & $result.kind & " didn't match any of {Ident, PostFix, PragmaExpr}")

proc genCppType(typeName: string, builtinNets, customNets: seq[(string, string)], params: seq[string]): string =
  result = "struct " & typeName & "Impl" & ": public torch::nn::Module {" # struct NetImpl: public torch::nn::Module {
  for field in builtinNets:
    let fieldName = field[0]
    let fieldType = field[1]
    result &= "\n\ttorch::nn::" & fieldType & " " & fieldName & "{nullptr};" # torch::nn::Linear fc1{nullptr};
  for field in customNets:
    let fieldName = field[0]
    let fieldType = field[1]
    result &= "\n\t" & "" & fieldType & " " & fieldName & "{nullptr};" # Net net{nullptr};
  for field in params:
    result &= "\n\t" & "torch::Tensor " & field & ";" # torch::Tensor t;
  result &= "\n};"
  result &= "\ntypedef std::shared_ptr<" & typeName & "Impl" & "> " & typeName & ";" # typedef std::shared_ptr<NetImpl> Net;
#std::shared_ptr<>

proc genInitProc(typeName: string, initParams: seq[(string, string, seq[NimNode], NetTypes)]): NimNode =
  ## (fieldName, fieldType, params, custom?)
  var body = newStmtList()
  let net = ident"net"
  for (fieldName, fieldType, params, netType) in initParams:
    let fieldNameIdent = ident(fieldName)
    let fieldTypeIdent = ident(fieldType)
    case netType:
    of netBuiltin:
      let initCall = newCall("init", @[fieldTypeIdent].concat(params))
      let moduleName = fieldName & "_module"
      body.add quote do:
        result.`fieldNameIdent` = result.register_module(`moduleName`, `initCall`)
    of netCustom:
      let moduleName = fieldName & "_module"
      let initCall = newCall("init", fieldTypeIdent)
      body.add quote do:
        result.`fieldNameIdent` = result.register_module(`moduleName`, `initCall`)
    of netParam:
      let moduleName = fieldName & "_param"
      let params = params[0]
      body.add quote do:
        result.`fieldNameIdent` = result.register_parameter(`moduleName`, `params`)
  let typeIdent = ident(typeName)
  let typeImplIdent = ident(typeName & "Impl")
  result = quote do:
    proc init*(T: type `typeIdent`): `typeIdent` =
      result = make_shared(`typeImplIdent`)
      `body`


macro defModule*(s: untyped): untyped =
  ## Nets that are nested must be defined AFTER their parent.
  # Could be solved by forward declaring all `init` procs.
  ## String in register_module is `fieldName & "_module"` to allow the field name to be used on a reassignment.
  s.expectKind(nnkStmtList)
  var typeSections = newStmtList()
  var cppTypes: string
  var initProcs = newStmtList()
  for stm in s:
    if stm.kind == nnkTypeSection:
      var newTypeSection = newNimNode(nnkTypeSection)
      for tDef in stm:
        if (tDef.matches do: # match type section that inherits from Module
          TypeDef:
            (@typeNameSection is (Ident(strVal: @typeName))) |
            (@typeNameSection is (Postfix[Ident(strVal: "*"), Ident(strVal: @typeName)])) |
            PragmaExpr[
              @typeNameSection is (Ident(strVal: @typeName) | Postfix[Ident(strVal: "*"), Ident(strVal: @typeName)]),
              Pragma[all @pragmas]
            ]
            _ 
            ObjectTy:
              _
              OfInherit:
                Ident(strVal: "Module")
              @typeFields
        ):
          echo "Jooooohooooooooo!!!!"
          var newFields = nnkRecList.newTree
          var builtinNets: seq[(string, string)]
          var customNets: seq[(string, string)] # (fieldName, typeName)
          var paramsList: seq[string]
          var initParams: seq[(string, string, seq[NimNode], NetTypes)] # (fieldName, fieldType, params, custom) 
          for f in typeFields:
            let fieldNameSection = f[0]
            fieldNameSection.assertMatch(Ident(strVal: @fieldName) | Postfix[Ident(strVal: "*"), Ident(strVal: @fieldName)])
            if f[2].matches(Command[Ident(strVal: "custom"), @a] | Call[Ident(strVal: "custom"), @a]): # custom Linear or custom(Linear)
              if a.kind == nnkIdent: # fc1 = custom Linear
                let typeName = a.strVal 
                customNets.add (fieldName, typeName)
                initParams.add (fieldName, typeName, @[], netCustom)
                newFields.add nnkIdentDefs.newTree(fieldNameSection, ident(typeName), newEmptyNode())
              else: # f1c = custom Linear(1, 2, 3)
                a.assertMatch(Call[Ident(strVal: @typeName), all @params])
                customNets.add (fieldName, typeName)
                initParams.add (fieldName, typeName, params, netCustom)
                newFields.add nnkIdentDefs.newTree(fieldNameSection, ident(typeName), newEmptyNode())
            elif f[2].matches(Command[Ident(strVal: "param"), @a] | Call[Ident(strVal: "param"), @a]): # param randn(10, 10) or param(randn(10, 10))
              #a.assertMatch(Call[Ident(strVal: @procName), all @params])
              paramsList.add fieldName
              initParams.add (fieldName, "Hello there! - Obi Wan Kenobi", @[a], netParam)
              newFields.add nnkIdentDefs.newTree(fieldNameSection, ident"Tensor", newEmptyNode())
            else: # not custom net
              let callStmt = f[2]
              if callStmt.kind == nnkIdent: # fc1 = Linear
                let typeName = callStmt.strVal 
                builtinNets.add (fieldName, typeName)
                initParams.add (fieldName, typeName, @[], netBuiltin)
                newFields.add nnkIdentDefs.newTree(fieldNameSection, ident(typeName), newEmptyNode())
              else:
                callStmt.assertMatch(Call[Ident(strVal: @typeName), all @params])
                builtinNets.add (fieldName, typeName)
                initParams.add (fieldName, typeName, params, netBuiltin)
                newFields.add nnkIdentDefs.newTree(fieldNameSection, ident(typeName), newEmptyNode())
          let cppString = genCppType(typeName, builtinNets, customNets, paramsList)
          cppTypes &= "\n" & cppString
          var newTypeDef = tDef.copy
          var pragmaList = concat(pragmas, @[ident"pure", ident"importcpp"])
          var newPragma = newTree(nnkPragmaExpr, typeNameSection.changeTypeName(typeName & "Impl"), nnkPragma.newTree(pragmaList))
          newTypeDef[0] = newPragma
          newTypeDef[2][2] = newFields          
          newTypeSection.add newTypeDef
          
          # add typeName = CppSharedPtr[typeNameImpl]
          let pragma2 = newTree(nnkPragmaExpr, typeNameSection, nnkPragma.newTree(pragmas))
          var newSharedType = nnkTypeDef.newTree(
            pragma2,
            newEmptyNode(),
            nnkBracketExpr.newTree(
              ident"CppSharedPtr",
              ident(typeName & "Impl")
            )
          )
          newTypeSection.add newSharedType
          initProcs.add genInitProc(typeName, initParams)
        else:
          echo "Noooooooooooo"
          newTypeSection.add tDef
      typeSections.add newTypeSection
    else:
      typeSections.add stm
  var emitSection = quote do:
    emitTypes:
      `cppTypes`
  result = newStmtList()
  result.add emitSection
  result.add typeSections
  result.add initProcs
  echo result.repr
