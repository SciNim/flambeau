{.push header: "ordered_dict.h".}


# Constructors and methods
proc constructor_OrderedDict<Key, Value>*(key_description: std::string): OrderedDict {.constructor,importcpp: "OrderedDict<Key, Value>(@)".}
  ## Constructs the `OrderedDict` with a short description of the kinds of
  ## keys stored in the `OrderedDict`. This description is used in error
  ## messages thrown by the `OrderedDict`.

proc constructor_OrderedDict<Key, Value>*(other: OrderedDict<Key, Value>): OrderedDict {.constructor,importcpp: "OrderedDict<Key, Value>(@)".}
  ## Copy constructs this `OrderedDict` from `other`.

proc constructor_OrderedDict<Key, Value>*(other: var OrderedDict<Key, Value> &): OrderedDict {.constructor,importcpp: "OrderedDict<Key, Value>(@)".}

proc constructor_OrderedDict<Key, Value>*(initializer_list: std::initializer_list<Item>): OrderedDict {.constructor,importcpp: "OrderedDict<Key, Value>(@)".}
  ## Constructs a new `OrderedDict` and pre-populates it with the given
  ## `Item`s.

proc constructor_Item*(key: Key, value: Value): Item {.constructor,importcpp: "Item(@)".}
  ## Constructs a new item.

proc constructor_OrderedDict<Key, Value>*(key_description: std::string): OrderedDict {.constructor,importcpp: "OrderedDict<Key, Value>(@)".}
  ## Constructs the `OrderedDict` with a short description of the kinds of
  ## keys stored in the `OrderedDict`. This description is used in error
  ## messages thrown by the `OrderedDict`.

proc constructor_OrderedDict<Key, Value>*(other: OrderedDict<Key, Value>): OrderedDict {.constructor,importcpp: "OrderedDict<Key, Value>(@)".}
  ## Copy constructs this `OrderedDict` from `other`.

proc constructor_OrderedDict<Key, Value>*(initializer_list: std::initializer_list<Item>): OrderedDict {.constructor,importcpp: "OrderedDict<Key, Value>(@)".}
  ## Constructs a new `OrderedDict` and pre-populates it with the given
  ## `Item`s.

proc `=`*(this: var OrderedDict, other: OrderedDict<Key, Value>): OrderedDict<Key, Value>  {.importcpp: "`=`".}
  ## Assigns items from `other` to this `OrderedDict`.

proc `=`*(this: var OrderedDict, other: var OrderedDict<Key, Value> &): OrderedDict<Key, Value>  {.importcpp: "`=`".}

proc key_description*(this: OrderedDict): std::string  {.importcpp: "key_description".}
  ## Returns the key description string the `OrderedDict` was constructed
  ## with.

proc front*(this: var OrderedDict): torch::OrderedDict::Item  {.importcpp: "front".}
  ## Returns the very first item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc front*(this: OrderedDict): torch::OrderedDict::Item  {.importcpp: "front".}
  ## Returns the very first item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc back*(this: var OrderedDict): torch::OrderedDict::Item  {.importcpp: "back".}
  ## Returns the very last item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc back*(this: OrderedDict): torch::OrderedDict::Item  {.importcpp: "back".}
  ## Returns the very last item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc `[]`*(this: var OrderedDict, index: cint): torch::OrderedDict::Item  {.importcpp: "`[]`".}
  ## Returns the item at the `index`-th position in the `OrderedDict`.
  ## Throws an exception if the index is out of bounds.

proc `[]`*(this: OrderedDict, index: cint): torch::OrderedDict::Item  {.importcpp: "`[]`".}
  ## Returns the item at the `index`-th position in the `OrderedDict`.
  ## Throws an exception if the index is out of bounds.

proc `[]`*(this: var OrderedDict, key: Key): Value  {.importcpp: "`[]`".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `OrderedDict`. Use `find()` for a non-
  ## throwing way of accessing a value if it is present.

proc `[]`*(this: OrderedDict, key: Key): Value  {.importcpp: "`[]`".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `OrderedDict`. Use `find()` for a non-
  ## throwing way of accessing a value if it is present.

proc find*(this: var OrderedDict, key: Key): Value *  {.importcpp: "find".}
  ## Returns a pointer to the value associated with the given key, or a
  ## `nullptr` if no such key is stored in the `OrderedDict`.

proc find*(this: OrderedDict, key: Key): Value *  {.importcpp: "find".}
  ## Returns a pointer to the value associated with the given key, or a
  ## `nullptr` if no such key is stored in the `OrderedDict`.

proc contains*(this: OrderedDict, key: Key): bool  {.importcpp: "contains".}
  ## Returns true if the key is present in the `OrderedDict`.

proc begin*(this: var OrderedDict): int  {.importcpp: "begin".}
  ## Returns an iterator to the first item in the `OrderedDict`. Iteration
  ## is ordered.

proc begin*(this: OrderedDict): int  {.importcpp: "begin".}
  ## Returns an iterator to the first item in the `OrderedDict`. Iteration
  ## is ordered.

proc end*(this: var OrderedDict): int  {.importcpp: "end".}
  ## Returns an iterator one past the last item in the `OrderedDict`.

proc end*(this: OrderedDict): int  {.importcpp: "end".}
  ## Returns an iterator one past the last item in the `OrderedDict`.

proc size*(this: OrderedDict): int  {.importcpp: "size".}
  ## Returns the number of items currently stored in the `OrderedDict`.

proc is_empty*(this: OrderedDict): bool  {.importcpp: "is_empty".}
  ## Returns true if the `OrderedDict` contains no elements.

proc reserve*(this: var OrderedDict, requested_capacity: cint)  {.importcpp: "reserve".}
  ## Resizes internal storage to fit at least `requested_capacity` items
  ## without requiring reallocation.

proc insert*(this: var OrderedDict, key: Key, value: var Value &): Value  {.importcpp: "insert".}
  ## Inserts a new `(key, value)` pair into the `OrderedDict`. Throws an
  ## exception if the key is already present. If insertion is successful,
  ## immediately returns a reference to the inserted value.

proc update*(this: var OrderedDict, other: var OrderedDict<Key, Value> &)  {.importcpp: "update".}
  ## Inserts all items from `other` into this `OrderedDict`. If any key
  ## from `other` is already present in this `OrderedDict`, an exception is
  ## thrown.

proc update*(this: var OrderedDict, other: OrderedDict<Key, Value>)  {.importcpp: "update".}
  ## Inserts all items from `other` into this `OrderedDict`. If any key
  ## from `other` is already present in this `OrderedDict`, an exception is
  ## thrown.

proc erase*(this: var OrderedDict, key: Key)  {.importcpp: "erase".}
  ## Removes the item that has `key` from this `OrderedDict` if exists and
  ## if it doesn't an exception is thrown.

proc clear*(this: var OrderedDict)  {.importcpp: "clear".}
  ## Removes all items from this `OrderedDict`.

proc items*(this: OrderedDict): int  {.importcpp: "items".}
  ## Returns the items stored in the `OrderedDict`.

proc keys*(this: OrderedDict): int  {.importcpp: "keys".}
  ## Returns a newly allocated vector and copies all keys from this
  ## `OrderedDict` into the vector.

proc values*(this: OrderedDict): int  {.importcpp: "values".}
  ## Returns a newly allocated vector and copies all values from this
  ## `OrderedDict` into the vector.

proc pairs*(this: OrderedDict): int  {.importcpp: "pairs".}
  ## Returns a newly allocated vector and copies all keys and values from
  ## this `OrderedDict` into a vector of `std::pair<Key, Value>`.

proc `*`*(this: var Item): Value  {.importcpp: "`*`".}
  ## Returns a reference to the value.

proc `*`*(this: Item): Value  {.importcpp: "`*`".}
  ## Returns a reference to the value.

proc `->`*(this: var Item): Value *  {.importcpp: "`->`".}
  ## Allows access to the value using the arrow operator.

proc `->`*(this: Item): Value *  {.importcpp: "`->`".}
  ## Allows access to the value using the arrow operator.

proc key*(this: Item): Key  {.importcpp: "key".}
  ## Returns a reference to the key.

proc value*(this: var Item): Value  {.importcpp: "value".}
  ## Returns a reference to the value.

proc value*(this: Item): Value  {.importcpp: "value".}
  ## Returns a reference to the value.

proc pair*(this: Item): std::pair<Key, Value>  {.importcpp: "pair".}
  ## Returns a `(key, value)` pair.

proc `=`*(this: var OrderedDict, other: OrderedDict<Key, Value>): OrderedDict<Key, Value>  {.importcpp: "`=`".}
  ## Assigns items from `other` to this `OrderedDict`.

proc begin*(this: var OrderedDict): typename OrderedDict<Key, Value>::Iterator  {.importcpp: "begin".}

proc begin*(this: OrderedDict): typename OrderedDict<Key, Value>::ConstIterator  {.importcpp: "begin".}

proc end*(this: var OrderedDict): typename OrderedDict<Key, Value>::Iterator  {.importcpp: "end".}

proc end*(this: OrderedDict): typename OrderedDict<Key, Value>::ConstIterator  {.importcpp: "end".}

proc front*(this: var OrderedDict): typename OrderedDict<Key, Value>::Item  {.importcpp: "front".}
  ## Returns the very first item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc front*(this: OrderedDict): typename OrderedDict<Key, Value>::Item  {.importcpp: "front".}
  ## Returns the very first item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc back*(this: var OrderedDict): typename OrderedDict<Key, Value>::Item  {.importcpp: "back".}
  ## Returns the very last item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc back*(this: OrderedDict): typename OrderedDict<Key, Value>::Item  {.importcpp: "back".}
  ## Returns the very last item in the `OrderedDict` and throws an
  ## exception if it is empty.

proc `[]`*(this: var OrderedDict, index: cint): typename OrderedDict<Key, Value>::Item  {.importcpp: "`[]`".}

proc `[]`*(this: OrderedDict, index: cint): typename OrderedDict<Key, Value>::Item  {.importcpp: "`[]`".}

proc `[]`*(this: var OrderedDict, key: Key): Value  {.importcpp: "`[]`".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `OrderedDict`. Use `find()` for a non-
  ## throwing way of accessing a value if it is present.

proc `[]`*(this: OrderedDict, key: Key): Value  {.importcpp: "`[]`".}
  ## Returns the value associated with the given `key`. Throws an exception
  ## if no such key is stored in the `OrderedDict`. Use `find()` for a non-
  ## throwing way of accessing a value if it is present.

proc insert*(this: var OrderedDict, key: Key, value: var Value &): Value  {.importcpp: "insert".}
  ## Inserts a new `(key, value)` pair into the `OrderedDict`. Throws an
  ## exception if the key is already present. If insertion is successful,
  ## immediately returns a reference to the inserted value.

proc update*(this: var OrderedDict, other: var OrderedDict<Key, Value> &)  {.importcpp: "update".}
  ## Inserts all items from `other` into this `OrderedDict`. If any key
  ## from `other` is already present in this `OrderedDict`, an exception is
  ## thrown.

proc update*(this: var OrderedDict, other: OrderedDict<Key, Value>)  {.importcpp: "update".}
  ## Inserts all items from `other` into this `OrderedDict`. If any key
  ## from `other` is already present in this `OrderedDict`, an exception is
  ## thrown.

proc find*(this: var OrderedDict, key: Key): Value *  {.importcpp: "find".}
  ## Returns a pointer to the value associated with the given key, or a
  ## `nullptr` if no such key is stored in the `OrderedDict`.

proc find*(this: OrderedDict, key: Key): Value *  {.importcpp: "find".}
  ## Returns a pointer to the value associated with the given key, or a
  ## `nullptr` if no such key is stored in the `OrderedDict`.

proc erase*(this: var OrderedDict, key: Key)  {.importcpp: "erase".}
  ## Removes the item that has `key` from this `OrderedDict` if exists and
  ## if it doesn't an exception is thrown.

proc contains*(this: OrderedDict, key: Key): bool  {.importcpp: "contains".}
  ## Returns true if the key is present in the `OrderedDict`.

proc clear*(this: var OrderedDict)  {.importcpp: "clear".}
  ## Removes all items from this `OrderedDict`.

proc size*(this: OrderedDict): int  {.importcpp: "size".}

proc is_empty*(this: OrderedDict): bool  {.importcpp: "is_empty".}
  ## Returns true if the `OrderedDict` contains no elements.

proc key_description*(this: OrderedDict): std::string  {.importcpp: "key_description".}
  ## Returns the key description string the `OrderedDict` was constructed
  ## with.

proc items*(this: OrderedDict): int  {.importcpp: "items".}

proc keys*(this: OrderedDict): int  {.importcpp: "keys".}

proc values*(this: OrderedDict): int  {.importcpp: "values".}

proc pairs*(this: OrderedDict): int  {.importcpp: "pairs".}

proc reserve*(this: var OrderedDict, requested_capacity: cint)  {.importcpp: "reserve".}

{.pop.} # header: "ordered_dict.h
