### weakjson

[![Build Status](https://travis-ci.org/defuz/weakjson.svg?branch=master)](https://travis-ci.org/defuz/weakjson)

The library is still under development. Please don't use it.

#### How to use it

There are only three functions that are provided by this library.

Decodes a json value from a string:

```rust
pub fn from_str(s: &str) -> Result<Json, BuilderError>
```

Shortcut function to decode a JSON `&str` into an object:
```rust
pub fn decode<T: Decodable>(s: &str) -> Result<T, DecoderError>
```

Decodes a json value from an `&mut io::Read`:

```rust
pub fn from_reader(rdr: &mut Read) -> Result<Json, BuilderError>
```

#### What is the difference with the standard JSON

Both inline (single-line) and block (multi-line) **comments** are allowed:

```javascript
{
    // this is an inline comment

    "foo": "bar" // another inline comment

    /*
       This is a block comment
       that continues on another line
    */
}
```

**Object keys** can be unquoted if they're valid identifiers or it's can be natural numbers:

```javascript
{
    foo: 'bar',
    while: true,
    sparse: {0: "Yankee", 273: "Hotel", 38: "Foxtrot"}
}
```

Objects and arrays can have **trailing commas**:

```javascript
{
    oh: [
        "we shouldn't forget",
        "arrays can have",
        "trailing commas",
    ],
    finally: "a trailing comma",
}
```

**Strings** can be single-quoted and contain unescaped control characters like linebreaks or tabulation.
Therefore, we can split string across multiple lines:

```javascript
[
    "This is a
multi-line string",

    "Here is another \
multi-line string with ignored backslash",

    'And say "hello" to single-quoted string!'
]
```

Weakjson ignore invalid escaping like `\f` so that it will be simple `f`.

**Numbers** can include `Infinity`, `-Infinity`, `NaN`, and `-NaN`,
can begin with an explicit plus sign,
begin with leading zero digits,
begin or end with a (leading or trailing) decimal point or
be hexadecimal (base 16):

```javascript
[
    Infinity,  // f64::INFINITY
    -Infinity, // f64::NEG_INFINITY
    NaN,       // f64::NAN
    -NaN,      // f64::NAN

    +42,       // 42
    042,       // 42
    .42,       // 0.42
    42.,       // 42.0
    0x2A       // 42
]
```
