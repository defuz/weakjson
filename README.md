### weakjson

[![Build Status](https://travis-ci.org/defuz/weakjson.svg?branch=master)](https://travis-ci.org/defuz/weakjson)

The library is still under development. Please don't use it.

##### Comments:

Both inline (single-line) and block (multi-line) comments are allowed:

```
{
    // this is an inline comment
    foo: 'bar', // inline comment

    /* this is a block comment
       that continues on another line */
}
```

##### Trailing commas:

Objects and arrays can have trailing commas:

```
{
    oh: [
        "we shouldn't forget",
        'arrays can have',
        'trailing commas too',
    ],
    finally: 'a trailing comma',
}
```


##### Object keys:

Object keys can be unquoted if they're valid identifiers or it's can be natural numbers:

```
{
    foo: 'bar',
    while: true,
    sparse: {0: "Yankee", 273: "Hotel", 38: "Foxtrot"}
}
```
##### Strings:

Strings can be single-quoted and contain unescaped control characters like '\n' or '\t'.
Therefore, we can split string across multiple lines:

```
{
    "this": "is a \
multi-line string",

    "here": 'is another
multi-line string too',

    'and': 'Say "Hello single-quoted string!"',
}
```

Weakjson ignore invalid escaping like `\f` so that it will be simple `f`.

##### Numbers:

Numbers can include `Infinity`, `-Infinity`, `NaN`, and `-NaN`,
begin with an explicit plus sign,
begin with leading zero digits,
begin or end with a (leading or trailing) decimal point,
be hexadecimal (base 16).

```
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
