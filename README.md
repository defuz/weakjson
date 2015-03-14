
- Allow trailing characters in stream
- Object keys can be unquoted if they're valid identifiers.
  Yes, even reserved keywords (like default) are valid unquoted keys in ES5 [§11.1.5, §7.6]. (More info)
  (TODO: Unicode characters and escape sequences aren’t yet supported in this implementation.)
- Objects can have trailing commas.
- Objects can have numeric keys.
- Arrays can have trailing commas.
- Strings can be single-quoted.
- Strings can be split across multiple lines; just prefix each newline with a backslash. [ES5 §7.8.4]
- Numbers can be hexadecimal (base 16).
- Numbers can begin or end with a (leading or trailing) decimal point.
- Numbers can include Infinity, -Infinity, NaN, and -NaN.
- Numbers can begin with an explicit plus sign.
- Both inline (single-line) and block (multi-line) comments are allowed.
