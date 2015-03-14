#![feature(unicode, io, old_io)]
#![cfg_attr(test, feature(test))]

extern crate "rustc-serialize" as rustc_serialize;
extern crate unicode;

use rustc_serialize::json::Json;
use rustc_serialize::json::JsonEvent;
use rustc_serialize::Decodable;
use rustc_serialize::json::Decoder;
use rustc_serialize::json::DecoderError;
use rustc_serialize::json::ErrorCode;
use rustc_serialize::json::ParserError;
use rustc_serialize::json::BuilderError;

use rustc_serialize::json::JsonEvent::*;
use rustc_serialize::json::ErrorCode::*;
use rustc_serialize::json::ParserError::*;
use rustc_serialize::json::DecoderError::*;

use ParserState::*;
use InternalStackElement::*;

use std::collections::{BTreeMap};
use std::mem::swap;
use std::num::{Float, Int};
use std::string;
use std::{char, io, str};

use unicode::str as unicode_str;
use unicode::str::Utf16Item;

/// Shortcut function to decode a JSON `&str` into an object
pub fn decode_non_strict<T: Decodable>(s: &str) -> Result<T, DecoderError> {
    let json = match json_from_str_non_strict(s) {
        Ok(x) => x,
        Err(e) => return Err(ParseError(e))
    };

    let mut decoder = Decoder::new(json);
    Decodable::decode(&mut decoder)
}

fn io_error_to_error(err: io::Error) -> ParserError {
    // fixme: remove old_io
    use std::old_io::IoErrorKind;
    IoError(IoErrorKind::EndOfFile, "")
}

/// Decodes a json value from an `&mut io::Read`
pub fn json_from_reader_non_strict(rdr: &mut io::Read) -> Result<Json, BuilderError> {
    let contents = {
        let mut c = Vec::new();
        match rdr.read_to_end(&mut c) {
            Ok(_)  => (),
            Err(e) => return Err(io_error_to_error(e))
        }
        c
    };
    let s = match str::from_utf8(&contents).ok() {
        Some(s) => s,
        _       => return Err(SyntaxError(NotUtf8, 0, 0))
    };
    let mut builder = Builder::new(s.chars());
    builder.build()
}

/// Decodes a json value from a string
pub fn json_from_str_non_strict(s: &str) -> Result<Json, BuilderError> {
    let mut builder = Builder::new(s.chars());
    builder.build()
}

#[derive(PartialEq, Debug)]
enum ParserState {
    // Parse a value in an array, true means first element.
    ParseArray(bool),
    // Parse ',' or ']' after an element in an array.
    ParseArrayComma,
    // Parse a key:value in an object, true means first element.
    ParseObject(bool),
    // Parse ',' or ']' after an element in an object.
    ParseObjectComma,
    // Initial state.
    ParseStart,
    // Expecting the stream to end.
    ParseBeforeFinish,
    // Parsing can't continue.
    ParseFinished,
}

/// A Stack represents the current position of the parser in the logical
/// structure of the JSON stream.
/// For example foo.bar[3].x
pub struct Stack {
    stack: Vec<InternalStackElement>,
    str_buffer: Vec<u8>,
}

/// StackElements compose a Stack.
/// For example, Key("foo"), Key("bar"), Index(3) and Key("x") are the
/// StackElements compositing the stack that represents foo.bar[3].x
#[derive(PartialEq, Clone, Debug)]
pub enum StackElement<'l> {
    Index(u32),
    Key(&'l str),
}

// Internally, Key elements are stored as indices in a buffer to avoid
// allocating a string for every member of an object.
#[derive(PartialEq, Clone, Debug)]
enum InternalStackElement {
    InternalIndex(u32),
    InternalKey(u16, u16), // start, size
}

impl Stack {
    pub fn new() -> Stack {
        Stack { stack: Vec::new(), str_buffer: Vec::new() }
    }

    /// Returns The number of elements in the Stack.
    pub fn len(&self) -> usize { self.stack.len() }

    /// Returns true if the stack is empty.
    pub fn is_empty(&self) -> bool { self.stack.is_empty() }

    /// Provides access to the StackElement at a given index.
    /// lower indices are at the bottom of the stack while higher indices are
    /// at the top.
    pub fn get<'l>(&'l self, idx: usize) -> StackElement<'l> {
        match self.stack[idx] {
            InternalIndex(i) => StackElement::Index(i),
            InternalKey(start, size) => {
                StackElement::Key(str::from_utf8(
                    &self.str_buffer[start as usize .. start as usize + size as usize]).unwrap())
            }
        }
    }

    /// Compares this stack with an array of StackElements.
    pub fn is_equal_to(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() != rhs.len() { return false; }
        for i in 0..rhs.len() {
            if self.get(i) != rhs[i] { return false; }
        }
        return true;
    }

    /// Returns true if the bottom-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn starts_with(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() < rhs.len() { return false; }
        for i in 0..rhs.len() {
            if self.get(i) != rhs[i] { return false; }
        }
        return true;
    }

    /// Returns true if the top-most elements of this stack are the same as
    /// the ones passed as parameter.
    pub fn ends_with(&self, rhs: &[StackElement]) -> bool {
        if self.stack.len() < rhs.len() { return false; }
        let offset = self.stack.len() - rhs.len();
        for i in 0..rhs.len() {
            if self.get(i + offset) != rhs[i] { return false; }
        }
        return true;
    }

    /// Returns the top-most element (if any).
    pub fn top<'l>(&'l self) -> Option<StackElement<'l>> {
        return match self.stack.last() {
            None => None,
            Some(&InternalIndex(i)) => Some(StackElement::Index(i)),
            Some(&InternalKey(start, size)) => {
                Some(StackElement::Key(str::from_utf8(
                    &self.str_buffer[start as usize .. (start+size) as usize]
                ).unwrap()))
            }
        }
    }

    // Used by Parser to insert Key elements at the top of the stack.
    fn push_key(&mut self, key: string::String) {
        self.stack.push(InternalKey(self.str_buffer.len() as u16, key.len() as u16));
        for c in key.as_bytes().iter() {
            self.str_buffer.push(*c);
        }
    }

    // Used by Parser to insert Index elements at the top of the stack.
    fn push_index(&mut self, index: u32) {
        self.stack.push(InternalIndex(index));
    }

    // Used by Parser to remove the top-most element of the stack.
    fn pop(&mut self) {
        assert!(!self.is_empty());
        match *self.stack.last().unwrap() {
            InternalKey(_, sz) => {
                let new_size = self.str_buffer.len() - sz as usize;
                self.str_buffer.truncate(new_size);
            }
            InternalIndex(_) => {}
        }
        self.stack.pop();
    }

    // Used by Parser to test whether the top-most element is an index.
    fn last_is_index(&self) -> bool {
        if self.is_empty() { return false; }
        return match *self.stack.last().unwrap() {
            InternalIndex(_) => true,
            _ => false,
        }
    }

    // Used by Parser to increment the index of the top-most element.
    fn bump_index(&mut self) {
        let len = self.stack.len();
        let idx = match *self.stack.last().unwrap() {
            InternalIndex(i) => { i + 1 }
            _ => { panic!(); }
        };
        self.stack[len - 1] = InternalIndex(idx);
    }
}

/// A streaming JSON parser implemented as an iterator of JsonEvent, consuming
/// an iterator of char.
pub struct Parser<T> {
    rdr: T,
    ch: Option<char>,
    line: usize,
    col: usize,
    // We maintain a stack representing where we are in the logical structure
    // of the JSON stream.
    stack: Stack,
    // A state machine is kept to make it possible to interrupt and resume parsing.
    state: ParserState,
}

impl<T: Iterator<Item = char>> Iterator for Parser<T> {
    type Item = JsonEvent;

    fn next(&mut self) -> Option<JsonEvent> {
        if self.state == ParseFinished {
            return None;
        }

        if self.state == ParseBeforeFinish {
            self.parse_whitespace();
            // Make sure there is no trailing characters.
            if self.eof() {
                self.state = ParseFinished;
                return None;
            } else {
                return Some(self.error_event(TrailingCharacters));
            }
        }

        return Some(self.parse());
    }
}

impl<T: Iterator<Item = char>> Parser<T> {
    /// Creates the JSON parser.
    pub fn new(rdr: T) -> Parser<T> {
        let mut p = Parser {
            rdr: rdr,
            ch: Some('\x00'),
            line: 1,
            col: 0,
            stack: Stack::new(),
            state: ParseStart,
        };
        p.bump();
        return p;
    }

    /// Provides access to the current position in the logical structure of the
    /// JSON stream.
    pub fn stack<'l>(&'l self) -> &'l Stack {
        return &self.stack;
    }

    fn eof(&self) -> bool { self.ch.is_none() }
    fn ch_or_null(&self) -> char { self.ch.unwrap_or('\x00') }
    fn bump(&mut self) {
        self.ch = self.rdr.next();

        if self.ch_is('\n') {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
    }

    fn next_char(&mut self) -> Option<char> {
        self.bump();
        self.ch
    }
    fn ch_is(&self, c: char) -> bool {
        self.ch == Some(c)
    }

    fn error<E>(&self, reason: ErrorCode) -> Result<E, ParserError> {
        Err(SyntaxError(reason, self.line, self.col))
    }

    fn parse_whitespace(&mut self) {
        while self.ch_is(' ') ||
              self.ch_is('\n') ||
              self.ch_is('\t') ||
              self.ch_is('\r') { self.bump(); }
    }

    fn parse_number(&mut self) -> JsonEvent {
        let mut neg = false;

        if self.ch_is('-') {
            self.bump();
            neg = true;
        }

        let res = match self.parse_u64() {
            Ok(res) => res,
            Err(e) => { return Error(e); }
        };

        if self.ch_is('.') || self.ch_is('e') || self.ch_is('E') {
            let mut res = res as f64;

            if self.ch_is('.') {
                res = match self.parse_decimal(res) {
                    Ok(res) => res,
                    Err(e) => { return Error(e); }
                };
            }

            if self.ch_is('e') || self.ch_is('E') {
                res = match self.parse_exponent(res) {
                    Ok(res) => res,
                    Err(e) => { return Error(e); }
                };
            }

            if neg {
                res *= -1.0;
            }

            F64Value(res)
        } else {
            if neg {
                let res = -(res as i64);

                // Make sure we didn't underflow.
                if res > 0 {
                    Error(SyntaxError(InvalidNumber, self.line, self.col))
                } else {
                    I64Value(res)
                }
            } else {
                U64Value(res)
            }
        }
    }

    fn parse_u64(&mut self) -> Result<u64, ParserError> {
        let mut accum = 0;

        match self.ch_or_null() {
            '0' => {
                self.bump();

                // A leading '0' must be the only digit before the decimal point.
                match self.ch_or_null() {
                    '0' ... '9' => return self.error(InvalidNumber),
                    _ => ()
                }
            },
            '1' ... '9' => {
                while !self.eof() {
                    match self.ch_or_null() {
                        c @ '0' ... '9' => {
                            macro_rules! try_or_invalid {
                                ($e: expr) => {
                                    match $e {
                                        Some(v) => v,
                                        None => return self.error(InvalidNumber)
                                    }
                                }
                            }
                            accum = try_or_invalid!(accum.checked_mul(10));
                            accum = try_or_invalid!(accum.checked_add((c as u64) - ('0' as u64)));

                            self.bump();
                        }
                        _ => break,
                    }
                }
            }
            _ => return self.error(InvalidNumber),
        }

        Ok(accum)
    }

    fn parse_decimal(&mut self, mut res: f64) -> Result<f64, ParserError> {
        self.bump();

        // Make sure a digit follows the decimal place.
        match self.ch_or_null() {
            '0' ... '9' => (),
             _ => return self.error(InvalidNumber)
        }

        let mut dec = 1.0;
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0' ... '9' => {
                    dec /= 10.0;
                    res += (((c as isize) - ('0' as isize)) as f64) * dec;
                    self.bump();
                }
                _ => break,
            }
        }

        Ok(res)
    }

    fn parse_exponent(&mut self, mut res: f64) -> Result<f64, ParserError> {
        self.bump();

        let mut exp = 0;
        let mut neg_exp = false;

        if self.ch_is('+') {
            self.bump();
        } else if self.ch_is('-') {
            self.bump();
            neg_exp = true;
        }

        // Make sure a digit follows the exponent place.
        match self.ch_or_null() {
            '0' ... '9' => (),
            _ => return self.error(InvalidNumber)
        }
        while !self.eof() {
            match self.ch_or_null() {
                c @ '0' ... '9' => {
                    exp *= 10;
                    exp += (c as usize) - ('0' as usize);

                    self.bump();
                }
                _ => break
            }
        }

        let exp = 10_f64.powi(exp as i32);
        if neg_exp {
            res /= exp;
        } else {
            res *= exp;
        }

        Ok(res)
    }

    fn decode_hex_escape(&mut self) -> Result<u16, ParserError> {
        let mut i = 0;
        let mut n = 0;
        while i < 4 && !self.eof() {
            self.bump();
            n = match self.ch_or_null() {
                c @ '0' ... '9' => n * 16 + ((c as u16) - ('0' as u16)),
                'a' | 'A' => n * 16 + 10,
                'b' | 'B' => n * 16 + 11,
                'c' | 'C' => n * 16 + 12,
                'd' | 'D' => n * 16 + 13,
                'e' | 'E' => n * 16 + 14,
                'f' | 'F' => n * 16 + 15,
                _ => return self.error(InvalidEscape)
            };

            i += 1;
        }

        // Error out if we didn't parse 4 digits.
        if i != 4 {
            return self.error(InvalidEscape);
        }

        Ok(n)
    }

    fn parse_str(&mut self) -> Result<string::String, ParserError> {
        let mut escape = false;
        let mut res = string::String::new();

        loop {
            self.bump();
            if self.eof() {
                return self.error(EOFWhileParsingString);
            }

            if escape {
                match self.ch_or_null() {
                    '"' => res.push('"'),
                    '\\' => res.push('\\'),
                    '/' => res.push('/'),
                    'b' => res.push('\x08'),
                    'f' => res.push('\x0c'),
                    'n' => res.push('\n'),
                    'r' => res.push('\r'),
                    't' => res.push('\t'),
                    'u' => match try!(self.decode_hex_escape()) {
                        0xDC00 ... 0xDFFF => {
                            return self.error(LoneLeadingSurrogateInHexEscape)
                        }

                        // Non-BMP characters are encoded as a sequence of
                        // two hex escapes, representing UTF-16 surrogates.
                        n1 @ 0xD800 ... 0xDBFF => {
                            match (self.next_char(), self.next_char()) {
                                (Some('\\'), Some('u')) => (),
                                _ => return self.error(UnexpectedEndOfHexEscape),
                            }

                            let buf = [n1, try!(self.decode_hex_escape())];
                            match unicode_str::utf16_items(&buf).next() {
                                Some(Utf16Item::ScalarValue(c)) => res.push(c),
                                _ => return self.error(LoneLeadingSurrogateInHexEscape),
                            }
                        }

                        n => match char::from_u32(n as u32) {
                            Some(c) => res.push(c),
                            None => return self.error(InvalidUnicodeCodePoint),
                        },
                    },
                    _ => return self.error(InvalidEscape),
                }
                escape = false;
            } else if self.ch_is('\\') {
                escape = true;
            } else {
                match self.ch {
                    Some('"') => {
                        self.bump();
                        return Ok(res);
                    },
                    Some(c) => res.push(c),
                    None => unreachable!()
                }
            }
        }
    }

    // Invoked at each iteration, consumes the stream until it has enough
    // information to return a JsonEvent.
    // Manages an internal state so that parsing can be interrupted and resumed.
    // Also keeps track of the position in the logical structure of the json
    // stream int the form of a stack that can be queried by the user using the
    // stack() method.
    fn parse(&mut self) -> JsonEvent {
        loop {
            // The only paths where the loop can spin a new iteration
            // are in the cases ParseArrayComma and ParseObjectComma if ','
            // is parsed. In these cases the state is set to (respectively)
            // ParseArray(false) and ParseObject(false), which always return,
            // so there is no risk of getting stuck in an infinite loop.
            // All other paths return before the end of the loop's iteration.
            self.parse_whitespace();

            match self.state {
                ParseStart => {
                    return self.parse_start();
                }
                ParseArray(first) => {
                    return self.parse_array(first);
                }
                ParseArrayComma => {
                    match self.parse_array_comma_or_end() {
                        Some(evt) => { return evt; }
                        None => {}
                    }
                }
                ParseObject(first) => {
                    return self.parse_object(first);
                }
                ParseObjectComma => {
                    self.stack.pop();
                    if self.ch_is(',') {
                        self.state = ParseObject(false);
                        self.bump();
                    } else {
                        return self.parse_object_end();
                    }
                }
                _ => {
                    return self.error_event(InvalidSyntax);
                }
            }
        }
    }

    fn parse_start(&mut self) -> JsonEvent {
        let val = self.parse_value();
        self.state = match val {
            Error(_) => ParseFinished,
            ArrayStart => ParseArray(true),
            ObjectStart => ParseObject(true),
            _ => ParseBeforeFinish,
        };
        return val;
    }

    fn parse_array(&mut self, first: bool) -> JsonEvent {
        if self.ch_is(']') {
            if !first {
                self.error_event(InvalidSyntax)
            } else {
                self.state = if self.stack.is_empty() {
                    ParseBeforeFinish
                } else if self.stack.last_is_index() {
                    ParseArrayComma
                } else {
                    ParseObjectComma
                };
                self.bump();
                ArrayEnd
            }
        } else {
            if first {
                self.stack.push_index(0);
            }
            let val = self.parse_value();
            self.state = match val {
                Error(_) => ParseFinished,
                ArrayStart => ParseArray(true),
                ObjectStart => ParseObject(true),
                _ => ParseArrayComma,
            };
            val
        }
    }

    fn parse_array_comma_or_end(&mut self) -> Option<JsonEvent> {
        if self.ch_is(',') {
            self.stack.bump_index();
            self.state = ParseArray(false);
            self.bump();
            None
        } else if self.ch_is(']') {
            self.stack.pop();
            self.state = if self.stack.is_empty() {
                ParseBeforeFinish
            } else if self.stack.last_is_index() {
                ParseArrayComma
            } else {
                ParseObjectComma
            };
            self.bump();
            Some(ArrayEnd)
        } else if self.eof() {
            Some(self.error_event(EOFWhileParsingArray))
        } else {
            Some(self.error_event(InvalidSyntax))
        }
    }

    fn parse_object(&mut self, first: bool) -> JsonEvent {
        if self.ch_is('}') {
            if !first {
                if self.stack.is_empty() {
                    return self.error_event(TrailingComma);
                } else {
                    self.stack.pop();
                }
            }
            self.state = if self.stack.is_empty() {
                ParseBeforeFinish
            } else if self.stack.last_is_index() {
                ParseArrayComma
            } else {
                ParseObjectComma
            };
            self.bump();
            return ObjectEnd;
        }
        if self.eof() {
            return self.error_event(EOFWhileParsingObject);
        }
        if !self.ch_is('"') {
            return self.error_event(KeyMustBeAString);
        }
        let s = match self.parse_str() {
            Ok(s) => s,
            Err(e) => {
                self.state = ParseFinished;
                return Error(e);
            }
        };
        self.parse_whitespace();
        if self.eof() {
            return self.error_event(EOFWhileParsingObject);
        } else if self.ch_or_null() != ':' {
            return self.error_event(ExpectedColon);
        }
        self.stack.push_key(s);
        self.bump();
        self.parse_whitespace();

        let val = self.parse_value();

        self.state = match val {
            Error(_) => ParseFinished,
            ArrayStart => ParseArray(true),
            ObjectStart => ParseObject(true),
            _ => ParseObjectComma,
        };
        return val;
    }

    fn parse_object_end(&mut self) -> JsonEvent {
        if self.ch_is('}') {
            self.state = if self.stack.is_empty() {
                ParseBeforeFinish
            } else if self.stack.last_is_index() {
                ParseArrayComma
            } else {
                ParseObjectComma
            };
            self.bump();
            ObjectEnd
        } else if self.eof() {
            self.error_event(EOFWhileParsingObject)
        } else {
            self.error_event(InvalidSyntax)
        }
    }

    fn parse_value(&mut self) -> JsonEvent {
        if self.eof() { return self.error_event(EOFWhileParsingValue); }
        match self.ch_or_null() {
            'n' => { self.parse_ident("ull", NullValue) }
            't' => { self.parse_ident("rue", BooleanValue(true)) }
            'f' => { self.parse_ident("alse", BooleanValue(false)) }
            '0' ... '9' | '-' => self.parse_number(),
            '"' => match self.parse_str() {
                Ok(s) => StringValue(s),
                Err(e) => Error(e),
            },
            '[' => {
                self.bump();
                ArrayStart
            }
            '{' => {
                self.bump();
                ObjectStart
            }
            _ => { self.error_event(InvalidSyntax) }
        }
    }

    fn parse_ident(&mut self, ident: &str, value: JsonEvent) -> JsonEvent {
        if ident.chars().all(|c| Some(c) == self.next_char()) {
            self.bump();
            value
        } else {
            Error(SyntaxError(InvalidSyntax, self.line, self.col))
        }
    }

    fn error_event(&mut self, reason: ErrorCode) -> JsonEvent {
        self.state = ParseFinished;
        Error(SyntaxError(reason, self.line, self.col))
    }
}

/// A Builder consumes a json::Parser to create a generic Json structure.
pub struct Builder<T> {
    parser: Parser<T>,
    token: Option<JsonEvent>,
}

impl<T: Iterator<Item = char>> Builder<T> {
    /// Create a JSON Builder.
    pub fn new(src: T) -> Builder<T> {
        Builder { parser: Parser::new(src), token: None, }
    }

    // Decode a Json value from a Parser.
    pub fn build(&mut self) -> Result<Json, BuilderError> {
        self.bump();
        let result = self.build_value();
        self.bump();
        match self.token {
            None => {}
            Some(Error(ref e)) => { return Err(e.clone()); }
            ref tok => { panic!("unexpected token {:?}", tok.clone()); }
        }
        result
    }

    fn bump(&mut self) {
        self.token = self.parser.next();
    }

    fn build_value(&mut self) -> Result<Json, BuilderError> {
        return match self.token {
            Some(NullValue) => Ok(Json::Null),
            Some(I64Value(n)) => Ok(Json::I64(n)),
            Some(U64Value(n)) => Ok(Json::U64(n)),
            Some(F64Value(n)) => Ok(Json::F64(n)),
            Some(BooleanValue(b)) => Ok(Json::Boolean(b)),
            Some(StringValue(ref mut s)) => {
                let mut temp = string::String::new();
                swap(s, &mut temp);
                Ok(Json::String(temp))
            }
            Some(Error(ref e)) => Err(e.clone()),
            Some(ArrayStart) => self.build_array(),
            Some(ObjectStart) => self.build_object(),
            Some(ObjectEnd) => self.parser.error(InvalidSyntax),
            Some(ArrayEnd) => self.parser.error(InvalidSyntax),
            None => self.parser.error(EOFWhileParsingValue),
        }
    }

    fn build_array(&mut self) -> Result<Json, BuilderError> {
        self.bump();
        let mut values = Vec::new();

        loop {
            if self.token == Some(ArrayEnd) {
                return Ok(Json::Array(values.into_iter().collect()));
            }
            match self.build_value() {
                Ok(v) => values.push(v),
                Err(e) => { return Err(e) }
            }
            self.bump();
        }
    }

    fn build_object(&mut self) -> Result<Json, BuilderError> {
        self.bump();

        let mut values = BTreeMap::new();

        loop {
            match self.token {
                Some(ObjectEnd) => { return Ok(Json::Object(values)); }
                Some(Error(ref e)) => { return Err(e.clone()); }
                None => { break; }
                _ => {}
            }
            let key = match self.parser.stack().top() {
                Some(StackElement::Key(k)) => { k.to_string() }
                _ => { panic!("invalid state"); }
            };
            match self.build_value() {
                Ok(value) => { values.insert(key, value); }
                Err(e) => { return Err(e); }
            }
            self.bump();
        }
        return self.parser.error(EOFWhileParsingObject);
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;

    // use rustc_serialize::json::*;
    // use {Decodable};

    use rustc_serialize::json::Json::*;
    use rustc_serialize::json::ErrorCode::*;
    use rustc_serialize::json::ParserError::*;
    use rustc_serialize::json::DecoderError::*;
    use rustc_serialize::json::JsonEvent::*;

    use super::Parser;
    use super::Stack;
    use super::StackElement;
    use super::StackElement::*;
    use rustc_serialize::json::{Json, DecodeResult, JsonEvent};
    use std::{i64, u64};
    use std::collections::BTreeMap;
    use std::string;

    fn mk_object(items: &[(string::String, Json)]) -> Json {
        let mut d = BTreeMap::new();

        for item in items.iter() {
            match *item {
                (ref key, ref value) => { d.insert((*key).clone(), (*value).clone()); },
            }
        };

        Object(d)
    }

    #[test]
    fn test_trailing_characters() {
        assert_eq!(super::json_from_str_non_strict("nulla"),  Err(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(super::json_from_str_non_strict("truea"),  Err(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(super::json_from_str_non_strict("falsea"), Err(SyntaxError(TrailingCharacters, 1, 6)));
        assert_eq!(super::json_from_str_non_strict("1a"),     Err(SyntaxError(TrailingCharacters, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("[]a"),    Err(SyntaxError(TrailingCharacters, 1, 3)));
        assert_eq!(super::json_from_str_non_strict("{}a"),    Err(SyntaxError(TrailingCharacters, 1, 3)));
    }

    #[test]
    fn test_read_identifiers() {
        assert_eq!(super::json_from_str_non_strict("n"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("nul"),  Err(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(super::json_from_str_non_strict("t"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("truz"), Err(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(super::json_from_str_non_strict("f"),    Err(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("faz"),  Err(SyntaxError(InvalidSyntax, 1, 3)));

        assert_eq!(super::json_from_str_non_strict("null"), Ok(Null));
        assert_eq!(super::json_from_str_non_strict("true"), Ok(Boolean(true)));
        assert_eq!(super::json_from_str_non_strict("false"), Ok(Boolean(false)));
        assert_eq!(super::json_from_str_non_strict(" null "), Ok(Null));
        assert_eq!(super::json_from_str_non_strict(" true "), Ok(Boolean(true)));
        assert_eq!(super::json_from_str_non_strict(" false "), Ok(Boolean(false)));
    }

    #[test]
    fn test_decode_identifiers() {
        let v: () = super::decode_non_strict("null").unwrap();
        assert_eq!(v, ());

        let v: bool = super::decode_non_strict("true").unwrap();
        assert_eq!(v, true);

        let v: bool = super::decode_non_strict("false").unwrap();
        assert_eq!(v, false);
    }

    #[test]
    fn test_read_number() {
        assert_eq!(super::json_from_str_non_strict("+"),   Err(SyntaxError(InvalidSyntax, 1, 1)));
        assert_eq!(super::json_from_str_non_strict("."),   Err(SyntaxError(InvalidSyntax, 1, 1)));
        assert_eq!(super::json_from_str_non_strict("NaN"), Err(SyntaxError(InvalidSyntax, 1, 1)));
        assert_eq!(super::json_from_str_non_strict("-"),   Err(SyntaxError(InvalidNumber, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("00"),  Err(SyntaxError(InvalidNumber, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("1."),  Err(SyntaxError(InvalidNumber, 1, 3)));
        assert_eq!(super::json_from_str_non_strict("1e"),  Err(SyntaxError(InvalidNumber, 1, 3)));
        assert_eq!(super::json_from_str_non_strict("1e+"), Err(SyntaxError(InvalidNumber, 1, 4)));

        assert_eq!(super::json_from_str_non_strict("18446744073709551616"), Err(SyntaxError(InvalidNumber, 1, 20)));
        assert_eq!(super::json_from_str_non_strict("18446744073709551617"), Err(SyntaxError(InvalidNumber, 1, 20)));
        assert_eq!(super::json_from_str_non_strict("-9223372036854775809"), Err(SyntaxError(InvalidNumber, 1, 21)));

        assert_eq!(super::json_from_str_non_strict("3"), Ok(U64(3)));
        assert_eq!(super::json_from_str_non_strict("3.1"), Ok(F64(3.1)));
        assert_eq!(super::json_from_str_non_strict("-1.2"), Ok(F64(-1.2)));
        assert_eq!(super::json_from_str_non_strict("0.4"), Ok(F64(0.4)));
        assert_eq!(super::json_from_str_non_strict("0.4e5"), Ok(F64(0.4e5)));
        assert_eq!(super::json_from_str_non_strict("0.4e+15"), Ok(F64(0.4e15)));
        assert_eq!(super::json_from_str_non_strict("0.4e-01"), Ok(F64(0.4e-01)));
        assert_eq!(super::json_from_str_non_strict(" 3 "), Ok(U64(3)));

        assert_eq!(super::json_from_str_non_strict("-9223372036854775808"), Ok(I64(i64::MIN)));
        assert_eq!(super::json_from_str_non_strict("9223372036854775807"), Ok(U64(i64::MAX as u64)));
        assert_eq!(super::json_from_str_non_strict("18446744073709551615"), Ok(U64(u64::MAX)));
    }

    #[test]
    fn test_decode_numbers() {
        let v: f64 = super::decode_non_strict("3").unwrap();
        assert_eq!(v, 3.0);

        let v: f64 = super::decode_non_strict("3.1").unwrap();
        assert_eq!(v, 3.1);

        let v: f64 = super::decode_non_strict("-1.2").unwrap();
        assert_eq!(v, -1.2);

        let v: f64 = super::decode_non_strict("0.4").unwrap();
        assert_eq!(v, 0.4);

        let v: f64 = super::decode_non_strict("0.4e5").unwrap();
        assert_eq!(v, 0.4e5);

        let v: f64 = super::decode_non_strict("0.4e15").unwrap();
        assert_eq!(v, 0.4e15);

        let v: f64 = super::decode_non_strict("0.4e-01").unwrap();
        assert_eq!(v, 0.4e-01);

        let v: u64 = super::decode_non_strict("0").unwrap();
        assert_eq!(v, 0);

        let v: u64 = super::decode_non_strict("18446744073709551615").unwrap();
        assert_eq!(v, u64::MAX);

        let v: i64 = super::decode_non_strict("-9223372036854775808").unwrap();
        assert_eq!(v, i64::MIN);

        let v: i64 = super::decode_non_strict("9223372036854775807").unwrap();
        assert_eq!(v, i64::MAX);

        let res: DecodeResult<i64> = super::decode_non_strict("765.25252");
        assert_eq!(res, Err(ExpectedError("Integer".to_string(), "765.25252".to_string())));
    }

    #[test]
    fn test_read_str() {
        assert_eq!(super::json_from_str_non_strict("\""),    Err(SyntaxError(EOFWhileParsingString, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("\"lol"), Err(SyntaxError(EOFWhileParsingString, 1, 5)));

        assert_eq!(super::json_from_str_non_strict("\"\""), Ok(String("".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"foo\""), Ok(String("foo".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"\\\"\""), Ok(String("\"".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"\\b\""), Ok(String("\x08".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"\\n\""), Ok(String("\n".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"\\r\""), Ok(String("\r".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"\\t\""), Ok(String("\t".to_string())));
        assert_eq!(super::json_from_str_non_strict(" \"foo\" "), Ok(String("foo".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"\\u12ab\""), Ok(String("\u{12ab}".to_string())));
        assert_eq!(super::json_from_str_non_strict("\"\\uAB12\""), Ok(String("\u{AB12}".to_string())));
    }

    #[test]
    fn test_decode_str() {
        let s = [("\"\"", ""),
                 ("\"foo\"", "foo"),
                 ("\"\\\"\"", "\""),
                 ("\"\\b\"", "\x08"),
                 ("\"\\n\"", "\n"),
                 ("\"\\r\"", "\r"),
                 ("\"\\t\"", "\t"),
                 ("\"\\u12ab\"", "\u{12ab}"),
                 ("\"\\uAB12\"", "\u{AB12}")];

        for &(i, o) in s.iter() {
            let v: string::String = super::decode_non_strict(i).unwrap();
            assert_eq!(v, o);
        }
    }

    #[test]
    fn test_read_array() {
        assert_eq!(super::json_from_str_non_strict("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("[1"),    Err(SyntaxError(EOFWhileParsingArray, 1, 3)));
        assert_eq!(super::json_from_str_non_strict("[1,"),   Err(SyntaxError(EOFWhileParsingValue, 1, 4)));
        assert_eq!(super::json_from_str_non_strict("[1,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
        assert_eq!(super::json_from_str_non_strict("[6 7]"), Err(SyntaxError(InvalidSyntax,        1, 4)));

        assert_eq!(super::json_from_str_non_strict("[]"), Ok(Array(vec![])));
        assert_eq!(super::json_from_str_non_strict("[ ]"), Ok(Array(vec![])));
        assert_eq!(super::json_from_str_non_strict("[true]"), Ok(Array(vec![Boolean(true)])));
        assert_eq!(super::json_from_str_non_strict("[ false ]"), Ok(Array(vec![Boolean(false)])));
        assert_eq!(super::json_from_str_non_strict("[null]"), Ok(Array(vec![Null])));
        assert_eq!(super::json_from_str_non_strict("[3, 1]"),
                     Ok(Array(vec![U64(3), U64(1)])));
        assert_eq!(super::json_from_str_non_strict("\n[3, 2]\n"),
                     Ok(Array(vec![U64(3), U64(2)])));
        assert_eq!(super::json_from_str_non_strict("[2, [4, 1]]"),
               Ok(Array(vec![U64(2), Array(vec![U64(4), U64(1)])])));
    }

    #[test]
    fn test_decode_array() {
        let v: Vec<()> = super::decode_non_strict("[]").unwrap();
        assert_eq!(v, vec![]);

        let v: Vec<()> = super::decode_non_strict("[null]").unwrap();
        assert_eq!(v, vec![()]);

        let v: Vec<bool> = super::decode_non_strict("[true]").unwrap();
        assert_eq!(v, vec![true]);

        let v: Vec<isize> = super::decode_non_strict("[3, 1]").unwrap();
        assert_eq!(v, vec![3, 1]);

        let v: Vec<Vec<usize>> = super::decode_non_strict("[[3], [1, 2]]").unwrap();
        assert_eq!(v, vec![vec![3], vec![1, 2]]);
    }

    #[test]
    fn test_decode_tuple() {
        let t: (usize, usize, usize) = super::decode_non_strict("[1, 2, 3]").unwrap();
        assert_eq!(t, (1, 2, 3));

        let t: (usize, string::String) = super::decode_non_strict("[1, \"two\"]").unwrap();
        assert_eq!(t, (1, "two".to_string()));
    }

    #[test]
    fn test_decode_tuple_malformed_types() {
        assert!(super::decode_non_strict::<(usize, string::String)>("[1, 2]").is_err());
    }

    #[test]
    fn test_decode_tuple_malformed_length() {
        assert!(super::decode_non_strict::<(usize, usize)>("[1, 2, 3]").is_err());
    }

    #[test]
    fn test_read_object() {
        assert_eq!(super::json_from_str_non_strict("{"),       Err(SyntaxError(EOFWhileParsingObject, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("{ "),      Err(SyntaxError(EOFWhileParsingObject, 1, 3)));
        assert_eq!(super::json_from_str_non_strict("{1"),      Err(SyntaxError(KeyMustBeAString,      1, 2)));
        assert_eq!(super::json_from_str_non_strict("{ \"a\""), Err(SyntaxError(EOFWhileParsingObject, 1, 6)));
        assert_eq!(super::json_from_str_non_strict("{\"a\""),  Err(SyntaxError(EOFWhileParsingObject, 1, 5)));
        assert_eq!(super::json_from_str_non_strict("{\"a\" "), Err(SyntaxError(EOFWhileParsingObject, 1, 6)));

        assert_eq!(super::json_from_str_non_strict("{\"a\" 1"),   Err(SyntaxError(ExpectedColon,         1, 6)));
        assert_eq!(super::json_from_str_non_strict("{\"a\":"),    Err(SyntaxError(EOFWhileParsingValue,  1, 6)));
        assert_eq!(super::json_from_str_non_strict("{\"a\":1"),   Err(SyntaxError(EOFWhileParsingObject, 1, 7)));
        assert_eq!(super::json_from_str_non_strict("{\"a\":1 1"), Err(SyntaxError(InvalidSyntax,         1, 8)));
        assert_eq!(super::json_from_str_non_strict("{\"a\":1,"),  Err(SyntaxError(EOFWhileParsingObject, 1, 8)));

        assert_eq!(super::json_from_str_non_strict("{}").unwrap(), mk_object(&[]));
        assert_eq!(super::json_from_str_non_strict("{\"a\": 3}").unwrap(),
                  mk_object(&[("a".to_string(), U64(3))]));

        assert_eq!(super::json_from_str_non_strict(
                      "{ \"a\": null, \"b\" : true }").unwrap(),
                  mk_object(&[
                      ("a".to_string(), Null),
                      ("b".to_string(), Boolean(true))]));
        assert_eq!(super::json_from_str_non_strict("\n{ \"a\": null, \"b\" : true }\n").unwrap(),
                  mk_object(&[
                      ("a".to_string(), Null),
                      ("b".to_string(), Boolean(true))]));
        assert_eq!(super::json_from_str_non_strict(
                      "{\"a\" : 1.0 ,\"b\": [ true ]}").unwrap(),
                  mk_object(&[
                      ("a".to_string(), F64(1.0)),
                      ("b".to_string(), Array(vec![Boolean(true)]))
                  ]));
        assert_eq!(super::json_from_str_non_strict(
                      "{\
                          \"a\": 1.0, \
                          \"b\": [\
                              true,\
                              \"foo\\nbar\", \
                              { \"c\": {\"d\": null} } \
                          ]\
                      }").unwrap(),
                  mk_object(&[
                      ("a".to_string(), F64(1.0)),
                      ("b".to_string(), Array(vec![
                          Boolean(true),
                          String("foo\nbar".to_string()),
                          mk_object(&[
                              ("c".to_string(), mk_object(&[("d".to_string(), Null)]))
                          ])
                      ]))
                  ]));
    }

    #[test]
    fn test_duplicate_keys() {
        assert_eq!(super::json_from_str_non_strict("{\"a\": false, \"a\": true}").unwrap(),
                   mk_object(&[("a".to_string(), Boolean(true))]));
    }

    fn assert_stream_equal(src: &str,
                           expected: Vec<(JsonEvent, Vec<StackElement>)>) {
        let mut parser = Parser::new(src.chars());
        let mut i = 0;
        loop {
            let evt = match parser.next() {
                Some(e) => e,
                None => { break; }
            };
            let (ref expected_evt, ref expected_stack) = expected[i];
            if !parser.stack().is_equal_to(&expected_stack) {
                panic!("Parser stack is not equal to {:?}", expected_stack);
            }
            assert_eq!(&evt, expected_evt);
            i+=1;
        }
    }

    #[test]
    #[cfg_attr(target_word_size = "32", ignore)] // FIXME(#14064)
    fn test_streaming_parser() {
        assert_stream_equal(
            r#"{ "foo":"bar", "array" : [0, 1, 2, 3, 4, 5], "idents":[null,true,false]}"#,
            vec![
                (ObjectStart,             vec![]),
                  (StringValue("bar".to_string()),   vec![Key("foo")]),
                  (ArrayStart,            vec![Key("array")]),
                    (U64Value(0),         vec![Key("array"), Index(0)]),
                    (U64Value(1),         vec![Key("array"), Index(1)]),
                    (U64Value(2),         vec![Key("array"), Index(2)]),
                    (U64Value(3),         vec![Key("array"), Index(3)]),
                    (U64Value(4),         vec![Key("array"), Index(4)]),
                    (U64Value(5),         vec![Key("array"), Index(5)]),
                  (ArrayEnd,              vec![Key("array")]),
                  (ArrayStart,            vec![Key("idents")]),
                    (NullValue,           vec![Key("idents"), Index(0)]),
                    (BooleanValue(true),  vec![Key("idents"), Index(1)]),
                    (BooleanValue(false), vec![Key("idents"), Index(2)]),
                  (ArrayEnd,              vec![Key("idents")]),
                (ObjectEnd,               vec![]),
            ]
        );
    }

    fn last_event(src: &str) -> JsonEvent {
        let mut parser = Parser::new(src.chars());
        let mut evt = NullValue;
        loop {
            evt = match parser.next() {
                Some(e) => e,
                None => return evt,
            }
        }
    }

    #[test]
    #[cfg_attr(target_word_size = "32", ignore)] // FIXME(#14064)
    fn test_read_object_streaming() {
        assert_eq!(last_event("{ "),      Error(SyntaxError(EOFWhileParsingObject, 1, 3)));
        assert_eq!(last_event("{1"),      Error(SyntaxError(KeyMustBeAString,      1, 2)));
        assert_eq!(last_event("{ \"a\""), Error(SyntaxError(EOFWhileParsingObject, 1, 6)));
        assert_eq!(last_event("{\"a\""),  Error(SyntaxError(EOFWhileParsingObject, 1, 5)));
        assert_eq!(last_event("{\"a\" "), Error(SyntaxError(EOFWhileParsingObject, 1, 6)));

        assert_eq!(last_event("{\"a\" 1"),   Error(SyntaxError(ExpectedColon,         1, 6)));
        assert_eq!(last_event("{\"a\":"),    Error(SyntaxError(EOFWhileParsingValue,  1, 6)));
        assert_eq!(last_event("{\"a\":1"),   Error(SyntaxError(EOFWhileParsingObject, 1, 7)));
        assert_eq!(last_event("{\"a\":1 1"), Error(SyntaxError(InvalidSyntax,         1, 8)));
        assert_eq!(last_event("{\"a\":1,"),  Error(SyntaxError(EOFWhileParsingObject, 1, 8)));
        assert_eq!(last_event("{\"a\":1,}"), Error(SyntaxError(TrailingComma, 1, 8)));

        assert_stream_equal(
            "{}",
            vec![(ObjectStart, vec![]), (ObjectEnd, vec![])]
        );
        assert_stream_equal(
            "{\"a\": 3}",
            vec![
                (ObjectStart,        vec![]),
                  (U64Value(3),      vec![Key("a")]),
                (ObjectEnd,          vec![]),
            ]
        );
        assert_stream_equal(
            "{ \"a\": null, \"b\" : true }",
            vec![
                (ObjectStart,           vec![]),
                  (NullValue,           vec![Key("a")]),
                  (BooleanValue(true),  vec![Key("b")]),
                (ObjectEnd,             vec![]),
            ]
        );
        assert_stream_equal(
            "{\"a\" : 1.0 ,\"b\": [ true ]}",
            vec![
                (ObjectStart,           vec![]),
                  (F64Value(1.0),       vec![Key("a")]),
                  (ArrayStart,          vec![Key("b")]),
                    (BooleanValue(true),vec![Key("b"), Index(0)]),
                  (ArrayEnd,            vec![Key("b")]),
                (ObjectEnd,             vec![]),
            ]
        );
        assert_stream_equal(
            r#"{
                "a": 1.0,
                "b": [
                    true,
                    "foo\nbar",
                    { "c": {"d": null} }
                ]
            }"#,
            vec![
                (ObjectStart,                   vec![]),
                  (F64Value(1.0),               vec![Key("a")]),
                  (ArrayStart,                  vec![Key("b")]),
                    (BooleanValue(true),        vec![Key("b"), Index(0)]),
                    (StringValue("foo\nbar".to_string()),  vec![Key("b"), Index(1)]),
                    (ObjectStart,               vec![Key("b"), Index(2)]),
                      (ObjectStart,             vec![Key("b"), Index(2), Key("c")]),
                        (NullValue,             vec![Key("b"), Index(2), Key("c"), Key("d")]),
                      (ObjectEnd,               vec![Key("b"), Index(2), Key("c")]),
                    (ObjectEnd,                 vec![Key("b"), Index(2)]),
                  (ArrayEnd,                    vec![Key("b")]),
                (ObjectEnd,                     vec![]),
            ]
        );
    }

    #[test]
    #[cfg_attr(target_word_size = "32", ignore)] // FIXME(#14064)
    fn test_read_array_streaming() {
        assert_stream_equal(
            "[]",
            vec![
                (ArrayStart, vec![]),
                (ArrayEnd,   vec![]),
            ]
        );
        assert_stream_equal(
            "[ ]",
            vec![
                (ArrayStart, vec![]),
                (ArrayEnd,   vec![]),
            ]
        );
        assert_stream_equal(
            "[true]",
            vec![
                (ArrayStart,             vec![]),
                    (BooleanValue(true), vec![Index(0)]),
                (ArrayEnd,               vec![]),
            ]
        );
        assert_stream_equal(
            "[ false ]",
            vec![
                (ArrayStart,              vec![]),
                    (BooleanValue(false), vec![Index(0)]),
                (ArrayEnd,                vec![]),
            ]
        );
        assert_stream_equal(
            "[null]",
            vec![
                (ArrayStart,    vec![]),
                    (NullValue, vec![Index(0)]),
                (ArrayEnd,      vec![]),
            ]
        );
        assert_stream_equal(
            "[3, 1]",
            vec![
                (ArrayStart,      vec![]),
                    (U64Value(3), vec![Index(0)]),
                    (U64Value(1), vec![Index(1)]),
                (ArrayEnd,        vec![]),
            ]
        );
        assert_stream_equal(
            "\n[3, 2]\n",
            vec![
                (ArrayStart,      vec![]),
                    (U64Value(3), vec![Index(0)]),
                    (U64Value(2), vec![Index(1)]),
                (ArrayEnd,        vec![]),
            ]
        );
        assert_stream_equal(
            "[2, [4, 1]]",
            vec![
                (ArrayStart,           vec![]),
                    (U64Value(2),      vec![Index(0)]),
                    (ArrayStart,       vec![Index(1)]),
                        (U64Value(4),  vec![Index(1), Index(0)]),
                        (U64Value(1),  vec![Index(1), Index(1)]),
                    (ArrayEnd,         vec![Index(1)]),
                (ArrayEnd,             vec![]),
            ]
        );

        assert_eq!(last_event("["), Error(SyntaxError(EOFWhileParsingValue, 1,  2)));

        assert_eq!(super::json_from_str_non_strict("["),     Err(SyntaxError(EOFWhileParsingValue, 1, 2)));
        assert_eq!(super::json_from_str_non_strict("[1"),    Err(SyntaxError(EOFWhileParsingArray, 1, 3)));
        assert_eq!(super::json_from_str_non_strict("[1,"),   Err(SyntaxError(EOFWhileParsingValue, 1, 4)));
        assert_eq!(super::json_from_str_non_strict("[1,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
        assert_eq!(super::json_from_str_non_strict("[6 7]"), Err(SyntaxError(InvalidSyntax,        1, 4)));

    }
    #[test]
    fn test_trailing_characters_streaming() {
        assert_eq!(last_event("nulla"),  Error(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(last_event("truea"),  Error(SyntaxError(TrailingCharacters, 1, 5)));
        assert_eq!(last_event("falsea"), Error(SyntaxError(TrailingCharacters, 1, 6)));
        assert_eq!(last_event("1a"),     Error(SyntaxError(TrailingCharacters, 1, 2)));
        assert_eq!(last_event("[]a"),    Error(SyntaxError(TrailingCharacters, 1, 3)));
        assert_eq!(last_event("{}a"),    Error(SyntaxError(TrailingCharacters, 1, 3)));
    }
    #[test]
    fn test_read_identifiers_streaming() {
        assert_eq!(Parser::new("null".chars()).next(), Some(NullValue));
        assert_eq!(Parser::new("true".chars()).next(), Some(BooleanValue(true)));
        assert_eq!(Parser::new("false".chars()).next(), Some(BooleanValue(false)));

        assert_eq!(last_event("n"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(last_event("nul"),  Error(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(last_event("t"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(last_event("truz"), Error(SyntaxError(InvalidSyntax, 1, 4)));
        assert_eq!(last_event("f"),    Error(SyntaxError(InvalidSyntax, 1, 2)));
        assert_eq!(last_event("faz"),  Error(SyntaxError(InvalidSyntax, 1, 3)));
    }

    #[test]
    fn test_stack() {
        let mut stack = Stack::new();

        assert!(stack.is_empty());
        assert!(stack.len() == 0);
        assert!(!stack.last_is_index());

        stack.push_index(0);
        stack.bump_index();

        assert!(stack.len() == 1);
        assert!(stack.is_equal_to(&[Index(1)]));
        assert!(stack.starts_with(&[Index(1)]));
        assert!(stack.ends_with(&[Index(1)]));
        assert!(stack.last_is_index());
        assert!(stack.get(0) == Index(1));

        stack.push_key("foo".to_string());

        assert!(stack.len() == 2);
        assert!(stack.is_equal_to(&[Index(1), Key("foo")]));
        assert!(stack.starts_with(&[Index(1), Key("foo")]));
        assert!(stack.starts_with(&[Index(1)]));
        assert!(stack.ends_with(&[Index(1), Key("foo")]));
        assert!(stack.ends_with(&[Key("foo")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == Index(1));
        assert!(stack.get(1) == Key("foo"));

        stack.push_key("bar".to_string());

        assert!(stack.len() == 3);
        assert!(stack.is_equal_to(&[Index(1), Key("foo"), Key("bar")]));
        assert!(stack.starts_with(&[Index(1)]));
        assert!(stack.starts_with(&[Index(1), Key("foo")]));
        assert!(stack.starts_with(&[Index(1), Key("foo"), Key("bar")]));
        assert!(stack.ends_with(&[Key("bar")]));
        assert!(stack.ends_with(&[Key("foo"), Key("bar")]));
        assert!(stack.ends_with(&[Index(1), Key("foo"), Key("bar")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == Index(1));
        assert!(stack.get(1) == Key("foo"));
        assert!(stack.get(2) == Key("bar"));

        stack.pop();

        assert!(stack.len() == 2);
        assert!(stack.is_equal_to(&[Index(1), Key("foo")]));
        assert!(stack.starts_with(&[Index(1), Key("foo")]));
        assert!(stack.starts_with(&[Index(1)]));
        assert!(stack.ends_with(&[Index(1), Key("foo")]));
        assert!(stack.ends_with(&[Key("foo")]));
        assert!(!stack.last_is_index());
        assert!(stack.get(0) == Index(1));
        assert!(stack.get(1) == Key("foo"));
    }

    // fixme: uncomment this
    // #[test]
    // fn test_bad_json_stack_depleted() {
    //     #[derive(Debug, RustcDecodable)]
    //     enum ChatEvent {
    //         Variant(i32)
    //     }
    //     let serialized = "{\"variant\": \"Variant\", \"fields\": []}";
    //     let r: Result<ChatEvent, _> = super::decode_non_strict(serialized);
    //     assert!(r.unwrap_err() == EOF);
    // }

    #[bench]
    fn bench_streaming_small(b: &mut Bencher) {
        b.iter( || {
            let mut parser = Parser::new(
                r#"{
                    "a": 1.0,
                    "b": [
                        true,
                        "foo\nbar",
                        { "c": {"d": null} }
                    ]
                }"#.chars()
            );
            loop {
                match parser.next() {
                    None => return,
                    _ => {}
                }
            }
        });
    }
    #[bench]
    fn bench_small(b: &mut Bencher) {
        b.iter( || {
            let _ = super::json_from_str_non_strict(r#"{
                "a": 1.0,
                "b": [
                    true,
                    "foo\nbar",
                    { "c": {"d": null} }
                ]
            }"#);
        });
    }

    fn big_json() -> string::String {
        let mut src = "[\n".to_string();
        for _ in 0..500 {
            src.push_str(r#"{ "a": true, "b": null, "c":3.1415, "d": "Hello world", "e": \
                            [1,2,3]},"#);
        }
        src.push_str("{}]");
        return src;
    }

    #[bench]
    fn bench_streaming_large(b: &mut Bencher) {
        let src = big_json();
        b.iter( || {
            let mut parser = Parser::new(src.chars());
            loop {
                match parser.next() {
                    None => return,
                    _ => {}
                }
            }
        });
    }
    #[bench]
    fn bench_large(b: &mut Bencher) {
        let src = big_json();
        b.iter( || { let _ = super::json_from_str_non_strict(&src); });
    }
}
