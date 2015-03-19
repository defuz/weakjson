use super::test::Bencher;

use rustc_serialize::json::Json::*;
use rustc_serialize::json::ErrorCode::*;
use rustc_serialize::json::ParserError::*;
use rustc_serialize::json::DecoderError::*;
use rustc_serialize::json::JsonEvent::*;

use parsing::{Parser, StackElement};
use parsing::StackElement::*;

use rustc_serialize::json::{Json, DecodeResult, JsonEvent};

use std::collections::BTreeMap;
use std::{i64, u64, string};

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
    assert_eq!(super::json_from_str_non_strict("+"),   Err(SyntaxError(InvalidNumber, 1, 2)));
    assert_eq!(super::json_from_str_non_strict("."),   Err(SyntaxError(InvalidNumber, 1, 2)));
    assert_eq!(super::json_from_str_non_strict("-"),   Err(SyntaxError(InvalidNumber, 1, 2)));
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
        let (ref expected_evt, _) = expected[i];
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

#[bench]
fn bench_decode_hex_escape(b: &mut Bencher) {
    let mut src = "\"".to_string();
    for _ in 0..10 {
        src.push_str("\\uF975\\uf9bc\\uF9A0\\uF9C4\\uF975\\uf9bc\\uF9A0\\uF9C4");
    }
    src.push_str("\"");
    b.iter( || {
        let _ = Json::from_str(&src);
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
