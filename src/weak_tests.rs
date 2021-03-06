use rustc_serialize::json::Json;

use rustc_serialize::json::Json::*;

use rustc_serialize::json::ErrorCode::*;
use rustc_serialize::json::ParserError::*;

use std::collections::BTreeMap;
use std::f64;

fn mk_object(items: &[(String, Json)]) -> Json {
    let mut d = BTreeMap::new();

    for item in items.iter() {
        match *item {
            (ref key, ref value) => { d.insert((*key).clone(), (*value).clone()); },
        }
    };

    Object(d)
}

#[test]
fn test_single_quote_string() {
    assert_eq!(super::from_str("'"), Err(SyntaxError(EOFWhileParsingString, 1, 2)));
    assert_eq!(super::from_str("'lol"), Err(SyntaxError(EOFWhileParsingString, 1, 5)));

    assert_eq!(super::from_str("''"), Ok(String("".to_string())));
    assert_eq!(super::from_str("'foo'"), Ok(String("foo".to_string())));
    assert_eq!(super::from_str("'foo\\'bar'"), Ok(String("foo'bar".to_string())));
    assert_eq!(super::from_str("'foo\"bar'"), Ok(String("foo\"bar".to_string())));
}

#[test]
fn test_ignore_invalid_escaping() {
    assert_eq!(super::from_str("\"foo\\abar\""), Ok(String("fooabar".to_string())));
}

#[test]
fn test_multiline_string() {
    assert_eq!(super::from_str("\"\n\""), Ok(String("\n".to_string())));
    assert_eq!(super::from_str("\"\n\n\""), Ok(String("\n\n".to_string())));
    assert_eq!(super::from_str("\"foo\nbar\""), Ok(String("foo\nbar".to_string())));
    assert_eq!(super::from_str("\"foo\\\nbar\""), Ok(String("foo\nbar".to_string())));
}

#[test]
fn test_number_with_plus_sign() {
    assert_eq!(super::from_str("+3"), Ok(U64(3)));
    assert_eq!(super::from_str("+3.1"), Ok(F64(3.1)));
    assert_eq!(super::from_str("+0.4"), Ok(F64(0.4)));
    assert_eq!(super::from_str("+0.4e5"), Ok(F64(0.4e5)));
    assert_eq!(super::from_str("+0.4e+15"), Ok(F64(0.4e15)));
    assert_eq!(super::from_str("+0.4e-01"), Ok(F64(0.4e-01)));
}

#[test]
fn test_trailing_zeros() {
    assert_eq!(super::from_str("00"), Ok(U64(0)));
    assert_eq!(super::from_str("01"), Ok(U64(1)));
    assert_eq!(super::from_str("-01"), Ok(I64(-1)));

    assert_eq!(super::from_str("0.0"), Ok(F64(0.0)));
    assert_eq!(super::from_str("0.4e05"), Ok(F64(0.4e5)));
    assert_eq!(super::from_str("0.4e+015"), Ok(F64(0.4e15)));
    assert_eq!(super::from_str("0.4e-1"), Ok(F64(0.4e-01)));
}

#[test]
fn test_infinity_and_nan() {
    match super::from_str("NaN") {
        Ok(F64(x)) => assert!(x != x, "Unable to parse NaN"),
        _ => panic!("Unable to parse NaN")
    }
    match super::from_str("+NaN") {
        Ok(F64(x)) => assert!(x != x, "Unable to parse +NaN"),
        _ => panic!("Unable to parse +NaN")
    }
    match super::from_str("-NaN") {
        Ok(F64(x)) => assert!(x != x, "Unable to parse -NaN"),
        _ => panic!("Unable to parse -NaN")
    }

    assert_eq!(super::from_str("Infinity"), Ok(F64(f64::INFINITY)));
    assert_eq!(super::from_str("+Infinity"), Ok(F64(f64::INFINITY)));
    assert_eq!(super::from_str("-Infinity"), Ok(F64(f64::NEG_INFINITY)));
}

#[test]
fn test_leading_and_trailing_decimal_point() {
    assert_eq!(super::from_str("3."), Ok(F64(3.0)));
    assert_eq!(super::from_str(".1"), Ok(F64(0.1)));
    assert_eq!(super::from_str("+.1"), Ok(F64(0.1)));
    assert_eq!(super::from_str("-.1"), Ok(F64(-0.1)));

    assert_eq!(super::from_str("0."), Ok(F64(0.0)));
    assert_eq!(super::from_str(".0"), Ok(F64(0.0)));
    assert_eq!(super::from_str("+0."), Ok(F64(0.0)));
    assert_eq!(super::from_str("-.0"), Ok(F64(0.0)));

    assert_eq!(super::from_str("3.e5"), Ok(F64(3.0e5)));
    assert_eq!(super::from_str(".1e5"), Ok(F64(0.1e5)));

    assert_eq!(super::from_str(".e5"), Err(SyntaxError(InvalidNumber, 1, 2)));
    assert_eq!(super::from_str("e5"), Err(SyntaxError(InvalidSyntax, 1, 1)));
}

#[test]
fn test_hexadecimal() {
    assert_eq!(super::from_str("0xff"), Ok(U64(255)));
    assert_eq!(super::from_str("0xFF"), Ok(U64(255)));
    assert_eq!(super::from_str("0Xff"), Ok(U64(255)));
    assert_eq!(super::from_str("-0Xff"), Ok(I64(-255)));

    assert_eq!(super::from_str("0x"), Err(SyntaxError(InvalidNumber, 1, 3)));
    assert_eq!(super::from_str("0x.0"), Err(SyntaxError(InvalidNumber, 1, 3)));
    assert_eq!(super::from_str("0xf.0"), Err(SyntaxError(TrailingCharacters, 1, 4)));
}

#[test]
fn test_comments() {
    assert_eq!(super::from_str("0 // comment"), Ok(U64(0)));
    assert_eq!(super::from_str("0 // comment\n"), Ok(U64(0)));
    assert_eq!(super::from_str("// comment\n0"), Ok(U64(0)));
    assert_eq!(super::from_str(" // comment\n0"), Ok(U64(0)));
    assert_eq!(super::from_str("// comment\n 0"), Ok(U64(0)));

    assert_eq!(super::from_str("0 /* comment */"), Ok(U64(0)));
    assert_eq!(super::from_str("0 // comment\n"), Ok(U64(0)));
    assert_eq!(super::from_str("/* comment */ 0"), Ok(U64(0)));
    assert_eq!(super::from_str("/**/0"), Ok(U64(0)));

    assert_eq!(super::from_str("/* // */0"), Ok(U64(0)));

    assert_eq!(super::from_str("/**/[/**/3/**/,/**/1/**/]/**/"),
               Ok(Array(vec![U64(3), U64(1)])));

    assert_eq!(super::from_str("/**/{/**/\"a\"/**/:/**/3/**/}/**/").unwrap(),
               mk_object(&[("a".to_string(), U64(3))]));

    assert_eq!(super::from_str("//c\n[//c\n3//c\n,//c\n1//c\n]//c\n"),
               Ok(Array(vec![U64(3), U64(1)])));

    assert_eq!(super::from_str("//c\n{//c\n\"a\"//c\n://c\n3//c\n}//c\n").unwrap(),
               mk_object(&[("a".to_string(), U64(3))]));

    assert_eq!(super::from_str("0/*"), Err(SyntaxError(InvalidSyntax, 1, 4)));
    assert_eq!(super::from_str("/*/0"), Err(SyntaxError(InvalidSyntax, 1, 5)));
}

#[test]
fn test_trailing_comma_in_array() {
    assert_eq!(super::from_str("[1,]"), Ok(Array(vec![U64(1)])));
    assert_eq!(super::from_str("[1, 2,]"), Ok(Array(vec![U64(1), U64(2)])));

    assert_eq!(super::from_str("[,]"), Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(super::from_str("[,1]"), Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(super::from_str("[,,]"), Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(super::from_str("[1,,]"), Err(SyntaxError(InvalidSyntax, 1, 4)));
    assert_eq!(super::from_str("[1,,2]"), Err(SyntaxError(InvalidSyntax, 1, 4)));
}

#[test]
fn test_trailing_comma_in_object() {
    assert_eq!(super::from_str("{\"a\": 1,}").unwrap(),
               mk_object(&[("a".to_string(), U64(1))]));
    assert_eq!(super::from_str("{\"a\":1, \"b\":2,}").unwrap(),
               mk_object(&[("a".to_string(), U64(1)), ("b".to_string(), U64(2))]));

    assert_eq!(super::from_str("{,}"), Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(super::from_str("{,\"a\": 1}"), Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(super::from_str("{,,}"), Err(SyntaxError(InvalidSyntax, 1, 2)));
    assert_eq!(super::from_str("{\"a\": 1,,}"), Err(SyntaxError(InvalidSyntax, 1, 9)));
    assert_eq!(super::from_str("{\"a\": 1,, \"a\": 2}"), Err(SyntaxError(InvalidSyntax, 1, 9)));
}

#[test]
fn test_unquoted_keys_in_object() {
    assert_eq!(super::from_str("{a: 1}").unwrap(), mk_object(&[("a".to_string(), U64(1))]));
    assert_eq!(super::from_str("{$a: 1}").unwrap(), mk_object(&[("$a".to_string(), U64(1))]));
    assert_eq!(super::from_str("{_1: 1}").unwrap(), mk_object(&[("_1".to_string(), U64(1))]));

    assert_eq!(super::from_str("{ф: 1}"), Err(SyntaxError(InvalidSyntax, 1, 2)));
}

#[test]
fn test_numeric_keys_in_object() {
    assert_eq!(super::from_str("{1: 1}").unwrap(), mk_object(&[("1".to_string(), U64(1))]));
    assert_eq!(super::from_str("{01: 1}").unwrap(), mk_object(&[("1".to_string(), U64(1))]));
    assert_eq!(super::from_str("{123: 1}").unwrap(), mk_object(&[("123".to_string(), U64(1))]));

    assert_eq!(super::from_str("{0a: 1}"), Err(SyntaxError(ExpectedColon, 1, 3)));
    assert_eq!(super::from_str("{0.2: 1}"), Err(SyntaxError(ExpectedColon, 1, 3)));
    assert_eq!(super::from_str("{-3: 1}"), Err(SyntaxError(InvalidSyntax, 1, 2)));
}
