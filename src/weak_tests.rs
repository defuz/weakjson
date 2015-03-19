use rustc_serialize::json::Json::*;

use rustc_serialize::json::ErrorCode::*;
use rustc_serialize::json::ParserError::*;

use std::f64;

#[test]
fn test_single_quote_string() {
    assert_eq!(super::json_from_str_non_strict("'"), Err(SyntaxError(EOFWhileParsingString, 1, 2)));
    assert_eq!(super::json_from_str_non_strict("'lol"), Err(SyntaxError(EOFWhileParsingString, 1, 5)));

    assert_eq!(super::json_from_str_non_strict("''"), Ok(String("".to_string())));
    assert_eq!(super::json_from_str_non_strict("'foo'"), Ok(String("foo".to_string())));
    assert_eq!(super::json_from_str_non_strict("'foo\\'bar'"), Ok(String("foo'bar".to_string())));
    assert_eq!(super::json_from_str_non_strict("'foo\"bar'"), Ok(String("foo\"bar".to_string())));
}

#[test]
fn test_ignore_invalid_escaping() {
    assert_eq!(super::json_from_str_non_strict("\"foo\\abar\""), Ok(String("fooabar".to_string())));
}

#[test]
fn test_multiline_string() {
    assert_eq!(super::json_from_str_non_strict("\"\n\""), Ok(String("\n".to_string())));
    assert_eq!(super::json_from_str_non_strict("\"\n\n\""), Ok(String("\n\n".to_string())));
    assert_eq!(super::json_from_str_non_strict("\"foo\nbar\""), Ok(String("foo\nbar".to_string())));
    assert_eq!(super::json_from_str_non_strict("\"foo\\\nbar\""), Ok(String("foo\nbar".to_string())));
}

#[test]
fn test_number_with_plus_sign() {
    assert_eq!(super::json_from_str_non_strict("+3"), Ok(U64(3)));
    assert_eq!(super::json_from_str_non_strict("+3.1"), Ok(F64(3.1)));
    assert_eq!(super::json_from_str_non_strict("+0.4"), Ok(F64(0.4)));
    assert_eq!(super::json_from_str_non_strict("+0.4e5"), Ok(F64(0.4e5)));
    assert_eq!(super::json_from_str_non_strict("+0.4e+15"), Ok(F64(0.4e15)));
    assert_eq!(super::json_from_str_non_strict("+0.4e-01"), Ok(F64(0.4e-01)));
}

#[test]
fn test_trailing_zeros() {
    assert_eq!(super::json_from_str_non_strict("00"), Ok(U64(0)));
    assert_eq!(super::json_from_str_non_strict("01"), Ok(U64(1)));
    assert_eq!(super::json_from_str_non_strict("-01"), Ok(I64(-1)));

    assert_eq!(super::json_from_str_non_strict("0.0"), Ok(F64(0.0)));
    assert_eq!(super::json_from_str_non_strict("0.4e05"), Ok(F64(0.4e5)));
    assert_eq!(super::json_from_str_non_strict("0.4e+015"), Ok(F64(0.4e15)));
    assert_eq!(super::json_from_str_non_strict("0.4e-1"), Ok(F64(0.4e-01)));
}

#[test]
fn test_infinity_and_nan() {
    match super::json_from_str_non_strict("NaN") {
        Ok(F64(x)) => assert!(x != x, "Unable to parse NaN"),
        _ => panic!("Unable to parse NaN")
    }
    match super::json_from_str_non_strict("+NaN") {
        Ok(F64(x)) => assert!(x != x, "Unable to parse +NaN"),
        _ => panic!("Unable to parse +NaN")
    }
    match super::json_from_str_non_strict("-NaN") {
        Ok(F64(x)) => assert!(x != x, "Unable to parse -NaN"),
        _ => panic!("Unable to parse -NaN")
    }

    assert_eq!(super::json_from_str_non_strict("Infinity"), Ok(F64(f64::INFINITY)));
    assert_eq!(super::json_from_str_non_strict("+Infinity"), Ok(F64(f64::INFINITY)));
    assert_eq!(super::json_from_str_non_strict("-Infinity"), Ok(F64(f64::NEG_INFINITY)));
}

#[test]
fn test_leading_and_trailing_decimal_point() {
    assert_eq!(super::json_from_str_non_strict("3."), Ok(F64(3.0)));
    assert_eq!(super::json_from_str_non_strict(".1"), Ok(F64(0.1)));
    assert_eq!(super::json_from_str_non_strict("+.1"), Ok(F64(0.1)));
    assert_eq!(super::json_from_str_non_strict("-.1"), Ok(F64(-0.1)));

    assert_eq!(super::json_from_str_non_strict("0."), Ok(F64(0.0)));
    assert_eq!(super::json_from_str_non_strict(".0"), Ok(F64(0.0)));
    assert_eq!(super::json_from_str_non_strict("+0."), Ok(F64(0.0)));
    assert_eq!(super::json_from_str_non_strict("-.0"), Ok(F64(0.0)));

    assert_eq!(super::json_from_str_non_strict("3.e5"), Ok(F64(3.0e5)));
    assert_eq!(super::json_from_str_non_strict(".1e5"), Ok(F64(0.1e5)));

    assert_eq!(super::json_from_str_non_strict(".e5"), Err(SyntaxError(InvalidNumber, 1, 2)));
    assert_eq!(super::json_from_str_non_strict("e5"), Err(SyntaxError(InvalidSyntax, 1, 1)));
}

#[test]
fn test_hexadecimal() {
    assert_eq!(super::json_from_str_non_strict("0xff"), Ok(U64(255)));
    assert_eq!(super::json_from_str_non_strict("0xFF"), Ok(U64(255)));
    assert_eq!(super::json_from_str_non_strict("0Xff"), Ok(U64(255)));
    assert_eq!(super::json_from_str_non_strict("-0Xff"), Ok(I64(-255)));

    assert_eq!(super::json_from_str_non_strict("0x"), Err(SyntaxError(InvalidNumber, 1, 3)));
    assert_eq!(super::json_from_str_non_strict("0x.0"), Err(SyntaxError(InvalidNumber, 1, 3)));
    assert_eq!(super::json_from_str_non_strict("0xf.0"), Err(SyntaxError(TrailingCharacters, 1, 4)));
}


// #[test]
// fn test_trailing_comma_in_array() {
//     // assert_eq!(super::json_from_str_non_strict("[,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
//     // assert_eq!(super::json_from_str_non_strict("[,1]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
//     // assert_eq!(super::json_from_str_non_strict("[,,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
//     // assert_eq!(super::json_from_str_non_strict("[1,,]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));
//     // assert_eq!(super::json_from_str_non_strict("[1,,2]"),  Err(SyntaxError(InvalidSyntax,        1, 4)));

//     assert_eq!(super::json_from_str_non_strict("[1,]"),
//                  Ok(Array(vec![U64(1)])));
//     assert_eq!(super::json_from_str_non_strict("[1, 2,]"),
//                  Ok(Array(vec![U64(1), U64(2)])));
// }
