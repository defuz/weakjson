use rustc_serialize::json::Json::*;

use rustc_serialize::json::ErrorCode::*;
use rustc_serialize::json::ParserError::*;

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
    assert_eq!(super::json_from_str_non_strict(" +3 "), Ok(U64(3)));
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
