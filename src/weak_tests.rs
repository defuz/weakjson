use rustc_serialize::json::Json::*;

#[test]
fn test_single_quote_string() {
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
