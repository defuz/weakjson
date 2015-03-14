use rustc_serialize::json::Json::*;

#[test]
fn test_single_quote_string() {
    assert_eq!(super::json_from_str_non_strict("'foo'"), Ok(String("foo".to_string())));
    assert_eq!(super::json_from_str_non_strict("'foo\\'bar'"), Ok(String("foo'bar".to_string())));
    assert_eq!(super::json_from_str_non_strict("'foo\"bar'"), Ok(String("foo\"bar".to_string())));
}
