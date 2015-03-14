#![cfg_attr(test, feature(test))]

extern crate "rustc-serialize" as rustc_serialize;

#[cfg(test)]
extern crate test;

mod parsing;

#[cfg(test)]
mod standart_tests;

#[cfg(test)]
mod weak_tests;

use parsing::{Builder};

use rustc_serialize::Decodable;
use rustc_serialize::json::Json;
use rustc_serialize::json::Decoder;
use rustc_serialize::json::DecoderError;
use rustc_serialize::json::ParserError;
use rustc_serialize::json::BuilderError;
use rustc_serialize::json::ErrorCode;

use std::{str, io};

/// Shortcut function to decode a JSON `&str` into an object
pub fn decode_non_strict<T: Decodable>(s: &str) -> Result<T, DecoderError> {
    let json = match json_from_str_non_strict(s) {
        Ok(x) => x,
        Err(e) => return Err(DecoderError::ParseError(e))
    };

    let mut decoder = Decoder::new(json);
    Decodable::decode(&mut decoder)
}

fn io_error_to_error(err: io::Error) -> ParserError {
    ParserError::IoError(err)
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
        _       => return Err(ParserError::SyntaxError(ErrorCode::NotUtf8, 0, 0))
    };
    let mut builder = Builder::new(s.chars());
    builder.build()
}

/// Decodes a json value from a string
pub fn json_from_str_non_strict(s: &str) -> Result<Json, BuilderError> {
    let mut builder = Builder::new(s.chars());
    builder.build()
}
