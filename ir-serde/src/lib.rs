//! # IR Serde
//!
//! This crate provides a simple way to serialize and deserialize data structures to and from a simple intermediate representation (IR)
//! The IR is a simple text format that can be easily read and written by humans.
//!
//! ## Basic Usage
//!
//! To use this crate, you need to derive the `IRSerde` trait for your data structures.
//!
//! ```rust
//! use ir_serde::{IRSerde, Serializer, IRSerializer, IRDeserializer};
//!
//! #[derive(IRSerde)]
//! struct MyStruct {
//!     a: u32,
//!     b: String,
//! }
//!
//! fn main() {
//!     let data = MyStruct { a: 1, b: "hello".to_string() };
//!     let mut ctx = ();
//!     let mut ser = Serializer::new(&mut ctx);
//!     ser.serialize(&data);
//!     println!("serialized: {:}", ser.buf);
//! }
//! ```
//!
//! ## Context
//!
//! The `IRSerde` derive macro can take an optional context parameter. This context is passed to the serializer and deserializer.
//!
//! ```rust
//! use ir_serde::{IRSerde, Serializer, IRSerializer, IRDeserializer};
//!
//! struct MyContext {
//!     value_types: Vec<String>,
//! }
//! struct ValueId(usize);
//! impl IRSerde<MyContext> for ValueId {
//!     fn ir_serialize(&self, ser: &mut dyn IRSerializer<MyContext>) {
//!         let tpe = ser.get_ctx().value_types[self.0].clone();
//!         self.0.ir_serialize(ser);
//!         ser.serialize_punct(":");
//!         ser.serialize_keyword(&tpe);
//!     }
//!     fn ir_deserialize(der: &mut dyn IRDeserializer<MyContext>) -> Result<Self, String> {
//!         let id = usize::ir_deserialize(der)?;
//!         der.expect_punct(":")?;
//!         let name = der.deserialize_keyword()?.to_owned();
//!         let ctx = der.get_ctx();
//!         if id >= ctx.value_types.len() {
//!             ctx.value_types.resize(id + 1, "".to_string());
//!         }
//!         ctx.value_types[id] = name.to_string();
//!         Ok(Self(id))
//!     }
//! }
//!
//! #[derive(IRSerde)]
//! #[context(MyContext)]
//! struct MyStruct {
//!     a: ValueId,
//!     b: String,
//! }
//!
//! fn main() {
//!     let data = MyStruct { a: ValueId(1), b: "hello".to_string() };
//!     let mut ctx = MyContext { value_types: vec!["".to_string(), "ty".to_string()] };
//!     let mut ser = Serializer::new(&mut ctx);
//!     ser.serialize(&data);
//!     println!("serialized: {:}", ser.buf);
//! }
//! ```
//!
//! ## Attrs
//!
//! The `IRSerde` derive macro supports the following attributes:
//!
//! - `#[keyword(name)]`: Insert a keyword before the field or variant.
//!     - For enums, the keyword is the name of the variant
//! - `#[punct(name)]`: Insert a punctuation before the field or variant.
//!     - For braces, brackets, and parentheses, the name should be `lbrace`, `rbrace`, `lsquare`, `rsquare`, `lparen`, `rparen`
//! - `#[mtline]`: Specify the next container should be placed on a new line.
//! - `#[punct_after(name)]`: Insert a punctuation after the field or variant, useful for the last field in a struct.
//!
//! An example of using these attributes:
//!
//! ```rust
//! use ir_serde::{IRSerde, Serializer, Deserializer, IRSerializer, IRDeserializer};
//!
//! #[derive(IRSerde)]
//! struct MyStruct {
//!     #[punct(lparen)]#[keyword(a)]#[punct(:)]
//!     a: u32,
//!     #[punct(,)]#[keyword(b)]#[punct(:)]
//!     #[punct_after(rparen)]
//!     b: String,
//! }
//!
//! fn main() {
//!     let data = MyStruct { a: 1, b: "hello".to_string() };
//!     let mut ctx = ();
//!     let mut ser = Serializer::new(&mut ctx);
//!     ser.serialize(&data);
//!     // This will print ( a: 1, b: "hello" )
//!     println!("serialized: {:}", ser.buf);
//!     let mut de = Deserializer::new(&ser.buf, &mut ctx);
//!     let res: MyStruct = de.deserialize();
//!     assert_eq!(res.a, 1);
//! }
//! ```
//!
//! ## Default Implementation
//!
//! By default, this crate provides an implementation for the `IRSerializer` and `IRDeserializer` traits.
//!
//! The default implementation uses the `logos` crate to tokenize the input, and the `codespan_reporting` crate to report errors.
//!
//! In the default implementation, some punctuation characters are treated as special characters:
//!
//! - `{` and `}`: Increase and decrease the indentation level.
//! - `;`: Start a new line.
//!
//! The default implementation also supports the `#[mtline]` attribute to start a new line after the field or variant.
//!
//! ```rust
//! use ir_serde::{IRSerde, Serializer, Deserializer, IRSerializer, IRDeserializer};
//!
//! #[derive(IRSerde)]
//! struct MyStruct {
//!     #[punct(lbrace)]
//!     #[keyword(b)]#[punct(:)]
//!     b: String,
//!     #[punct(;)] // newline
//!     #[keyword(a)]#[punct(:)]
//!     #[mtline]   // use brace and semicolon to display the next container
//!     #[punct_after(rbrace)]
//!     a: Vec<u32>,
//! }
//!
//! fn main() {
//!     let data = MyStruct { a: vec![1, 2], b: "hello".to_string() };
//!     let mut ctx = ();
//!     let mut ser = Serializer::new(&mut ctx);
//!     ser.serialize(&data);
//!     // This will print
//!     // {
//!     //   b: "hello";
//!     //   a: {
//!     //     1;
//!     //     2
//!     //   }
//!     // }
//!     println!("serialized:\n{:}", ser.buf);
//!     let mut de = Deserializer::new(&ser.buf, &mut ctx);
//!     let res: MyStruct = de.deserialize();
//!     assert_eq!(res.a, vec![1, 2]);
//! }
//! ```
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    hash::Hash,
    ops::Range,
};

#[cfg(feature = "derive")]
/// Derive macro for the IRSerde trait
pub use ir_serde_macros::IRSerde;

/// Intermediate representation token
///
/// The `Token` enum represents the different types of tokens in the intermediate representation.
///
/// - `Number`: A number token, regex: `[+-]?[0-9]+(.[+-]?[0-9]+)?(e[+-]?[0-9]+)?`
/// - `Keyword`: A keyword token, regex: `[a-zA-Z_][a-zA-Z0-9_]*`
/// - `Str`: A string token, regex: `"[^"]*"`
/// - `Punct`: A punctuation token, regex: `[!@#$%^&*(){}/\|;:',.<>\[\]=]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Number,
    Keyword,
    Str,
    Punct,
}

fn escape_str(s: &str) -> String {
    let mut res = String::new();
    for c in s.chars() {
        match c {
            '\n' => res.push_str("\\n"),
            '\r' => res.push_str("\\r"),
            '\t' => res.push_str("\\t"),
            '\\' => res.push_str("\\\\"),
            '"' => res.push_str("\\\""),
            _ => res.push(c),
        }
    }
    res
}
fn unescape_str(s: &str) -> String {
    let mut res = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => res.push('\n'),
                Some('r') => res.push('\r'),
                Some('t') => res.push('\t'),
                Some('\\') => res.push('\\'),
                Some('"') => res.push('"'),
                Some(c) => {
                    res.push('\\');
                    res.push(c);
                }
                None => res.push('\\'),
            }
        } else {
            res.push(c);
        }
    }
    res
}

/// Serialize data to an intermediate representation
///
/// The `IRSerializer` trait provides methods to serialize data to an intermediate representation.
///
/// The `C` type parameter is the context type, which can be used to store additional information during serialization.
///
pub trait IRSerializer<C> {
    fn serialize_raw(&mut self, data: &str);
    fn serialize_number(&mut self, value: &str) {
        self.serialize_raw(value);
    }
    fn serialize_keyword(&mut self, value: &str) {
        self.serialize_raw(value);
    }
    fn serialize_str_escaped(&mut self, value: &str) {
        self.serialize_raw(&format!("\"{}\"", value));
    }
    fn serialize_str(&mut self, value: &str) {
        self.serialize_str_escaped(&escape_str(value));
    }
    fn serialize_punct(&mut self, value: &str) {
        self.serialize_raw(value);
    }
    fn take_container_puncts(&mut self) -> (&'static str, &'static str, &'static str) {
        ("(", ",", ")")
    }
    fn set_container_puncts(
        &mut self,
        _start: &'static str,
        _sep: &'static str,
        _end: &'static str,
    ) {
    }
    fn get_ctx(&mut self) -> &mut C;
}

macro_rules! decl_de_method {
    ($des_func:ident, $exp_func:ident, $token:pat, $e:expr) => {
        fn $des_func(&mut self) -> Result<&str, String> {
            match self.get()? {
                ($token, value) => Ok(value),
                (tpe, value) => Err(format!("Expected {:?}, got {:?}({:?})", $e, tpe, value)),
            }
        }
        fn $exp_func(&mut self, e: &str) -> Result<(), String> {
            match self.get()? {
                ($token, value) if value == e => Ok(()),
                (tpe, value) => Err(format!(
                    "Expected {:?}({:?}), got {:?}({:?})",
                    $e, e, tpe, value
                )),
            }
        }
    };
}

/// Deserialize data from an intermediate representation
///
/// The `IRDeserializer` trait provides methods to deserialize data from an intermediate representation.
///
/// The `C` type parameter is the context type, which can be used to store additional information during deserialization.
pub trait IRDeserializer<C> {
    fn peek(&mut self) -> Result<(Token, &str), String>;
    fn get(&mut self) -> Result<(Token, &str), String>;
    decl_de_method!(
        deserialize_number,
        expect_number,
        Token::Number,
        Token::Number
    );
    decl_de_method!(
        deserialize_keyword,
        expect_keyword,
        Token::Keyword,
        Token::Keyword
    );
    fn deserialize_str_escaped(&mut self) -> Result<&str, String> {
        let (tok, r) = self.get()?;
        match tok {
            Token::Str => Ok(&r[1..r.len() - 1]),
            tpe => Err(format!("Expected Str, got {:?}({:?})", tpe, r)),
        }
    }
    fn expect_str_escaped(&mut self, s: &str) -> Result<(), String> {
        let get = self.deserialize_str_escaped()?;
        if get == s {
            Ok(())
        } else {
            Err(format!("Expected Str({:?}), got Str({:?})", s, get))
        }
    }
    fn deserialize_str(&mut self) -> Result<String, String> {
        self.deserialize_str_escaped().map(unescape_str)
    }
    fn expect_str(&mut self, s: &str) -> Result<(), String> {
        let get = self.deserialize_str()?;
        if get == s {
            Ok(())
        } else {
            Err(format!("Expected Str({:?}), got Str({:?})", s, get))
        }
    }
    decl_de_method!(deserialize_punct, expect_punct, Token::Punct, Token::Punct);
    fn take_container_puncts(&mut self) -> (&'static str, &'static str, &'static str) {
        ("(", ",", ")")
    }
    fn set_container_puncts(
        &mut self,
        _start: &'static str,
        _sep: &'static str,
        _end: &'static str,
    ) {
    }
    fn get_ctx(&mut self) -> &mut C;
}

/// Deseralize function, with auto type inference
pub fn deserialize_with<C, T: IRSerde<C>>(der: &mut dyn IRDeserializer<C>) -> Result<T, String> {
    T::ir_deserialize(der)
}

/// Serialize and deserialize data to and from an intermediate representation
///
/// The `IRSerde` trait provides methods to serialize and deserialize data to and from an intermediate representation.
///
/// The `C` type parameter is the context type, which can be used to store additional information during serialization and deserialization.
pub trait IRSerde<C>: Sized {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>);
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String>;
}

macro_rules! impl_serde_number {
    ($($(#[$meta:meta])? $t:ty),*) => {
        $(
            $(#[$meta])?
            impl<C> IRSerde<C> for $t {
                fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
                    ser.serialize_number(&self.to_string());
                }
                fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
                    let number = der.deserialize_number()?;
                    number.parse().map_err(|e| format!("Failed to parse number {:?} : {:?}", number, e))
                }
            }
        )*
    }
}
impl_serde_number!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, isize, usize);
#[cfg(feature = "num")]
use num::{BigInt, BigUint};
impl_serde_number!(
    #[cfg(feature = "num")]
    BigInt,
    #[cfg(feature = "num")]
    BigUint
);

impl<C> IRSerde<C> for String {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        ser.serialize_str(self);
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        der.deserialize_str()
    }
}
impl<C> IRSerde<C> for () {
    fn ir_serialize(&self, _ser: &mut dyn IRSerializer<C>) {}
    fn ir_deserialize(_der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        Ok(())
    }
}

pub fn serialize_iterable<'r, C>(
    ser: &mut dyn IRSerializer<C>,
    iter: impl 'r + IntoIterator<Item = &'r (impl 'r + IRSerde<C>)>,
) {
    let (start, sep, end) = ser.take_container_puncts();
    ser.serialize_punct(start);
    let mut first = true;
    for i in iter {
        if !first {
            ser.serialize_punct(sep);
        } else {
            first = false;
        }
        i.ir_serialize(ser);
    }
    ser.serialize_punct(end);
}

impl<C, T: IRSerde<C>> IRSerde<C> for Vec<T> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        serialize_iterable(ser, self.iter());
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        let (start, sep, end) = der.take_container_puncts();
        der.expect_punct(start)?;
        let mut res = Self::new();
        if der.peek()?.1 == end {
            der.deserialize_punct()?;
            return Ok(res);
        }
        res.push(T::ir_deserialize(der)?);
        loop {
            let (tpe, s) = der.peek()?;
            if s == end {
                der.deserialize_punct()?;
                break Ok(res);
            } else if s == sep {
                der.deserialize_punct()?;
                res.push(T::ir_deserialize(der)?);
            } else {
                break Err(format!(
                    "Expected separator {:?} or end {:?}, got {:?}({:?})",
                    sep, end, tpe, s
                ));
            }
        }
    }
}
impl<C, T: IRSerde<C>, const N: usize> IRSerde<C> for [T; N] {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        serialize_iterable(ser, self.iter());
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        let vec = Vec::ir_deserialize(der)?;
        if vec.len() != N {
            return Err(format!(
                "Expected array of size {:?}, got {:?}",
                N,
                vec.len()
            ));
        }
        vec.try_into()
            .map_err(|_| format!("Failed to convert vec to array"))
    }
}
impl<C, T: IRSerde<C>> IRSerde<C> for Option<T> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        serialize_iterable(ser, self.iter());
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        let vec = Vec::ir_deserialize(der)?;
        match vec.len() {
            0 => Ok(None),
            1 => Ok(Some(vec.into_iter().next().unwrap())),
            _ => Err(format!("Option too large, got {:?}", vec.len())),
        }
    }
}
impl<C, T: IRSerde<C>> IRSerde<C> for Box<T> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        self.as_ref().ir_serialize(ser);
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        Ok(Box::new(T::ir_deserialize(der)?))
    }
}
impl<C, T: IRSerde<C>> IRSerde<C> for Range<T> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        ser.serialize_punct("[");
        self.start.ir_serialize(ser);
        ser.serialize_punct(",");
        self.end.ir_serialize(ser);
        ser.serialize_punct("]");
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        der.expect_punct("[")?;
        let start = T::ir_deserialize(der)?;
        der.expect_punct(",")?;
        let end = T::ir_deserialize(der)?;
        der.expect_punct("]")?;
        Ok(start..end)
    }
}

macro_rules! impl_tuple {
    ($(($t1:ident, $($t:ident),*)),*) => {
        $(
            impl<_C, $t1: IRSerde<_C>, $($t: IRSerde<_C>),*> IRSerde<_C> for ($t1, $($t,)*) {
                fn ir_serialize(&self, ser: &mut dyn IRSerializer<_C>) {
                    #[allow(unused)]
                    let (start, sep, end) = ser.take_container_puncts();
                    ser.serialize_punct(start);
                    #[allow(non_snake_case)]
                    let ($t1, $($t,)*) = self;
                    $t1.ir_serialize(ser);
                    $(
                        ser.serialize_punct(sep);
                        $t.ir_serialize(ser);
                    )*
                    ser.serialize_punct(end);
                }
                fn ir_deserialize(der: &mut dyn IRDeserializer<_C>) -> Result<Self, String> {
                    #[allow(unused)]
                    let (start, sep, end) = der.take_container_puncts();
                    der.expect_punct(start)?;
                    let res = (
                        $t1::ir_deserialize(der)?,
                        $({
                            der.expect_punct(sep)?;
                            $t::ir_deserialize(der)?
                        },)*
                    );
                    der.expect_punct(end)?;
                    Ok(res)
                }
            }
        )*
    }
}
impl_tuple! { (A,), (A, B), (A, B, C), (A, B, C, D), (A, B, C, D, E), (A, B, C, D, E, F), (A, B, C, D, E, F, G), (A, B, C, D, E, F, G, H) }

mod detail {
    use super::*;
    use crate as ir_serde;

    #[derive(IRSerde)]
    pub(crate) struct KVHelper<K, V> {
        pub(crate) key: K,
        #[punct(:)]
        pub(crate) value: V,
    }
}

pub fn serialize_kv<'r, C, A: 'r + IRSerde<C>, B: 'r + IRSerde<C>>(
    ser: &mut dyn IRSerializer<C>,
    iter: impl 'r + IntoIterator<Item = (&'r A, &'r B)>,
) {
    let (start, sep, end) = ser.take_container_puncts();
    ser.serialize_punct(start);
    let mut first = true;
    for i in iter {
        if !first {
            ser.serialize_punct(sep);
        } else {
            first = false;
        }
        i.0.ir_serialize(ser);
        ser.serialize_punct(":");
        i.1.ir_serialize(ser);
    }
    ser.serialize_punct(end);
}

macro_rules! impl_map {
    () => {
        fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
            Ok(Vec::<detail::KVHelper<K, V>>::ir_deserialize(der)?
                .into_iter()
                .map(|h| (h.key, h.value))
                .collect())
        }
    };
}

impl<C, K: IRSerde<C> + Eq + Hash, V: IRSerde<C>> IRSerde<C> for HashMap<K, V> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        serialize_kv(ser, self.iter());
    }
    impl_map!();
}
impl<C, K: IRSerde<C> + Ord, V: IRSerde<C>> IRSerde<C> for BTreeMap<K, V> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        serialize_kv(ser, self.iter());
    }
    impl_map!();
}
impl<C, K: IRSerde<C> + Eq + Hash> IRSerde<C> for HashSet<K> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        serialize_iterable(ser, self.iter());
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        Ok(Vec::ir_deserialize(der)?.into_iter().collect())
    }
}
impl<C, K: IRSerde<C> + Ord> IRSerde<C> for BTreeSet<K> {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<C>) {
        serialize_iterable(ser, self.iter());
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<C>) -> Result<Self, String> {
        Ok(Vec::ir_deserialize(der)?.into_iter().collect())
    }
}

#[cfg(feature = "default_impl")]
pub use default::*;

#[cfg(feature = "default_impl")]
mod default {
    use super::*;
    use codespan_reporting::{
        diagnostic::{Diagnostic, Label},
        files::SimpleFiles,
        term::{
            self,
            termcolor::{ColorChoice, StandardStream},
        },
    };
    use logos::{Lexer, Logos};

    #[derive(Logos, Clone, Copy)]
    #[logos(skip r"[ \n\t]")]
    pub enum Token {
        #[regex(r"[+-]?[0-9]+(.[+-]?[0-9]+)?(e[+-]?[0-9]+)?")]
        Number,
        #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
        Keyword,
        #[regex(r#""[^"]*""#)]
        Str,
        #[regex(r#"[!@#$%^&*(){}/\|;:',.<>\[\]=]"#)]
        Punct,
    }
    impl Into<super::Token> for Token {
        fn into(self) -> super::Token {
            match self {
                Token::Number => super::Token::Number,
                Token::Keyword => super::Token::Keyword,
                Token::Str => super::Token::Str,
                Token::Punct => super::Token::Punct,
            }
        }
    }

    /// Default implementation of IRSerializer
    pub struct Serializer<'c, C> {
        pub buf: String,
        pub ctx: &'c mut C,
        container: (&'static str, &'static str, &'static str),
        space: bool,
        ident: u32,
    }
    impl<'c, C> Serializer<'c, C> {
        pub fn new(ctx: &'c mut C) -> Self {
            Self {
                buf: String::new(),
                container: ("(", ",", ")"),
                ctx,
                space: false,
                ident: 0,
            }
        }
        pub fn serialize<T: IRSerde<C>>(&mut self, data: &T) -> &mut Self {
            data.ir_serialize(self);
            self
        }
        fn newline(&mut self) {
            self.buf.push('\n');
            for _ in 0..self.ident {
                self.buf.push_str("  ");
            }
            self.space = false;
        }
    }
    impl<'c, C> IRSerializer<C> for Serializer<'c, C> {
        fn serialize_raw(&mut self, data: &str) {
            if self.space {
                self.buf.push(' ');
            }
            self.buf.push_str(data);
            self.space = true;
        }
        fn serialize_punct(&mut self, value: &str) {
            match value {
                "{" => {
                    self.serialize_raw(value);
                    self.ident += 1;
                    self.newline();
                }
                ";" => {
                    self.space = false;
                    self.serialize_raw(value);
                    self.newline();
                }
                "}" => {
                    self.ident -= 1;
                    self.newline();
                    self.serialize_raw(value);
                }
                "," | ":" => {
                    self.space = false;
                    self.serialize_raw(value);
                }
                _ => self.serialize_raw(value),
            };
        }
        fn set_container_puncts(
            &mut self,
            start: &'static str,
            sep: &'static str,
            end: &'static str,
        ) {
            self.container = (start, sep, end);
        }
        fn take_container_puncts(&mut self) -> (&'static str, &'static str, &'static str) {
            let res = self.container;
            self.container = ("(", ",", ")");
            res
        }
        fn get_ctx(&mut self) -> &mut C {
            self.ctx
        }
    }

    /// Default implementation of IRDeserializer
    pub struct Deserializer<'s, C> {
        pub input: &'s str,
        pub lexer: Lexer<'s, Token>,
        pub peek: Option<(super::Token, &'s str)>,
        pub ctx: &'s mut C,
        container: (&'static str, &'static str, &'static str),
    }
    impl<'s, C> Deserializer<'s, C> {
        pub fn new(input: &'s str, ctx: &'s mut C) -> Self {
            Self {
                input,
                lexer: Token::lexer(input),
                peek: None,
                container: ("(", ",", ")"),
                ctx,
            }
        }
        fn lexer_next(&mut self) -> Result<(super::Token, &'s str), String> {
            match self.lexer.next() {
                Some(Ok(t)) => {
                    let s = self.lexer.slice();
                    Ok((t.into(), s))
                }
                Some(Err(())) => Err("Invalid token for lexer".to_string()),
                None => Err("Unexpected end of input".to_string()),
            }
        }
        pub fn report_error(&self, file_name: &str, err: &str) {
            let mut files = SimpleFiles::new();
            let file_id = files.add(file_name, self.input);
            let diagnostic = Diagnostic::error()
                .with_message(err)
                .with_labels(vec![Label::primary(file_id, self.lexer.span())]);
            let writer = StandardStream::stderr(ColorChoice::Always);
            let config = codespan_reporting::term::Config::default();
            term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        }
        pub fn deserialize<T: IRSerde<C>>(&mut self) -> T {
            match T::ir_deserialize(self) {
                Ok(t) => t,
                Err(e) => {
                    self.report_error("input", &e);
                    panic!("Failed to deserialize")
                }
            }
        }
    }
    impl<'s, C> IRDeserializer<C> for Deserializer<'s, C> {
        fn peek(&mut self) -> Result<(crate::Token, &str), String> {
            match &self.peek {
                Some((t, s)) => Ok((*t, s)),
                None => {
                    let (t, s) = self.lexer_next()?;
                    self.peek = Some((t, s));
                    Ok((t, s))
                }
            }
        }
        fn get(&mut self) -> Result<(crate::Token, &str), String> {
            match self.peek.take() {
                Some((t, s)) => Ok((t, s)),
                None => self.lexer_next(),
            }
        }
        fn get_ctx(&mut self) -> &mut C {
            self.ctx
        }
        fn take_container_puncts(&mut self) -> (&'static str, &'static str, &'static str) {
            let res = self.container;
            self.container = ("(", ",", ")");
            res
        }
        fn set_container_puncts(
            &mut self,
            start: &'static str,
            sep: &'static str,
            end: &'static str,
        ) {
            self.container = (start, sep, end);
        }
    }
}

#[cfg(test)]
mod tests {
    use default::Serializer;

    use super::*;
    use crate as ir_serde;

    struct MyContext {}

    #[derive(IRSerde)]
    #[context(MyContext)]
    struct TestStruct<T> {
        #[keyword(c)]
        a: T,
        #[punct(,)]
        #[mtline]
        b: Vec<String>,
    }

    #[derive(Debug, PartialEq, Eq, IRSerde)]
    enum TestEnum<T> {
        #[keyword(c)]
        A(T),
        #[keyword(d)]
        B(T, T),
        C {
            #[punct(lsquare)]
            #[keyword(a)]
            #[punct(=)]
            a: T,
            #[punct(,)]
            #[keyword(b)]
            #[punct(=)]
            #[punct_after(rsquare)]
            b: T,
        },
    }

    #[derive(IRSerde)]
    struct Simple {
        a: String,
    }

    #[derive(IRSerde)]
    #[context(MyContext)]
    struct SimpleCtx {
        a: String,
    }

    #[test]
    fn test_str() {
        let mut ctx = ();
        let mut ser = Serializer::new(&mut ctx);
        ser.serialize(&Simple {
            a: "hello".to_string(),
        });
        println!("serialized: {:}", ser.buf);
        let mut de = default::Deserializer::new(&ser.buf, &mut ctx);
        let res: Simple = de.deserialize();
        assert_eq!(res.a, "hello");
    }

    #[test]
    fn test_struct() {
        let mut ctx = MyContext {};
        let mut ser = Serializer::new(&mut ctx);
        ser.serialize(&TestStruct {
            a: 1,
            b: vec!["a".to_string(), "b".to_string()],
        });
        println!("serialized: {:}", ser.buf);
        let mut de = default::Deserializer::new(&ser.buf, &mut ctx);
        let res: TestStruct<u32> = de.deserialize();
        assert_eq!(res.a, 1);
    }

    #[test]
    fn test_enum() {
        let mut ctx = MyContext {};
        let mut ser = Serializer::new(&mut ctx);
        ser.serialize(&TestEnum::C { a: 1, b: 2 });
        println!("serialized: {:}", ser.buf);
        let mut de = default::Deserializer::new(&ser.buf, &mut ctx);
        let res: TestEnum<u32> = de.deserialize();
        assert_eq!(res, TestEnum::C { a: 1, b: 2 });
    }

    #[derive(IRSerde)]
    struct MyStruct {
        #[punct(lbrace)]
        #[keyword(b)]
        #[punct(:)]
        b: String,
        #[keyword(a)]
        #[punct(:)]
        #[mtline]
        #[punct_after(rbrace)]
        a: Vec<u32>,
    }

    #[test]
    fn test_mtline() {
        let data = MyStruct {
            a: vec![1, 2],
            b: "hello".to_string(),
        };
        let mut ctx = ();
        let mut ser = Serializer::new(&mut ctx);
        ser.serialize(&data);
        println!("serialized:\n{:}", ser.buf);
        let mut de = Deserializer::new(&ser.buf, &mut ctx);
        let res: MyStruct = de.deserialize();
        assert_eq!(res.a, vec![1, 2]);
    }
}
