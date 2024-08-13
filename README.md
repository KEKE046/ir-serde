# IR Serde

This crate provides a simple way to serialize and deserialize data structures to and from a simple intermediate representation (IR)
The IR is a simple text format that can be easily read and written by humans.

## Basic Usage

To use this crate, you need to derive the `IRSerde` trait for your data structures.

```rust
use ir_serde::{IRSerde, Serializer, IRSerializer, IRDeserializer};

#[derive(IRSerde)]
struct MyStruct {
    a: u32,
    b: String,
}

fn main() {
    let data = MyStruct { a: 1, b: "hello".to_string() };
    let mut ctx = ();
    let mut ser = Serializer::new(&mut ctx);
    ser.serialize(&data);
    println!("serialized: {:}", ser.buf);
}
```

## Context

The `IRSerde` derive macro can take an optional context parameter. This context is passed to the serializer and deserializer.

```rust
use ir_serde::{IRSerde, Serializer, IRSerializer, IRDeserializer};

struct MyContext {
    value_types: Vec<String>,
}
struct ValueId(usize);
impl IRSerde<MyContext> for ValueId {
    fn ir_serialize(&self, ser: &mut dyn IRSerializer<MyContext>) {
        let tpe = ser.get_ctx().value_types[self.0].clone();
        self.0.ir_serialize(ser);
        ser.serialize_punct(":");
        ser.serialize_keyword(&tpe);
    }
    fn ir_deserialize(der: &mut dyn IRDeserializer<MyContext>) -> Result<Self, String> {
        let id = usize::ir_deserialize(der)?;
        der.expect_punct(":")?;
        let name = der.deserialize_keyword()?.to_owned();
        let ctx = der.get_ctx();
        if id >= ctx.value_types.len() {
            ctx.value_types.resize(id + 1, "".to_string());
        }
        ctx.value_types[id] = name.to_string();
        Ok(Self(id))
    }
}

#[derive(IRSerde)]
#[context(MyContext)]
struct MyStruct {
    a: ValueId,
    b: String,
}

fn main() {
    let data = MyStruct { a: ValueId(1), b: "hello".to_string() };
    let mut ctx = MyContext { value_types: vec!["".to_string(), "ty".to_string()] };
    let mut ser = Serializer::new(&mut ctx);
    ser.serialize(&data);
    println!("serialized: {:}", ser.buf);
}
```

## Attrs

The `IRSerde` derive macro supports the following attributes:

- `#[keyword(name)]`: Insert a keyword before the field or variant.
    - For enums, the keyword is the name of the variant
- `#[punct(name)]`: Insert a punctuation before the field or variant.
    - For braces, brackets, and parentheses, the name should be `lbrace`, `rbrace`, `lsquare`, `rsquare`, `lparen`, `rparen`
- `#[mtline]`: Specify the next container should be placed on a new line.
- `#[punct_after(name)]`: Insert a punctuation after the field or variant, useful for the last field in a struct.

An example of using these attributes:

```rust
use ir_serde::{IRSerde, Serializer, Deserializer, IRSerializer, IRDeserializer};

#[derive(IRSerde)]
struct MyStruct {
    #[punct(lparen)]#[keyword(a)]#[punct(:)]
    a: u32,
    #[punct(,)]#[keyword(b)]#[punct(:)]
    #[punct_after(rparen)]
    b: String,
}

fn main() {
    let data = MyStruct { a: 1, b: "hello".to_string() };
    let mut ctx = ();
    let mut ser = Serializer::new(&mut ctx);
    ser.serialize(&data);
    // This will print ( a: 1, b: "hello" )
    println!("serialized: {:}", ser.buf);
    let mut de = Deserializer::new(&ser.buf, &mut ctx);
    let res: MyStruct = de.deserialize();
    assert_eq!(res.a, 1);
}
```

## Default Implementation

By default, this crate provides an implementation for the `IRSerializer` and `IRDeserializer` traits.

The default implementation uses the `logos` crate to tokenize the input, and the `codespan_reporting` crate to report errors.

In the default implementation, some punctuation characters are treated as special characters:

- `{` and `}`: Increase and decrease the indentation level.
- `;`: Start a new line.

The default implementation also supports the `#[mtline]` attribute to start a new line after the field or variant.

```rust
use ir_serde::{IRSerde, Serializer, Deserializer, IRSerializer, IRDeserializer};

#[derive(IRSerde)]
struct MyStruct {
    #[punct(lbrace)]
    #[keyword(b)]#[punct(:)]
    b: String,
    #[punct(;)] // newline
    #[keyword(a)]#[punct(:)]
    #[mtline]   // use brace and semicolon to display the next container
    #[punct_after(rbrace)]
    a: Vec<u32>,
}

fn main() {
    let data = MyStruct { a: vec![1, 2], b: "hello".to_string() };
    let mut ctx = ();
    let mut ser = Serializer::new(&mut ctx);
    ser.serialize(&data);
    // This will print
    // {
    //   b: "hello";
    //   a: {
    //     1;
    //     2
    //   }
    // }
    println!("serialized:\n{:}", ser.buf);
    let mut de = Deserializer::new(&ser.buf, &mut ctx);
    let res: MyStruct = de.deserialize();
    assert_eq!(res.a, vec![1, 2]);
}
```