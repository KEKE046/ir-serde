use std::collections::HashSet;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use regex::Regex;
use syn::{
    spanned::Spanned, ConstParam, DeriveInput, Field, Fields, GenericParam, Generics, Ident,
    LifetimeParam, Meta, Path, TypeParam,
};

fn get_path(path: &Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

enum Token {
    Keyword(String),
    Punct(String),
}

impl Token {
    fn punct(s: String) -> Self {
        if s == "lbrace" {
            return Self::Punct("{".to_string());
        }
        if s == "rbrace" {
            return Self::Punct("}".to_string());
        }
        if s == "lbracket" || s == "lsquare" {
            return Self::Punct("[".to_string());
        }
        if s == "rbracket" || s == "rsquare" {
            return Self::Punct("]".to_string());
        }
        if s == "lparen" {
            return Self::Punct("(".to_string());
        }
        if s == "rparen" {
            return Self::Punct(")".to_string());
        }
        let mat = Regex::new(r#"[!@#$%^&*/\|;:',.<>\[\]=]"#)
            .unwrap()
            .is_match(&s);
        if !mat {
            panic!("Invalid punct: {}", s);
        }
        Self::Punct(s)
    }
    fn keyword(s: String) -> Self {
        let mat = Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]*").unwrap().is_match(&s);
        if !mat {
            panic!("Invalid keyword: {}", s);
        }
        Self::Keyword(s)
    }
    fn emit_ser(&self) -> TokenStream2 {
        match self {
            Self::Keyword(kw) => quote! {ser.serialize_keyword(#kw);},
            Self::Punct(p) => quote! {ser.serialize_punct(#p);},
        }
    }
    fn emit_der(&self) -> TokenStream2 {
        match self {
            Self::Keyword(kw) => quote! {der.expect_keyword(#kw)?;},
            Self::Punct(p) => quote! {der.expect_punct(#p)?;},
        }
    }
}

struct MetaInfo {
    seps: Vec<Token>,
    seps_after: Vec<Token>,
    keyword: Option<String>,
    mtline: bool,
}
fn extract_attrs(attrs: &[syn::Attribute]) -> MetaInfo {
    let mut keyword = None;
    let mut mtline = false;
    let mut seps = vec![];
    let mut seps_after = vec![];
    for attr in attrs {
        match &attr.meta {
            Meta::List(ml) if get_path(&ml.path) == "keyword" => {
                seps.push(Token::keyword(ml.tokens.to_string()));
                keyword = Some(ml.tokens.to_string());
            }
            Meta::List(ml) if get_path(&ml.path) == "punct" => {
                seps.push(Token::punct(ml.tokens.to_string()));
            }
            Meta::List(ml) if get_path(&ml.path) == "punct_after" => {
                seps_after.push(Token::punct(ml.tokens.to_string()));
            }
            Meta::Path(mp) if get_path(&mp) == "mtline" => {
                mtline = true;
            }
            _ => {}
        }
    }
    MetaInfo {
        seps,
        seps_after,
        keyword,
        mtline,
    }
}
fn extract_fields_pat(fields: &syn::Fields) -> TokenStream2 {
    match fields {
        syn::Fields::Named(n) => {
            let fields = n.named.iter().map(|f| {
                let name = &f.ident;
                quote! { #name }
            });
            quote! { { #(#fields),* } }
        }
        syn::Fields::Unnamed(u) => {
            let fields = u.unnamed.iter().enumerate().map(|(i, f)| {
                let name = syn::Ident::new(&format!("_{}", i), f.span());
                quote! { #name }
            });
            quote! { ( #(#fields),* ) }
        }
        syn::Fields::Unit => {
            quote! {}
        }
    }
}
fn impl_fields_slice<'r>(
    fields: impl 'r + IntoIterator<Item = &'r Field>,
) -> (TokenStream2, TokenStream2) {
    let mut ser = vec![];
    let mut der = vec![];
    for (i, field) in fields.into_iter().enumerate() {
        let MetaInfo {
            seps,
            mtline,
            seps_after,
            ..
        } = extract_attrs(&field.attrs);
        for sep in seps {
            ser.push(sep.emit_ser());
            der.push(sep.emit_der());
        }
        if mtline {
            ser.push(quote! {ser.set_container_puncts("{", ";", "}");});
            der.push(quote! {der.set_container_puncts("{", ";", "}");});
        }
        let name = field.ident.clone().unwrap_or_else(|| {
            let span = field.span();
            syn::Ident::new(&format!("_{}", i), span)
        });
        ser.push(quote! {#name.ir_serialize(ser);});
        der.push(quote! {let #name = ir_serde::deserialize_with(der)?;});
        if mtline {
            ser.push(quote! {ser.take_container_puncts();});
            der.push(quote! {der.take_container_puncts();});
        }
        for sep in seps_after {
            ser.push(sep.emit_ser());
            der.push(sep.emit_der());
        }
    }
    (quote! {#(#ser)*}, quote! {#(#der)*})
}
fn impl_fields(fields: &mut Fields) -> (TokenStream2, TokenStream2) {
    match fields {
        Fields::Named(named) => impl_fields_slice(named.named.iter()),
        Fields::Unnamed(unamed) => impl_fields_slice(unamed.unnamed.iter()),
        Fields::Unit => (quote! {}, quote! {}),
    }
}
fn extract_generic_names<'r>(
    params: impl 'r + IntoIterator<Item = &'r GenericParam>,
) -> TokenStream2 {
    params
        .into_iter()
        .map(|p| match p {
            GenericParam::Type(TypeParam { ident, .. }) => quote! {#ident,},
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => quote! {#lifetime,},
            GenericParam::Const(ConstParam { ident, .. }) => quote! {#ident,},
        })
        .collect()
}
fn extract_types<'r>(params: impl 'r + IntoIterator<Item = &'r GenericParam>) -> Vec<&'r Ident> {
    params
        .into_iter()
        .filter_map(|p| match p {
            GenericParam::Type(TypeParam { ident, .. }) => Some(ident),
            _ => None,
        })
        .collect()
}

#[proc_macro_derive(IRSerde, attributes(keyword, mtline, punct, context, punct_after))]
pub fn ir_serde_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    let DeriveInput {
        attrs,
        ident,
        generics,
        data,
        ..
    } = input;
    let Generics {
        params,
        where_clause,
        ..
    } = generics;
    let mut ctx = None;
    for attr in attrs {
        match attr.meta {
            syn::Meta::List(ml) if get_path(&ml.path) == "context" => {
                ctx = Some(ml.tokens);
            }
            _ => {}
        }
    }
    let extra_params;
    let extra_ir_serde_generic;
    match ctx {
        Some(ctx) => {
            extra_params = quote! {};
            extra_ir_serde_generic = quote! {<#ctx>};
        }
        None => {
            extra_params = quote! {__IR_SERDE_CTX__, };
            extra_ir_serde_generic = quote! {<__IR_SERDE_CTX__>};
        }
    }
    let param_names = extract_generic_names(&params);
    let types = extract_types(&params);
    let mut wheres = vec![];
    for t in types {
        wheres.push(quote! {#t: IRSerde #extra_ir_serde_generic, });
    }
    let other_wheres = where_clause.map(|w| w.predicates).unwrap_or_default();
    match data {
        syn::Data::Struct(s) => {
            let syn::DataStruct { mut fields, .. } = s;
            let fields_pat = extract_fields_pat(&fields);
            let (ser_impl, der_impl) = impl_fields(&mut fields);
            quote! {
                impl<#extra_params #params> IRSerde #extra_ir_serde_generic for #ident<#param_names> where #(#wheres)* #other_wheres {
                    fn ir_serialize(&self, ser: &mut dyn IRSerializer #extra_ir_serde_generic) {
                        let Self #fields_pat = self;
                        #ser_impl
                    }
                    fn ir_deserialize(der: &mut dyn IRDeserializer #extra_ir_serde_generic) -> Result<Self, String> {
                        #der_impl
                        Ok(Self #fields_pat)
                    }
                }
            }
            .into()
        }
        syn::Data::Enum(e) => {
            let syn::DataEnum { variants, .. } = e;
            let mut ser = vec![];
            let mut der = vec![];
            let mut used_keywords = HashSet::new();
            for mut v in variants {
                let name = &v.ident;
                let MetaInfo { keyword, .. } = extract_attrs(&mut v.attrs);
                let fields_pat = extract_fields_pat(&v.fields);
                let (ser_impl, der_impl) = impl_fields(&mut v.fields);
                let keyword = keyword.unwrap_or_else(|| name.to_string().to_lowercase());
                if !used_keywords.insert(keyword.clone()) {
                    panic!("Duplicate keyword: {}", keyword);
                }
                ser.push(quote! {
                    Self::#name #fields_pat => {
                        ser.serialize_keyword(#keyword);
                        #ser_impl
                    }
                });
                der.push(quote! {
                    #keyword => {
                        der.deserialize_keyword()?;
                        #der_impl
                        Ok(Self::#name #fields_pat)
                    }
                });
            }
            quote! {
                impl<#extra_params #params> IRSerde #extra_ir_serde_generic for #ident<#param_names> where #(#wheres)* #other_wheres {
                    fn ir_serialize(&self, ser: &mut dyn IRSerializer #extra_ir_serde_generic) {
                        match self { #(#ser),* }
                    }
                    fn ir_deserialize(der: &mut dyn IRDeserializer #extra_ir_serde_generic) -> Result<Self, String> {
                        let keyword = der.peek()?.1;
                        match keyword {
                            #(#der),*
                            _ => Err(format!("Unknown keyword {:?}", keyword)),
                        }
                    }
                }
            }
            .into()
        }
        syn::Data::Union(_) => todo!(),
    }
}
