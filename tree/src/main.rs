use tree::morton::{less_than, Key};

fn main() {
    let a = Key(0b1010001);
    let b = Key(0b1010010);
    let result = less_than(&a, &b);
    println!("result {:?}", result);
}