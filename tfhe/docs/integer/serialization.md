# Serialization / Deserialization

As explained in the introduction, some types (`Serverkey`, `Ciphertext`) are meant to be shared
with the server that does the computations.

The easiest way to send these data to a server is to use the serialization and deserialization features.
concrete-integer uses the serde framework, serde's Serialize and Deserialize are implemented.

To be able to serialize our data, we need to pick a [data format], for our use case,
[bincode] is a good choice, mainly because it is binary format.


```toml
# Cargo.toml

[dependencies]
# ...
bincode = "1.3.3"
```


```rust
// main.rs

use bincode;

use std::io::Cursor;
use tfhe::integer::{gen_keys_radix, ServerKey, RadixCiphertextBig};
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // We generate a set of client/server keys, using the default parameters:
    let num_block = 4;
    let (client_key, server_key) = gen_keys_radix(&PARAM_MESSAGE_2_CARRY_2, num_block);

    let msg1 = 201;
    let msg2 = 12;

    // message_modulus^vec_length
    let modulus = client_key.parameters().message_modulus.0.pow(num_block as u32) as u64;
    
    let ct_1 = client_key.encrypt(msg1);
    let ct_2 = client_key.encrypt(msg2);

    let mut serialized_data = Vec::new();
    bincode::serialize_into(&mut serialized_data, &server_key)?;
    bincode::serialize_into(&mut serialized_data, &ct_1)?;
    bincode::serialize_into(&mut serialized_data, &ct_2)?;

    // Simulate sending serialized data to a server and getting
    // back the serialized result
    let serialized_result = server_function(&serialized_data)?;
    let result: RadixCiphertextBig = bincode::deserialize(&serialized_result)?;

    let output: u64 = client_key.decrypt(&result);
    assert_eq!(output, (msg1 + msg2) % modulus);
    Ok(())
}


fn server_function(serialized_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut serialized_data = Cursor::new(serialized_data);
    let server_key: ServerKey = bincode::deserialize_from(&mut serialized_data)?;
    let ct_1: RadixCiphertextBig = bincode::deserialize_from(&mut serialized_data)?;
    let ct_2: RadixCiphertextBig = bincode::deserialize_from(&mut serialized_data)?;

    let result = server_key.unchecked_add(&ct_1, &ct_2);

    let serialized_result = bincode::serialize(&result)?;

    Ok(serialized_result)
}
```

[serde]: https://crates.io/crates/serde
[data format]: https://serde.rs/#data-formats
[bincode]: https://crates.io/crates/bincode 