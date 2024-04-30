## About

This repository is a copy from the original <a href="https://github.com/zama-ai/tfhe-rs">tfhe-rs library</a>.

It has been adapted to implement a secure biometric authentication protocol based on the LLR similarity metric.

The methods for biometric authentication are located in tfhe/src/core_crypto/biometrics folder. This is due to visibility constraints in the TFHE library.

A main function is provided to execute a simple run. It can be executed with:
```cargo run --release --bin main --manifest-path ./main/Cargo.toml```

### References
Zama. 2022. TFHE-rs: A Pure Rust Implementation of the TFHE Scheme for Boolean and Integer Arithmetics Over Encrypted Data. https://github.com/zama-ai/tfhe-rs.
