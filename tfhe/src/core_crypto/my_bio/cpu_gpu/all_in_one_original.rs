use std::time::{Duration, Instant};
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::core_crypto::gpu::{CudaStream};
use crate::integer::gpu::ciphertext::boolean_value::CudaBooleanBlock;
use crate::integer::gpu::ciphertext::{ CudaUnsignedRadixCiphertext};
use crate::integer::gpu::{CudaServerKey};
use crate::integer::{IntegerCiphertext, RadixCiphertext};
use crate::{integer, shortint};
use crate::shortint::{Ciphertext};
use crate::shortint::server_key::LookupTableOwned;

pub fn authenticate(
    server_key: shortint::ServerKey,
    cuda_server_key: CudaServerKey,
    probe: Vec<Vec<Ciphertext>>,
    luts: Vec<Vec<LookupTableOwned>>,
    threshold: usize,
    num_sum_blocks: usize,
    stream: &CudaStream,
) -> (CudaBooleanBlock, Duration) {
    let big_server_key = integer::ServerKey::new_radix_server_key_from_shortint(server_key.clone());

    let start = Instant::now();
    //tfhe_functions::do_pbs(&keys, &buffer, &mut encrypted_probes, &encrypted_luts, &pbs_indices);
    let mut lut_values = Vec::with_capacity(luts.len());
    probe.into_par_iter().zip(luts).map(|(mut probes, luts)| {
        let cur_len = probes.len();
        probes.par_iter_mut().zip(luts).for_each(|(probe, lut)| {
            server_key.apply_lookup_table_assign(probe, &lut);
        });
        let mut as_radix = RadixCiphertext::from_blocks(probes);
        big_server_key.extend_radix_with_trivial_zero_blocks_msb_assign(&mut as_radix, num_sum_blocks - cur_len);
        as_radix
    }).collect_into_vec(&mut lut_values);
    let values_cuda = lut_values.into_iter().map(|r| {
        CudaUnsignedRadixCiphertext::from_radix_ciphertext(&r, &stream)
    }).collect_vec();

    let sum = cuda_server_key.unchecked_sum_ciphertexts(
        &values_cuda, &stream
    ).unwrap();

    let res = cuda_server_key.unchecked_scalar_ge(&sum, threshold as u64, &stream);

    let elapsed = start.elapsed();
    (res, elapsed)
}

pub fn authenticate_debug(
    decrypt_radix: Box<dyn Fn(&RadixCiphertext) -> u64>,
    server_key: shortint::ServerKey,
    cuda_server_key: CudaServerKey,
    probe: Vec<Vec<Ciphertext>>,
    luts: Vec<Vec<LookupTableOwned>>,
    threshold: usize,
    num_sum_blocks: usize,
    stream: &CudaStream,
) -> (CudaBooleanBlock, Duration) {
    let big_server_key = integer::ServerKey::new_radix_server_key_from_shortint(server_key.clone());

    let start = Instant::now();
    //tfhe_functions::do_pbs(&keys, &buffer, &mut encrypted_probes, &encrypted_luts, &pbs_indices);
    let mut lut_values = Vec::with_capacity(luts.len());
    probe.into_par_iter().zip(luts).map(|(mut probes, luts)| {
        let cur_len = probes.len();
        probes.par_iter_mut().zip(luts).for_each(|(probe, lut)| {
            server_key.apply_lookup_table_assign(probe, &lut);
        });
        let mut as_radix = RadixCiphertext::from_blocks(probes);
        //big_server_key.extend_radix_with_trivial_zero_blocks_msb_assign(&mut as_radix, num_sum_blocks - cur_len);
        as_radix
    }).collect_into_vec(&mut lut_values);
    let values_cuda = lut_values.into_iter().map(|r| {
        CudaUnsignedRadixCiphertext::from_radix_ciphertext(&r, &stream)
    }).collect_vec();

    println!("LUTs: {:?}",
        values_cuda.iter().map(|v| {
            decrypt_radix(&v.to_radix_ciphertext(&stream))
        }).collect_vec()
    );

    let sum = cuda_server_key.unchecked_sum_ciphertexts(
        &values_cuda, &stream
    ).unwrap();

    println!("SUM: {}", decrypt_radix(&sum.to_radix_ciphertext(&stream)));

    let res = cuda_server_key.unchecked_scalar_ge(&sum, threshold as u64, &stream);

    let elapsed = start.elapsed();
    (res, elapsed)
}