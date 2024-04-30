use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::integer::{BooleanBlock, IntegerCiphertext, RadixCiphertext};
use crate::{integer, shortint};
use crate::shortint::{Ciphertext};
use crate::shortint::server_key::LookupTableOwned;

pub fn authenticate(
    server_key: shortint::ServerKey,
    probe: Vec<Vec<Ciphertext>>,
    luts: Vec<Vec<LookupTableOwned>>,
    threshold: usize,
    num_sum_blocks: usize
) -> (BooleanBlock, Duration) {
    let big_server_key = integer::ServerKey::new_radix_server_key_from_shortint(server_key.clone());
    let big_server_key_insde = big_server_key.clone();
    let start = Instant::now();
    //tfhe_functions::do_pbs(&keys, &buffer, &mut encrypted_probes, &encrypted_luts, &pbs_indices);
    let (tx, rx) = mpsc::sync_channel(probe.len());
    thread::spawn(move || {
        probe.into_par_iter().zip(luts).for_each(|(mut probes, luts)| {
            let cur_len = probes.len();
            probes.par_iter_mut().zip(luts).for_each(|(probe, lut)| {
                server_key.apply_lookup_table_assign(probe, &lut);
            });
            let mut as_radix = RadixCiphertext::from_blocks(probes);
            big_server_key_insde.extend_radix_with_trivial_zero_blocks_msb_assign(&mut as_radix, num_sum_blocks - cur_len);
            tx.send(as_radix).unwrap();
        });
    });
    let lut_values = rx.iter().collect_vec();
    let sum = big_server_key.unchecked_sum_ciphertexts_vec_parallelized(lut_values).unwrap();

    let res = big_server_key.unchecked_scalar_ge_parallelized(&sum, threshold as u64);

    let elapsed = start.elapsed();
    (res, elapsed)
}

pub fn authenticate_debug(
    decrypt_radix: Box<dyn Fn(&RadixCiphertext) -> u64>,
    server_key: shortint::ServerKey,
    probe: Vec<Vec<Ciphertext>>,
    luts: Vec<Vec<LookupTableOwned>>,
    threshold: usize,
    num_sum_blocks: usize
) -> (BooleanBlock, Duration) {
    let big_server_key = integer::ServerKey::new_radix_server_key_from_shortint(server_key.clone());
    let big_server_key_insde = big_server_key.clone();
    let start = Instant::now();
    //tfhe_functions::do_pbs(&keys, &buffer, &mut encrypted_probes, &encrypted_luts, &pbs_indices);
    let (tx, rx) = mpsc::sync_channel(probe.len());
    thread::spawn(move || {
        probe.into_par_iter().zip(luts).for_each(|(mut probes, luts)| {
            let cur_len = probes.len();
            probes.par_iter_mut().zip(luts).for_each(|(probe, lut)| {
                server_key.apply_lookup_table_assign(probe, &lut);
            });
            let mut as_radix = RadixCiphertext::from_blocks(probes);
            big_server_key_insde.extend_radix_with_trivial_zero_blocks_msb_assign(&mut as_radix, num_sum_blocks - cur_len);
            tx.send(as_radix).unwrap();
        });
    });

    let lut_values = rx.iter().collect_vec();
    println!("LUTs: {:?}", lut_values.iter().map(|l| decrypt_radix(l)).collect_vec());
    let sum = big_server_key.unchecked_sum_ciphertexts_vec_parallelized(lut_values).unwrap();
    println!("SUM: {:?}", decrypt_radix(&sum));

    let res = big_server_key.unchecked_scalar_ge_parallelized(&sum, threshold as u64);

    let elapsed = start.elapsed();
    (res, elapsed)
}