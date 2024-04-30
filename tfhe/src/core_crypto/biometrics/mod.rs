
pub mod gpu;
pub mod common;
pub mod cpu_gpu;
pub mod cpu;

pub fn this_name_is_very_expressive(){
    /*
    let (params,stream,mut engine) = biometrics::make_context(PARAM_MESSAGE_3_CARRY_3_KS_PBS.into());
    let keys = biometrics::make_keys(params, stream, &mut engine);

    let mut ct_in = biometrics::encrypt_cuda_ciphertextlist(
        (0..NUM_CTS as u64).collect(),
        &mut engine,
        &keys,
    );

    let mut ct_out = biometrics::make_cuda_lweciphertextlist(
        NUM_CTS,
        keys.lwe_secret_key.lwe_dimension().to_lwe_size(),
        keys.parameters.ciphertext_modulus(),
        &keys.stream,
    );

    let ks_indices = biometrics::make_indices(NUM_CTS as u64, 2, &keys);
    biometrics::do_keyswitch(
        &keys,
        &ct_in,
        &mut ct_out,
        &ks_indices,
    );
    let fs = [
        |_v| 0,
        |_v| 1,
        |v| v + 1,
        |v| v - 1,
        |v| v * 2,
        |v| v * 2 -1,
        |v| v + 2,
        |v| v - 2,
        |v| v,
        |v| v + v,
    ];

    let cuda_lut_list= biometrics::generate_cuda_lookup_tables(
        fs.to_vec(),
        &keys,
        &mut engine,
    );

    let pbs_indices = biometrics::make_indices(NUM_CTS as u64, 3, &keys);
    biometrics::do_pbs(&keys, &ct_out, &mut ct_in, &cuda_lut_list, &pbs_indices);

    let values = biometrics::decrypt_cuda_ciphertextlist(&ct_in, &keys);
    println!("{:?}", values);
     */
}
