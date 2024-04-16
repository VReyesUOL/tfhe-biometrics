use std::time::Duration;
use tfhe::core_crypto::my_bio::common;
use bio_aux;
use config::BLOCK_LENGTH;
use tfhe::core_crypto::my_bio::all_in_one_classic as classic;
use tfhe::core_crypto::my_bio::all_in_one_multibit as multibit;
use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS, PARAM_GPU_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_2_KS_PBS, PARAM_GPU_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_3_KS_PBS};
use tfhe::shortint::prelude::{PARAM_MESSAGE_2_CARRY_2_KS_PBS, PARAM_MESSAGE_3_CARRY_3_KS_PBS};

fn main() {
    println!("Hello, world!");
    const RUNS: usize = 100;
    let template_nums: [usize; RUNS] = [1690, 4144, 2686, 1079, 1975, 2277, 3193, 1814, 1942, 743, 3178, 2004, 4538, 4215, 1126, 2038, 332, 3977, 996, 1377, 153, 2912, 4632, 3400, 1104, 394, 1276, 2053, 2276, 382, 523, 1457, 4024, 4691, 2624, 4409, 2648, 3531, 3947, 3951, 1959, 1682, 4427, 2886, 2342, 1085, 3922, 4606, 3497, 94, 3578, 4053, 6, 1019, 3226, 1576, 4699, 3923, 919, 2918, 185, 1610, 494, 57, 1695, 167, 2378, 2225, 2686, 1956, 1188, 3374, 2293, 552, 3672, 1130, 4427, 1448, 1068, 4051, 560, 3487, 2262, 609, 3854, 577, 1353, 1503, 1190, 1586, 3295, 348, 643, 3765, 2190, 4381, 2389, 2515, 3875, 2826];
    let probe_nums: [usize; RUNS] = [1690, 4144, 2686, 1079, 1975, 2277, 3193, 1814, 1942, 743, 3178, 2004, 4538, 4215, 1126, 2038, 332, 3977, 996, 1377, 153, 2912, 4632, 3400, 1104, 394, 1276, 2053, 2276, 382, 523, 1457, 4024, 4691, 2624, 4409, 2648, 3531, 3947, 3951, 1959, 1682, 4427, 2886, 2342, 1085, 3922, 4606, 3497, 94, 1940, 1676, 2364, 434, 3348, 1059, 3436, 1923, 1529, 609, 925, 3205, 3138, 392, 2932, 1763, 3419, 794, 86, 1405, 1356, 2293, 3747, 2543, 1783, 4668, 4464, 2565, 3112, 3304, 4679, 4458, 4075, 1024, 1549, 3705, 507, 468, 3491, 2814, 3212, 3117, 3866, 4560, 264, 4561, 3694, 3717, 3620, 794];

    let results: Vec<(Duration, bool)> = template_nums.iter().enumerate().zip(probe_nums.iter()).map(|((idx, t), p)| {
        println!("Run {} with {} and {}", idx, *t, *p);

        let parameter_set = if BLOCK_LENGTH == 2 {
            PARAM_GPU_MULTI_BIT_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS
        } else {
            PARAM_GPU_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_3_KS_PBS
        };

        //Setup
        let (_params, stream, mut engine) = common::make_context(parameter_set.into());
        let ((params, glwe_secret_key), (ksk, bsk), (delta, total_modulus)) = tfhe::core_crypto::my_bio::tfhe_functions_multibit::make_keys_multibit(parameter_set, &stream, &mut engine);

        //Fetch probe and template
        let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(*p, *t);

        let _num_values = probe.len();
        //Create Lookup tables from template
        let (functions, threshold, _blocks_after_sum) = bio_aux::generate_functions(&template);
        let num_blocks = functions[0].len();

        //Flatten and repeat
        let (r_probe, f_functions) = flatten_and_repeat(&probe, functions);

        //Encrypt probes
        let encrypted_probes = common::encrypt_cuda_ciphertextlist(r_probe, &mut engine, &glwe_secret_key.as_lwe_secret_key(), total_modulus, delta, params.into(), &stream);

        //Make lookup tables
        let encrypted_luts = common::generate_cuda_lookup_tables(f_functions, &glwe_secret_key, params.into(), &mut engine, &stream);
        let (d, r) = multibit::authenticate(
            bsk,
            ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus,
            params.carry_modulus,
            threshold as usize,
            &stream,
        );

        //Decrypt
        let result = common::decrypt_boolean_block(&d, &glwe_secret_key.as_lwe_secret_key(), params.into(), delta, &stream);
        (r, result)
    }).collect();

    let mut total = 0_f64;
    let mut failed = Vec::new();
    for (i, (d, b)) in results.iter().enumerate() {
        total += d.as_secs_f64();
        if (template_nums[i] == probe_nums[i]) != *b {
            failed.push(i);
        }
    }

    println!("Average runtime: {}", total / RUNS as f64);
    println!("Fails: {:?}", failed);
    println!("Run times: {:?}", results.iter().map(|(d, _)| (d.as_secs_f64() * 1000f64) as u64 ).collect::<Vec<u64>>())
}
fn test_classic() {
    const RUNS: usize = 100;
    let template_nums: [usize; RUNS] = [1690, 4144, 2686, 1079, 1975, 2277, 3193, 1814, 1942, 743, 3178, 2004, 4538, 4215, 1126, 2038, 332, 3977, 996, 1377, 153, 2912, 4632, 3400, 1104, 394, 1276, 2053, 2276, 382, 523, 1457, 4024, 4691, 2624, 4409, 2648, 3531, 3947, 3951, 1959, 1682, 4427, 2886, 2342, 1085, 3922, 4606, 3497, 94, 3578, 4053, 6, 1019, 3226, 1576, 4699, 3923, 919, 2918, 185, 1610, 494, 57, 1695, 167, 2378, 2225, 2686, 1956, 1188, 3374, 2293, 552, 3672, 1130, 4427, 1448, 1068, 4051, 560, 3487, 2262, 609, 3854, 577, 1353, 1503, 1190, 1586, 3295, 348, 643, 3765, 2190, 4381, 2389, 2515, 3875, 2826];
    let probe_nums: [usize; RUNS] = [1690, 4144, 2686, 1079, 1975, 2277, 3193, 1814, 1942, 743, 3178, 2004, 4538, 4215, 1126, 2038, 332, 3977, 996, 1377, 153, 2912, 4632, 3400, 1104, 394, 1276, 2053, 2276, 382, 523, 1457, 4024, 4691, 2624, 4409, 2648, 3531, 3947, 3951, 1959, 1682, 4427, 2886, 2342, 1085, 3922, 4606, 3497, 94, 1940, 1676, 2364, 434, 3348, 1059, 3436, 1923, 1529, 609, 925, 3205, 3138, 392, 2932, 1763, 3419, 794, 86, 1405, 1356, 2293, 3747, 2543, 1783, 4668, 4464, 2565, 3112, 3304, 4679, 4458, 4075, 1024, 1549, 3705, 507, 468, 3491, 2814, 3212, 3117, 3866, 4560, 264, 4561, 3694, 3717, 3620, 794];

    let results: Vec<(Duration, bool)> = template_nums.iter().enumerate().zip(probe_nums.iter()).map(|((idx, t), p)| {
        println!("Run {} with {} and {}", idx, *t, *p);

        let parameter_set = if BLOCK_LENGTH == 2 {
            PARAM_MESSAGE_2_CARRY_2_KS_PBS
        } else {
            PARAM_MESSAGE_3_CARRY_3_KS_PBS
        };

        //Setup
        let (params,stream,mut engine) = common::make_context(parameter_set.into());
        let (_lwe_secret_key, glwe_secret_key, ksk, bsk, delta, total_modulus, params,) = tfhe::core_crypto::my_bio::tfhe_functions_classic::make_keys_no_server_key(params, &stream, &mut engine);

        //Fetch probe and template
        let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(*p, *t);

        let _num_values = probe.len();
        //Create Lookup tables from template
        let (functions, threshold, _blocks_after_sum) = bio_aux::generate_functions(&template);
        let num_blocks = functions[0].len();

        //Flatten and repeat
        let (r_probe, f_functions) = flatten_and_repeat(&probe, functions);

        //Encrypt probes
        let encrypted_probes = common::encrypt_cuda_ciphertextlist(r_probe, &mut engine, &glwe_secret_key.as_lwe_secret_key(), total_modulus, delta, params, &stream);

        //Make lookup tables
        let encrypted_luts = common::generate_cuda_lookup_tables(f_functions, &glwe_secret_key, params, &mut engine, &stream);
        let (d, r) = classic::authenticate(
            bsk,
            ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus(),
            params.carry_modulus(),
            threshold as usize,
            &stream,
        );


        //Decrypt
        let result = common::decrypt_boolean_block(&d, &glwe_secret_key.as_lwe_secret_key(), params, delta, &stream);
        (r, result)
    }).collect();

    let mut total = 0_f64;
    let mut failed = Vec::new();
    for (i, (d, b)) in results.iter().enumerate() {
        total += d.as_secs_f64();
        if (template_nums[i] == probe_nums[i]) != *b {
            failed.push(i);
        }
    }

    println!("Average runtime: {}", total / RUNS as f64);
    println!("Fails: {:?}", failed);
    println!("Run times: {:?}", results.iter().map(|(d, _)| (d.as_secs_f64() * 1000f64) as u64 ).collect::<Vec<u64>>())
}


fn flatten_and_repeat<F>(probe: &Vec<u8>, functions: Vec<Vec<F>>) -> (Vec<u64>, Vec<F>)
    where
    F: Fn(u64) -> u64
{
    let mut r_probe = Vec::new();
    let mut f_functions = Vec::new();
    functions.into_iter().zip(probe.iter()).for_each(|(fs, p) | {
        fs.into_iter().for_each(|f| {
            f_functions.push(f);
            r_probe.push(*p as u64)
        })
    });
    (r_probe, f_functions)
}