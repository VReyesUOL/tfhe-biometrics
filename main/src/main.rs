use std::time::Duration;
use tfhe::core_crypto::biometrics::common;
use itertools::Itertools;
use bio_aux;
use config::Config;
use tfhe::core_crypto::biometrics::cpu::{all_in_one_classic, all_in_one_multibit};
use tfhe::core_crypto::biometrics::gpu::all_in_one_classic as classic;
use tfhe::core_crypto::biometrics::gpu::all_in_one_multibit as multibit;
use tfhe::core_crypto::biometrics::cpu_gpu::{all_in_one_multibit as multibit_cpu_gpu, all_in_one_original};
use tfhe::integer;
use tfhe::integer::RadixClientKey;
use tfhe::shortint::engine::ShortintEngine;
use tfhe::shortint::MessageModulus;
use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS, PARAM_GPU_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_3_KS_PBS, PARAM_MULTI_BIT_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS, PARAM_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_3_KS_PBS};
use tfhe::shortint::prelude::{PARAM_MESSAGE_2_CARRY_2_KS_PBS, PARAM_MESSAGE_3_CARRY_3_KS_PBS};

fn main() {
    println!("Hello, world!");

    let debug = false;

    const RUNS: usize = 1;
    let config = config::FRGC;
    let template_nums = vec![1690, 4144, 2686, 1079, 1975, 2277, 3193, 1814, 1942, 743, 3178, 2004, 4538, 4215, 1126, 2038, 332, 3977, 996, 1377, 153, 2912, 4632, 3400, 1104, 394, 1276, 2053, 2276, 382, 523, 1457, 4024, 4691, 2624, 4409, 2648, 3531, 3947, 3951, 1959, 1682, 4427, 2886, 2342, 1085, 3922, 4606, 3497, 94, 3578, 4053, 6, 1019, 3226, 1576, 4699, 3923, 919, 2918, 185, 1610, 494, 57, 1695, 167, 2378, 2225, 2686, 1956, 1188, 3374, 2293, 552, 3672, 1130, 4427, 1448, 1068, 4051, 560, 3487, 2262, 609, 3854, 577, 1353, 1503, 1190, 1586, 3295, 348, 643, 3765, 2190, 4381, 2389, 2515, 3875, 2826];
    let probe_nums = vec![1690, 4144, 2686, 1079, 1975, 2277, 3193, 1814, 1942, 743, 3178, 2004, 4538, 4215, 1126, 2038, 332, 3977, 996, 1377, 153, 2912, 4632, 3400, 1104, 394, 1276, 2053, 2276, 382, 523, 1457, 4024, 4691, 2624, 4409, 2648, 3531, 3947, 3951, 1959, 1682, 4427, 2886, 2342, 1085, 3922, 4606, 3497, 94, 1940, 1676, 2364, 434, 3348, 1059, 3436, 1923, 1529, 609, 925, 3205, 3138, 392, 2932, 1763, 3419, 794, 86, 1405, 1356, 2293, 3747, 2543, 1783, 4668, 4464, 2565, 3112, 3304, 4679, 4458, 4075, 1024, 1549, 3705, 507, 468, 3491, 2814, 3212, 3117, 3866, 4560, 264, 4561, 3694, 3717, 3620, 794];
    let mut vec_res_classic_cpu_original = Vec::with_capacity(RUNS);
    let mut vec_res_multibit_cpu_original = Vec::with_capacity(RUNS);
    let mut vec_res_classic_cpu_gpu_original = Vec::with_capacity(RUNS);
    let mut vec_res_classic_gpu = Vec::with_capacity(RUNS);

    println!("Safe measures: ");
    (0..RUNS).for_each(|idx| {
        let template = template_nums[idx];
        let probe  = probe_nums[idx];
        vec_res_classic_cpu_original.push(test_classic_cpu_original(idx, template, probe, &config, debug));
        vec_res_multibit_cpu_original.push(test_multibit_cpu_original(idx, template, probe, &config, debug));
        vec_res_classic_cpu_gpu_original.push(test_original(idx, template, probe, &config, debug));
        vec_res_classic_gpu.push(test_classic_gpu(idx, template, probe, &config, debug));
    });
    eval_measurements("classic_cpu_original", vec_res_classic_cpu_original);
    eval_measurements("multibit_cpu_original", vec_res_multibit_cpu_original);
    eval_measurements("classic_cpu_gpu_original", vec_res_classic_cpu_gpu_original);
    eval_measurements("classic_gpu", vec_res_classic_gpu);

    if config.block_length == 2 {
        println!("Unsafe measures: ");
        let mut vec_res_multibit_gpu_cpu = Vec::with_capacity(RUNS);
        let mut vec_res_multibit_gpu = Vec::with_capacity(RUNS);
        (0..RUNS).for_each(|idx| {
            let template = template_nums[idx];
            let probe  = probe_nums[idx];
            vec_res_multibit_gpu_cpu.push(test_multibit_gpu_cpu(idx, template, probe, &config, debug));
            vec_res_multibit_gpu.push(test_multibit_gpu(idx, template, probe, &config, debug));
        });
        eval_measurements("multibit_gpu_cpu", vec_res_multibit_gpu_cpu);
        eval_measurements("multibit_gpu", vec_res_multibit_gpu);
    }
}

pub fn eval_measurements(name: &str, measurements: Vec<(Duration, bool)>){
    let mut total = 0_f64;
    let count = measurements.len();
    for (d, _b) in measurements.iter() {
        total += d.as_secs_f64();
    }
    println!("Results: {}", name);
    println!("Average runtime: {}", total / count as f64);
    println!("Auth: {:?}", measurements.iter().map(|(_, b)| *b).collect_vec());
    println!("Run times: {:?}", measurements.iter().map(|(d, _)| (d.as_secs_f64() * 1000f64) as u64 ).collect::<Vec<u64>>())
}

pub fn test_classic_cpu_original(test_idx: usize, template_idx: usize, probe_idx: usize, config: &Config, debug: bool) -> (Duration, bool) {
    println!("classic_cpu_original {} with {} and {}", test_idx, template_idx, probe_idx);

    let parameter_set = if config.block_length == 2 {
        PARAM_MESSAGE_2_CARRY_2_KS_PBS
    } else {
        PARAM_MESSAGE_3_CARRY_3_KS_PBS
    };

    let mut engine = ShortintEngine::new();

    //Setup
    let (client_key,server_key) = tfhe::core_crypto::biometrics::cpu::tfhe_functions_classic::make_keys_classic(parameter_set, &mut engine);

    //Fetch probe and template
    let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(probe_idx, template_idx, &config);

    //Create Lookup tables from template
    let (functions, threshold) = bio_aux::generate_functions_stop_early(&template, &config);

    if debug {
        println!("Clear LUTs: {:?}",
            functions.iter().zip(&probe).map(|(fs, p)| {
                fs.iter().map(|f| {
                    f(*p as u64)
                }).collect_vec()
            }).collect_vec()
        );
    }

    //Flatten and repeat
    let r_probe = repeat_probes_to_match_functions(&probe, &functions);
    if debug {
        println!("Probes decomp: {:?}", r_probe);
    }

    //Encrypt probes
    let encrypted_probes = r_probe.iter().map(|ps| {
        ps.iter().map(|p| {
            client_key.encrypt_with_message_modulus(*p, MessageModulus(1 << (2 * config.block_length)))
        }).collect_vec()
    }).collect_vec();

    if debug {
        println!("Probes decrypted: {:?}",
             encrypted_probes.iter().map(|ps| {
                 ps.iter().map(|p| {
                     client_key.decrypt_message_and_carry(&p)
                 }).collect_vec()
             }).collect_vec()
        );
    }

    //Make lookup tables
    let encrypted_luts = common::generate_lookup_tables_individual(functions, &client_key, parameter_set.into(), &mut engine);
    let (d, r) = if debug {
        let key_clone = client_key.clone();
        let big_client_key = integer::ClientKey::from_raw_parts(client_key.clone());
        let big_client_key = RadixClientKey::from((big_client_key, config.num_blocks_sum));
        all_in_one_classic::authenticate(
            //Box::new(move |v| key_clone.decrypt(v)),
            //Box::new(move |v| big_client_key.decrypt::<u64>(v)),
            server_key,
            encrypted_probes,
            encrypted_luts,
            threshold,
            config.num_blocks_sum,
        )
    } else {
        all_in_one_classic::authenticate(
            server_key,
            encrypted_probes,
            encrypted_luts,
            threshold,
            config.num_blocks_sum,
        )
    };

    //Decrypt
    let result = common::decrypt_boolean_block_client_key(&d, &client_key);
    (r, result)
}


pub fn test_multibit_cpu_original(test_idx: usize, template_idx: usize, probe_idx: usize, config: &Config, debug: bool) -> (Duration, bool) {
    println!("multibit_cpu_original {} with {} and {}", test_idx, template_idx, probe_idx);

    let parameter_set = if config.block_length == 2 {
        PARAM_MULTI_BIT_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS
    } else {
        PARAM_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_3_KS_PBS
    };

    let thread_count_bs = 4;

    let mut engine = ShortintEngine::new();

    //Setup
    let (client_key,server_key) = tfhe::core_crypto::biometrics::cpu::tfhe_functions_multibit::make_keys_multibit(parameter_set, thread_count_bs, &mut engine);

    //Fetch probe and template
    let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(probe_idx, template_idx, &config);

    //Create Lookup tables from template
    let (functions, threshold) = bio_aux::generate_functions_stop_early(&template, &config);

    //Flatten and repeat
    let r_probe = repeat_probes_to_match_functions(&probe, &functions);

    //Encrypt probes
    let encrypted_probes = r_probe.iter().map(|ps| {
        ps.iter().map(|p| {
            client_key.encrypt_with_message_modulus(*p, MessageModulus(1 << (2 * config.block_length)))
        }).collect_vec()
    }).collect_vec();

    //Make lookup tables
    let encrypted_luts = common::generate_lookup_tables_individual(functions, &client_key, parameter_set.into(), &mut engine);
    let (d, r) = if debug {
        let big_client_key = integer::ClientKey::from_raw_parts(client_key.clone());
        let big_client_key = RadixClientKey::from((big_client_key, config.num_blocks_sum));
        all_in_one_multibit::authenticate_debug(
            Box::new(move |v| big_client_key.decrypt::<u64>(v)),
            server_key,
            encrypted_probes,
            encrypted_luts,
            threshold,
            config.num_blocks_sum,
        )
    } else {
        all_in_one_multibit::authenticate(
            server_key,
            encrypted_probes,
            encrypted_luts,
            threshold,
            config.num_blocks_sum,
        )
    };

    //Decrypt
    let result = common::decrypt_boolean_block_client_key(&d, &client_key);
    (r, result)
}



pub fn test_original(test_idx: usize, template_idx: usize, probe_idx: usize, config: &Config, debug: bool) -> (Duration, bool) {
    println!("original {} with {} and {}", test_idx, template_idx, probe_idx);

    let parameter_set = if config.block_length == 2 {
        PARAM_MESSAGE_2_CARRY_2_KS_PBS
    } else {
        PARAM_MESSAGE_3_CARRY_3_KS_PBS
    };

    //Setup
    let (stream, mut engine) = common::make_context_gpu();
    let (
        (params, client_key),
        (server_key, cuda_server_key)
    ) = tfhe::core_crypto::biometrics::cpu_gpu::tfhe_functions_original::make_keys_original(parameter_set, &stream, &mut engine);

    //Fetch probe and template
    let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(probe_idx, template_idx, &config);

    //Create Lookup tables from template
    let (functions, threshold) = bio_aux::generate_functions_const_length(&template, &config);

    //Flatten and repeat
    let r_probe = repeat_probes_to_match_functions(&probe, &functions);

    //Encrypt probes
    let encrypted_probes = r_probe.iter().map(|ps| {
        ps.iter().map(|p| {
            client_key.encrypt_with_message_modulus(*p, MessageModulus(1 << (2 * config.block_length)))
        }).collect_vec()
    }).collect_vec();

    //Make lookup tables
    let encrypted_luts = common::generate_lookup_tables_individual(functions, &client_key, params.into(), &mut engine);

    let (d, r) = if debug {
        let big_client_key = integer::ClientKey::from_raw_parts(client_key.clone());
        let big_client_key = RadixClientKey::from((big_client_key, config.num_blocks_sum));
        all_in_one_original::authenticate_debug(
            Box::new(move |v| big_client_key.decrypt::<u64>(v)),
            server_key,
            cuda_server_key,
            encrypted_probes,
            encrypted_luts,
            threshold,
            config.num_blocks_sum,
            &stream,
        )
    } else {
        all_in_one_original::authenticate(
            server_key,
            cuda_server_key,
            encrypted_probes,
            encrypted_luts,
            threshold,
            config.num_blocks_sum,
            &stream,
        )
    };

    //Decrypt
    let result = common::decrypt_cuda_boolean_block_client_key(&d, &client_key, &stream);
    (r, result)
}


pub fn test_multibit_gpu_cpu(test_idx: usize, template_idx: usize, probe_idx: usize, config: &Config, debug: bool) -> (Duration, bool) {
    println!("multibit_gpu_cpu {} with {} and {}", test_idx, template_idx, probe_idx);
    let thread_count_bs = 7;
    let thread_count_ks = 10;

    let cuda_param_set = if config.block_length == 2 {
        PARAM_GPU_MULTI_BIT_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS
    } else {
        PARAM_GPU_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_3_KS_PBS
    };

    //Setup
    let (stream, mut engine) = common::make_context_gpu();
    let (
        (params, glwe_secret_key),
        (ksk, bsk),
        (cuda_ksk, cuda_bsk),
        (delta, total_modulus)
    ) = tfhe::core_crypto::biometrics::cpu_gpu::tfhe_functions_multibit::make_keys_multibit(cuda_param_set, &stream, &mut engine);

    //Fetch probe and template
    let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(probe_idx, template_idx, &config);

    //Create Lookup tables from template
    let (functions, threshold) = bio_aux::generate_functions_const_length(&template, &config);
    let num_blocks = functions[0].len();

    //Flatten and repeat
    let (r_probe, f_functions) = flatten_and_repeat(&probe, functions);

    //Encrypt probes
    let encrypted_probes = common::encrypt_ciphertextlist(r_probe, &mut engine, &glwe_secret_key.as_lwe_secret_key(), total_modulus, delta, params.into());

    //Make lookup tables
    let encrypted_luts = common::generate_lookup_tables(f_functions, &glwe_secret_key, params.into(), &mut engine);

    let (d, r) = if debug {
        let key_clone = glwe_secret_key.clone();
        multibit_cpu_gpu::authenticate_debug(
            Box::new(move |v| {
                common::decrypt(&v, &key_clone.as_lwe_secret_key(), delta)
            }),
            bsk,
            ksk,
            cuda_bsk,
            cuda_ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus,
            params.carry_modulus,
            threshold,
            thread_count_ks,
            thread_count_bs,
            &stream,
        )
    } else {
        multibit_cpu_gpu::authenticate(
            bsk,
            ksk,
            cuda_bsk,
            cuda_ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus,
            params.carry_modulus,
            threshold,
            thread_count_ks,
            thread_count_bs,
            &stream,
        )
    };

    //Decrypt
    let result = common::decrypt_boolean_block(&d, &glwe_secret_key.as_lwe_secret_key(), params.into(), delta, &stream);
    (r, result)
}

pub fn test_multibit_gpu(test_idx: usize, template_idx: usize, probe_idx: usize, config: &Config, debug: bool) -> (Duration, bool) {
    println!("multibit_gpu {} with {} and {}", test_idx, template_idx, probe_idx);

    let parameter_set = if config.block_length == 2 {
        PARAM_GPU_MULTI_BIT_MESSAGE_2_CARRY_2_GROUP_3_KS_PBS
    } else {
        PARAM_GPU_MULTI_BIT_MESSAGE_3_CARRY_3_GROUP_3_KS_PBS
    };

    //Setup
    let (stream, mut engine) = common::make_context_gpu();
    let ((params, glwe_secret_key), (ksk, bsk), (delta, total_modulus)) = tfhe::core_crypto::biometrics::gpu::tfhe_functions_multibit::make_keys_multibit(parameter_set, &stream, &mut engine);

    //Fetch probe and template
    let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(probe_idx, template_idx, &config);

    //Create Lookup tables from template
    let (functions, threshold) = bio_aux::generate_functions_const_length(&template, &config);
    let num_blocks = functions[0].len();

    //Flatten and repeat
    let (r_probe, f_functions) = flatten_and_repeat(&probe, functions);

    //Encrypt probes
    let encrypted_probes = common::encrypt_cuda_ciphertextlist(r_probe, &mut engine, &glwe_secret_key.as_lwe_secret_key(), total_modulus, delta, params.into(), &stream);

    //Make lookup tables
    let encrypted_luts = common::generate_cuda_lookup_tables(f_functions, &glwe_secret_key, params.into(), &mut engine, &stream);

    let (d, r) = if debug {
        let key_clone = glwe_secret_key.clone();
        multibit::authenticate_debug(
            Box::new(move |v| {
                common::decrypt(&v, &key_clone.as_lwe_secret_key(), delta)
            }),
            bsk,
            ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus,
            params.carry_modulus,
            threshold,
            &stream,
        )
    } else {
        multibit::authenticate(
            bsk,
            ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus,
            params.carry_modulus,
            threshold,
            &stream,
        )
    };

    //Decrypt
    let result = common::decrypt_boolean_block(&d, &glwe_secret_key.as_lwe_secret_key(), params.into(), delta, &stream);
    (r, result)
}

fn test_classic_gpu(test_idx: usize, template_idx: usize, probe_idx: usize, config: &Config, debug: bool) -> (Duration, bool) {
    println!("classic_gpu {} with {} and {}", test_idx, template_idx, probe_idx);

    let parameter_set = if config.block_length == 2 {
        PARAM_MESSAGE_2_CARRY_2_KS_PBS
    } else {
        PARAM_MESSAGE_3_CARRY_3_KS_PBS
    };

    //Setup
    let (stream,mut engine) = common::make_context_gpu();
    let (_lwe_secret_key, glwe_secret_key, ksk, bsk, delta, total_modulus, params,) = tfhe::core_crypto::biometrics::gpu::tfhe_functions_classic::make_keys_no_server_key(parameter_set.into(), &stream, &mut engine);

    //Fetch probe and template
    let (probe, template) = bio_aux::io::probe_and_template_generation_radix_prepare(probe_idx, template_idx, &config);

    //Create Lookup tables from template
    let (functions, threshold) = bio_aux::generate_functions_const_length(&template, &config);
    let num_blocks = functions[0].len();

    //Flatten and repeat
    let (r_probe, f_functions) = flatten_and_repeat(&probe, functions);

    //Encrypt probes
    let encrypted_probes = common::encrypt_cuda_ciphertextlist(r_probe, &mut engine, &glwe_secret_key.as_lwe_secret_key(), total_modulus, delta, params, &stream);

    //Make lookup tables
    let encrypted_luts = common::generate_cuda_lookup_tables(f_functions, &glwe_secret_key, params, &mut engine, &stream);
    let (d, r) = if debug {
        let key_clone = glwe_secret_key.clone();
        classic::authenticate_debug(
            Box::new(move |v| {
                common::decrypt(&v, &key_clone.as_lwe_secret_key(), delta)
            }),
            bsk,
            ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus(),
            params.carry_modulus(),
            threshold,
            &stream,
        )
    } else {
        classic::authenticate(
            bsk,
            ksk,
            encrypted_probes,
            encrypted_luts,
            num_blocks,
            params.message_modulus(),
            params.carry_modulus(),
            threshold,
            &stream,
        )
    };

    //Decrypt
    let result = common::decrypt_boolean_block(&d, &glwe_secret_key.as_lwe_secret_key(), params, delta, &stream);
    (r, result)
}

fn repeat_probes_to_match_functions<F>(probe: &Vec<u8>, functions: &Vec<Vec<F>>) -> Vec<Vec<u64>>
    where
        F: Fn(u64) -> u64
{
    let mut r_probe = Vec::new();
    functions.iter().zip(probe.iter()).for_each(|(fs, p) | {
        let mut r_probe_inner = Vec::new();
        fs.iter().for_each(|_| {
            r_probe_inner.push(*p as u64)
        });
        r_probe.push(r_probe_inner);
    });
    r_probe
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
