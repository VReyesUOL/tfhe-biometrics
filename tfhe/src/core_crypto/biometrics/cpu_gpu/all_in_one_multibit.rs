use std::time::{Duration, Instant};
use itertools::Itertools;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelBridge;
use crate::core_crypto::algorithms::par_keyswitch_lwe_ciphertext_with_thread_count;
use crate::core_crypto::gpu::{CudaStream};
use crate::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use crate::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use crate::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::biometrics::common;
use crate::core_crypto::prelude::{ContiguousEntityContainer, ContiguousEntityContainerMut, FourierLweMultiBitBootstrapKeyOwned, GlweCiphertextListOwned, LweCiphertextCount, LweCiphertextListOwned, LweCiphertextOwned, LweKeyswitchKeyOwned, multi_bit_programmable_bootstrap_lwe_ciphertext, ThreadCount};
use crate::integer::block_decomposition::BlockDecomposer;
use crate::integer::gpu::ciphertext::boolean_value::CudaBooleanBlock;
use crate::integer::gpu::ciphertext::{CudaRadixCiphertext};
use crate::integer::gpu::ciphertext::info::{CudaBlockInfo, CudaRadixCiphertextInfo};
use crate::integer::gpu::ComparisonType;
use crate::shortint::{CarryModulus, CiphertextModulus, MessageModulus, PBSOrder};
use crate::shortint::ciphertext::{Degree, NoiseLevel};

pub fn authenticate(
    bsk: FourierLweMultiBitBootstrapKeyOwned,
    ksk: LweKeyswitchKeyOwned<u64>,
    cuda_bsk: CudaLweMultiBitBootstrapKey,
    cuda_ksk: CudaLweKeyswitchKey<u64>,
    mut probe: LweCiphertextListOwned<u64>,
    luts: GlweCiphertextListOwned<u64>,
    num_blocks: usize,
    message_modulus: MessageModulus,
    carry_modulus: CarryModulus,
    threshold: usize,
    thread_count_ks_u: usize,
    thread_count_bs_u: usize,
    stream: &CudaStream,
) -> (CudaBooleanBlock, Duration) {
    let thread_count_ks = ThreadCount(thread_count_ks_u);
    let thread_count_bs = ThreadCount(thread_count_bs_u);

    let flat_length = probe.lwe_ciphertext_count().0;
    let lwe_size = probe.lwe_size().to_lwe_dimension().to_lwe_size();
    let ct_count = flat_length / num_blocks;

    //Buffer
    let mut buffer = common::make_lweciphertextlist(
        flat_length,
        ksk.output_lwe_size(),
        probe.ciphertext_modulus(),
    );

    let mut buffer_cuda = CudaVec::new(lwe_size.0 * flat_length, &stream);
    let mut sum = CudaVec::new(lwe_size.0 * num_blocks, &stream);

    let mut scalar_blocks =
        BlockDecomposer::with_early_stop_at_zero(threshold as u64, message_modulus.0.ilog2())
            .iter_as::<u64>()
            .collect::<Vec<_>>();

    scalar_blocks.truncate(num_blocks);
    let d_scalar_blocks: CudaVec<u64>;
    unsafe {
        d_scalar_blocks = CudaVec::from_cpu_async(&scalar_blocks, &stream);
    }

    let result = CudaLweCiphertextList::new(
        probe.lwe_size().to_lwe_dimension(),
        LweCiphertextCount(1),
        CiphertextModulus::new_native(),
        &stream,
    );

    let ct_info = CudaRadixCiphertextInfo {
        blocks: vec![CudaBlockInfo {
            degree: Degree::new(0),
            message_modulus: message_modulus,
            carry_modulus: carry_modulus,
            pbs_order: PBSOrder::KeyswitchBootstrap,
            noise_level: NoiseLevel::NOMINAL,
        }]
    };

    let mut result = CudaBooleanBlock::from_cuda_radix_ciphertext(CudaRadixCiphertext::new(result, ct_info));

    //Apply pbs
    //tfhe_functions::do_keyswitch(&keys, &encrypted_probes, &mut buffer, &ks_indices);

    buffer.iter_mut().zip(probe.iter()).par_bridge()
        .for_each(|(mut b, p)| {
        par_keyswitch_lwe_ciphertext_with_thread_count(
            &ksk,
            &p,
            &mut b,
            thread_count_ks,
        );
    });

    let start = Instant::now();
    //tfhe_functions::do_pbs(&keys, &buffer, &mut encrypted_probes, &encrypted_luts, &pbs_indices);
    probe.iter_mut().zip(buffer.iter()).zip(luts.iter()).par_bridge()
        .for_each(|((mut p, b), lut)| {
        multi_bit_programmable_bootstrap_lwe_ciphertext(
            &b,
            &mut p,
            &lut,
            &bsk,
            thread_count_bs,
        );
    });


    unsafe {
        buffer_cuda.copy_from_cpu_async(
            &probe.as_ref(),
            &stream,
        );

        //Do sum
        //let mut sum = tfhe_functions::do_sum(&keys, num_blocks, num_values, &mut encrypted_probes);
        sum.copy_src_range_gpu_to_gpu_async(
            ..lwe_size.0 * num_blocks,
            &buffer_cuda,
            &stream
        );

        stream.unchecked_sum_ciphertexts_integer_radix_multibit_kb_assign_async(
            &mut sum,
            &mut buffer_cuda,
            &cuda_bsk.d_vec,
            &cuda_ksk.d_vec,
            message_modulus,
            carry_modulus,
            cuda_bsk.glwe_dimension,
            cuda_bsk.polynomial_size,
            cuda_ksk.output_key_lwe_size().to_lwe_dimension(),
            cuda_ksk.decomposition_level_count(),
            cuda_ksk.decomposition_base_log(),
            cuda_bsk.decomp_level_count,
            cuda_bsk.decomp_base_log,
            cuda_bsk.grouping_factor,
            num_blocks as u32,
            ct_count as u32,
        );

        //let block = tfhe_functions::do_comparison(&keys, &mut sum, num_blocks, threshold);
        //propagate
        stream.full_propagate_multibit_assign_async(
            &mut sum,
            &cuda_bsk.d_vec,
            &cuda_ksk.d_vec,
            cuda_bsk.input_lwe_dimension,
            cuda_bsk.glwe_dimension,
            cuda_bsk.polynomial_size,
            cuda_ksk.decomposition_level_count(),
            cuda_ksk.decomposition_base_log(),
            cuda_bsk.decomp_level_count,
            cuda_bsk.decomp_base_log,
            cuda_bsk.grouping_factor,
            num_blocks as u32,
            message_modulus,
            carry_modulus,
        );

        //Compare
        stream.unchecked_scalar_comparison_integer_radix_multibit_kb_async(
            &mut result.as_mut().ciphertext.d_blocks.0.d_vec,
            &sum,
            &d_scalar_blocks,
            &cuda_bsk.d_vec,
            &cuda_ksk.d_vec,
            message_modulus,
            carry_modulus,
            cuda_bsk.glwe_dimension,
            cuda_bsk.polynomial_size,
            cuda_ksk
                .input_key_lwe_size()
                .to_lwe_dimension(),
            cuda_ksk
                .output_key_lwe_size()
                .to_lwe_dimension(),
            cuda_ksk.decomposition_level_count(),
            cuda_ksk.decomposition_base_log(),
            cuda_bsk.decomp_level_count,
            cuda_bsk.decomp_base_log,
            cuda_bsk.grouping_factor,
            scalar_blocks.len() as u32,
            scalar_blocks.len() as u32,
            ComparisonType::GE,
            false,
        );
    }
    stream.synchronize();
    let elapsed = start.elapsed();
    (result, elapsed)
}

pub fn authenticate_debug(
    decrypt: Box<dyn Fn(&LweCiphertextOwned<u64>) -> u64>,
    bsk: FourierLweMultiBitBootstrapKeyOwned,
    ksk: LweKeyswitchKeyOwned<u64>,
    cuda_bsk: CudaLweMultiBitBootstrapKey,
    cuda_ksk: CudaLweKeyswitchKey<u64>,
    mut probe: LweCiphertextListOwned<u64>,
    luts: GlweCiphertextListOwned<u64>,
    num_blocks: usize,
    message_modulus: MessageModulus,
    carry_modulus: CarryModulus,
    threshold: usize,
    thread_count_ks_u: usize,
    thread_count_bs_u: usize,
    stream: &CudaStream,
) -> (CudaBooleanBlock, Duration) {
    let thread_count_ks = ThreadCount(thread_count_ks_u);
    let thread_count_bs = ThreadCount(thread_count_bs_u);

    let flat_length = probe.lwe_ciphertext_count().0;
    let ciphertext_modulus = probe.ciphertext_modulus();
    let lwe_size = probe.lwe_size().to_lwe_dimension().to_lwe_size();
    let ct_count = flat_length / num_blocks;

    //Buffer
    let mut buffer = common::make_lweciphertextlist(
        flat_length,
        ksk.output_lwe_size(),
        probe.ciphertext_modulus(),
    );

    let mut buffer_cuda = CudaVec::new(lwe_size.0 * flat_length, &stream);
    let mut sum = CudaVec::new(lwe_size.0 * num_blocks, &stream);

    let mut scalar_blocks =
        BlockDecomposer::with_early_stop_at_zero(threshold as u64, message_modulus.0.ilog2())
            .iter_as::<u64>()
            .collect::<Vec<_>>();

    scalar_blocks.truncate(num_blocks);
    let d_scalar_blocks: CudaVec<u64>;
    unsafe {
        d_scalar_blocks = CudaVec::from_cpu_async(&scalar_blocks, &stream);
    }

    let result = CudaLweCiphertextList::new(
        probe.lwe_size().to_lwe_dimension(),
        LweCiphertextCount(1),
        CiphertextModulus::new_native(),
        &stream,
    );

    let ct_info = CudaRadixCiphertextInfo {
        blocks: vec![CudaBlockInfo {
            degree: Degree::new(0),
            message_modulus: message_modulus,
            carry_modulus: carry_modulus,
            pbs_order: PBSOrder::KeyswitchBootstrap,
            noise_level: NoiseLevel::NOMINAL,
        }]
    };

    let mut result = CudaBooleanBlock::from_cuda_radix_ciphertext(CudaRadixCiphertext::new(result, ct_info));

    //Apply pbs
    //tfhe_functions::do_keyswitch(&keys, &encrypted_probes, &mut buffer, &ks_indices);

    buffer.iter_mut().zip(probe.iter()).par_bridge()
        .for_each(|(mut b, p)| {
            par_keyswitch_lwe_ciphertext_with_thread_count(
                &ksk,
                &p,
                &mut b,
                thread_count_ks,
            );
        });

    let start = Instant::now();
    //tfhe_functions::do_pbs(&keys, &buffer, &mut encrypted_probes, &encrypted_luts, &pbs_indices);
    probe.iter_mut().zip(buffer.iter()).zip(luts.iter()).par_bridge()
        .for_each(|((mut p, b), lut)| {
            multi_bit_programmable_bootstrap_lwe_ciphertext(
                &b,
                &mut p,
                &lut,
                &bsk,
                thread_count_bs,
            );
        });

    let dec = probe.iter().map(|p| {
        decrypt(&LweCiphertextOwned::from_container(p.into_container().to_vec(), p.ciphertext_modulus()))
    }).collect_vec();
    println!("LUTs: {:?}", dec);

    unsafe {
        buffer_cuda.copy_from_cpu_async(
            &probe.as_ref(),
            &stream,
        );

        //Do sum
        //let mut sum = tfhe_functions::do_sum(&keys, num_blocks, num_values, &mut encrypted_probes);
        sum.copy_src_range_gpu_to_gpu_async(
            ..lwe_size.0 * num_blocks,
            &buffer_cuda,
            &stream
        );

        stream.unchecked_sum_ciphertexts_integer_radix_multibit_kb_assign_async(
            &mut sum,
            &mut buffer_cuda,
            &cuda_bsk.d_vec,
            &cuda_ksk.d_vec,
            message_modulus,
            carry_modulus,
            cuda_bsk.glwe_dimension,
            cuda_bsk.polynomial_size,
            cuda_ksk.output_key_lwe_size().to_lwe_dimension(),
            cuda_ksk.decomposition_level_count(),
            cuda_ksk.decomposition_base_log(),
            cuda_bsk.decomp_level_count,
            cuda_bsk.decomp_base_log,
            cuda_bsk.grouping_factor,
            num_blocks as u32,
            ct_count as u32,
        );

        //let block = tfhe_functions::do_comparison(&keys, &mut sum, num_blocks, threshold);
        //propagate
        stream.full_propagate_multibit_assign_async(
            &mut sum,
            &cuda_bsk.d_vec,
            &cuda_ksk.d_vec,
            cuda_bsk.input_lwe_dimension,
            cuda_bsk.glwe_dimension,
            cuda_bsk.polynomial_size,
            cuda_ksk.decomposition_level_count(),
            cuda_ksk.decomposition_base_log(),
            cuda_bsk.decomp_level_count,
            cuda_bsk.decomp_base_log,
            cuda_bsk.grouping_factor,
            num_blocks as u32,
            message_modulus,
            carry_modulus,
        );
    }
    stream.synchronize();
    let mut d = LweCiphertextOwned::new(0, lwe_size, ciphertext_modulus);
    unsafe {
        sum.copy_to_cpu_async(d.as_mut(), &stream);
    }
    stream.synchronize();
    println!("SUM: {}", decrypt(&d));

    unsafe {
        //Compare
        stream.unchecked_scalar_comparison_integer_radix_multibit_kb_async(
            &mut result.as_mut().ciphertext.d_blocks.0.d_vec,
            &sum,
            &d_scalar_blocks,
            &cuda_bsk.d_vec,
            &cuda_ksk.d_vec,
            message_modulus,
            carry_modulus,
            cuda_bsk.glwe_dimension,
            cuda_bsk.polynomial_size,
            cuda_ksk
                .input_key_lwe_size()
                .to_lwe_dimension(),
            cuda_ksk
                .output_key_lwe_size()
                .to_lwe_dimension(),
            cuda_ksk.decomposition_level_count(),
            cuda_ksk.decomposition_base_log(),
            cuda_bsk.decomp_level_count,
            cuda_bsk.decomp_base_log,
            cuda_bsk.grouping_factor,
            scalar_blocks.len() as u32,
            scalar_blocks.len() as u32,
            ComparisonType::GE,
            false,
        );
    }
    stream.synchronize();
    let elapsed = start.elapsed();
    (result, elapsed)
}