use std::time::{Duration, Instant};
use itertools::Itertools;
use crate::core_crypto::gpu::{cuda_keyswitch_lwe_ciphertext_async, cuda_multi_bit_programmable_bootstrap_lwe_ciphertext_async, CudaStream};
use crate::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use crate::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use crate::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use crate::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::my_bio::common;
use crate::core_crypto::prelude::LweCiphertextCount;
use crate::integer::block_decomposition::BlockDecomposer;
use crate::integer::gpu::ciphertext::boolean_value::CudaBooleanBlock;
use crate::integer::gpu::ciphertext::{CudaRadixCiphertext};
use crate::integer::gpu::ciphertext::info::{CudaBlockInfo, CudaRadixCiphertextInfo};
use crate::integer::gpu::ComparisonType;
use crate::shortint::{CarryModulus, CiphertextModulus, MessageModulus, PBSOrder};
use crate::shortint::ciphertext::{Degree, NoiseLevel};

pub fn authenticate(
    bsk: CudaLweMultiBitBootstrapKey,
    ksk: CudaLweKeyswitchKey<u64>,
    mut probe: CudaLweCiphertextList<u64>,
    luts: CudaGlweCiphertextList<u64>,
    num_blocks: usize,
    message_modulus: MessageModulus,
    carry_modulus: CarryModulus,
    threshold: usize,
    stream: &CudaStream,
) -> (CudaBooleanBlock, Duration) {
    let flat_length = probe.lwe_ciphertext_count().0;
    let lwe_size = probe.lwe_dimension().to_lwe_size();
    let ct_count = flat_length / num_blocks;
    //Buffer
    let mut buffer = common::make_cuda_lweciphertextlist(
        flat_length,
        ksk.output_key_lwe_size(),
        probe.ciphertext_modulus(),
        &stream,
    );

    //Prepare indices
    let total = 5;
    let indices_raw = (0..flat_length as u64).collect_vec();
    let mut indices: Vec<CudaVec<u64>> = Vec::with_capacity(total);
    (0..total).for_each(|_| {
        let mut in_indices = unsafe { CudaVec::<u64>::new_async(flat_length, &stream) };
        stream.synchronize();
        unsafe {
            in_indices.copy_from_cpu_async(indices_raw.as_ref(), &stream);
        }
        stream.synchronize();
        indices.push(in_indices);
    });

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
        probe.lwe_dimension(),
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
    unsafe {
        //tfhe_functions::do_keyswitch(&keys, &encrypted_probes, &mut buffer, &ks_indices);
        cuda_keyswitch_lwe_ciphertext_async(
            &ksk,
            &probe,
            &mut buffer,
            &indices[0],
            &indices[1],
            &stream,
        );
    }
    stream.synchronize();

    let start = Instant::now();
    unsafe {
        //tfhe_functions::do_pbs(&keys, &buffer, &mut encrypted_probes, &encrypted_luts, &pbs_indices);
        cuda_multi_bit_programmable_bootstrap_lwe_ciphertext_async(
            &buffer,
            &mut probe,
            &luts,
            &indices[2],
            &indices[3],
            &indices[3],
            &bsk,
            &stream,
        );

        //Do sum
        //let mut sum = tfhe_functions::do_sum(&keys, num_blocks, num_values, &mut encrypted_probes);
        sum.copy_src_range_gpu_to_gpu_async(
            ..lwe_size.0 * num_blocks,
            &probe.0.d_vec,
            &stream
        );

        stream.unchecked_sum_ciphertexts_integer_radix_multibit_kb_assign_async(
            &mut sum,
            &mut probe.0.d_vec,
            &bsk.d_vec,
            &ksk.d_vec,
            message_modulus,
            carry_modulus,
            bsk.glwe_dimension,
            bsk.polynomial_size,
            ksk.output_key_lwe_size().to_lwe_dimension(),
            ksk.decomposition_level_count(),
            ksk.decomposition_base_log(),
            bsk.decomp_level_count,
            bsk.decomp_base_log,
            bsk.grouping_factor,
            num_blocks as u32,
            ct_count as u32,
        );

        //let block = tfhe_functions::do_comparison(&keys, &mut sum, num_blocks, threshold);
        //propagate
        stream.full_propagate_multibit_assign_async(
            &mut sum,
            &bsk.d_vec,
            &ksk.d_vec,
            bsk.input_lwe_dimension,
            bsk.glwe_dimension,
            bsk.polynomial_size,
            ksk.decomposition_level_count(),
            ksk.decomposition_base_log(),
            bsk.decomp_level_count,
            bsk.decomp_base_log,
            bsk.grouping_factor,
            num_blocks as u32,
            message_modulus,
            carry_modulus,
        );

        //Compare
        stream.unchecked_scalar_comparison_integer_radix_multibit_kb_async(
            &mut result.as_mut().ciphertext.d_blocks.0.d_vec,
            &sum,
            &d_scalar_blocks,
            &bsk.d_vec,
            &ksk.d_vec,
            message_modulus,
            carry_modulus,
            bsk.glwe_dimension,
            bsk.polynomial_size,
            ksk
                .input_key_lwe_size()
                .to_lwe_dimension(),
            ksk
                .output_key_lwe_size()
                .to_lwe_dimension(),
            ksk.decomposition_level_count(),
            ksk.decomposition_base_log(),
            bsk.decomp_level_count,
            bsk.decomp_base_log,
            bsk.grouping_factor,
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