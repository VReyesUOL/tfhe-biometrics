use crate::core_crypto::algorithms::{allocate_and_generate_new_binary_glwe_secret_key, allocate_and_generate_new_binary_lwe_secret_key, allocate_and_generate_new_lwe_keyswitch_key, par_allocate_and_generate_new_lwe_bootstrap_key};
use crate::core_crypto::entities::{LweBootstrapKeyOwned};
use crate::core_crypto::gpu::{cuda_keyswitch_lwe_ciphertext, cuda_programmable_bootstrap_lwe_ciphertext, CudaStream};
use crate::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use crate::core_crypto::gpu::lwe_bootstrap_key::CudaLweBootstrapKey;
use crate::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use crate::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::my_bio::common::Keys;
use crate::core_crypto::prelude::{GlweSecretKeyOwned, LweCiphertextCount, LweSecretKeyOwned };
use crate::integer::block_decomposition::BlockDecomposer;
use crate::integer::gpu::ciphertext::{CudaRadixCiphertext, CudaUnsignedRadixCiphertext};
use crate::integer::gpu::ciphertext::boolean_value::CudaBooleanBlock;
use crate::integer::gpu::ciphertext::info::{CudaBlockInfo, CudaRadixCiphertextInfo};
use crate::integer::gpu::{ComparisonType, CudaServerKey};
use crate::integer::gpu::server_key::CudaBootstrappingKey;
use crate::integer::gpu::server_key::CudaBootstrappingKey::Classic;
use crate::shortint::{CiphertextModulus, ShortintParameterSet};
use crate::shortint::ciphertext::{Degree, MaxDegree};
use crate::shortint::engine::ShortintEngine;
use crate::shortint::parameters::NoiseLevel;


pub fn make_keys(params: ShortintParameterSet, stream: CudaStream, engine: &mut ShortintEngine) -> Keys {

    let lwe_secret_key = allocate_and_generate_new_binary_lwe_secret_key(
        params.lwe_dimension(),
        &mut engine.secret_generator,
    );

    // generate the rlwe secret key
    let glwe_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
        params.glwe_dimension(),
        params.polynomial_size(),
        &mut engine.secret_generator,
    );

    let max_degree = MaxDegree::integer_radix_server_key(
        params.message_modulus(),
        params.carry_modulus(),
    );
    let h_bootstrap_key: LweBootstrapKeyOwned<u64> =
        par_allocate_and_generate_new_lwe_bootstrap_key(
            &lwe_secret_key,
            &glwe_secret_key,
            params.pbs_base_log(),
            params.pbs_level(),
            params.glwe_noise_distribution(),
            params.ciphertext_modulus(),
            &mut engine.encryption_generator,
        );

    let d_bootstrap_key =
        CudaLweBootstrapKey::from_lwe_bootstrap_key(&h_bootstrap_key, &stream);

    // Creation of the key switching key
    let h_key_switching_key = allocate_and_generate_new_lwe_keyswitch_key(
        &glwe_secret_key.as_lwe_secret_key(),
        &lwe_secret_key,
        params.ks_base_log(),
        params.ks_level(),
        params.lwe_noise_distribution(),
        params.ciphertext_modulus(),
        &mut engine.encryption_generator,
    );

    let d_key_switching_key =
        CudaLweKeyswitchKey::from_lwe_keyswitch_key(&h_key_switching_key, &stream);

    let delta = (1_u64 << 63)
        / (params.message_modulus().0 * params.carry_modulus().0)
        as u64;

    let total_modulus = params.message_modulus().0 * params.carry_modulus().0;

    let server_key = CudaServerKey {
        key_switching_key: d_key_switching_key,
        bootstrapping_key: Classic(d_bootstrap_key),
        message_modulus: params.message_modulus(),
        carry_modulus: params.carry_modulus(),
        max_degree: max_degree,
        max_noise_level: params.max_noise_level(),
        ciphertext_modulus: params.ciphertext_modulus(),
        pbs_order: params.encryption_key_choice().into(),
    };

    Keys {
        lwe_secret_key: lwe_secret_key,
        glwe_secret_key: glwe_secret_key,
        server_key: server_key,
        delta: delta,
        total_modulus: total_modulus,
        stream: stream,
        parameters: params,
    }
}

pub fn make_keys_no_server_key(params: ShortintParameterSet, stream: &CudaStream, engine: &mut ShortintEngine) -> (LweSecretKeyOwned<u64>, GlweSecretKeyOwned<u64>, CudaLweKeyswitchKey<u64>, CudaLweBootstrapKey, u64, u64, ShortintParameterSet) {

    let lwe_secret_key = allocate_and_generate_new_binary_lwe_secret_key(
        params.lwe_dimension(),
        &mut engine.secret_generator,
    );

    // generate the rlwe secret key
    let glwe_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
        params.glwe_dimension(),
        params.polynomial_size(),
        &mut engine.secret_generator,
    );

    let h_bootstrap_key: LweBootstrapKeyOwned<u64> =
        par_allocate_and_generate_new_lwe_bootstrap_key(
            &lwe_secret_key,
            &glwe_secret_key,
            params.pbs_base_log(),
            params.pbs_level(),
            params.glwe_noise_distribution(),
            params.ciphertext_modulus(),
            &mut engine.encryption_generator,
        );

    let d_bootstrap_key =
        CudaLweBootstrapKey::from_lwe_bootstrap_key(&h_bootstrap_key, &stream);

    // Creation of the key switching key
    let h_key_switching_key = allocate_and_generate_new_lwe_keyswitch_key(
        &glwe_secret_key.as_lwe_secret_key(),
        &lwe_secret_key,
        params.ks_base_log(),
        params.ks_level(),
        params.lwe_noise_distribution(),
        params.ciphertext_modulus(),
        &mut engine.encryption_generator,
    );

    let d_key_switching_key =
        CudaLweKeyswitchKey::from_lwe_keyswitch_key(&h_key_switching_key, &stream);

    let delta = (1_u64 << 63)
        / (params.message_modulus().0 * params.carry_modulus().0)
        as u64;

    let total_modulus = params.message_modulus().0 * params.carry_modulus().0;

    (
        lwe_secret_key,
        glwe_secret_key,
        d_key_switching_key,
        d_bootstrap_key,
        delta,
        total_modulus as u64,
        params,
    )
}

pub fn do_keyswitch(
    keys: &Keys,
    input: &CudaLweCiphertextList<u64>,
    mut output: &mut CudaLweCiphertextList<u64>,
    indices: &Vec<CudaVec<u64>>,
) {
    assert_eq!(indices.len(), 2, "keyswitch expects two sets of indices");
    cuda_keyswitch_lwe_ciphertext(
        &keys.server_key.key_switching_key,
        &input,
        &mut output,
        &indices[0],
        &indices[1],
        &keys.stream,
    );
}

pub fn do_pbs(
    keys: &Keys,
    input: &CudaLweCiphertextList<u64>,
    mut output: &mut CudaLweCiphertextList<u64>,
    luts: &CudaGlweCiphertextList<u64>,
    indices: &Vec<CudaVec<u64>>
) {
    assert_eq!(indices.len(), 3, "bootstrap expects three sets of indices");
    if let Classic(bsk) = &keys.server_key.bootstrapping_key {
        cuda_programmable_bootstrap_lwe_ciphertext(
            &input,
            &mut output,
            &luts,
            &indices[0],
            &indices[1],
            &indices[2],
            LweCiphertextCount(luts.glwe_ciphertext_count().0),
            bsk,
            &keys.stream,
        );
    } else {
        panic!("Not implemented");
    }
}

pub fn expand_radix_with_trivial_zeros(keys: &Keys, new_num_blocks: usize, ct_list: &mut CudaLweCiphertextList<u64>) {
    let lwe_size = ct_list.lwe_dimension().to_lwe_size();

    let mut extended_ct_vec =
        unsafe { CudaVec::new_async(new_num_blocks * lwe_size.0, &keys.stream) };
    unsafe {
        extended_ct_vec.memset_async(0u64, &keys.stream);
        extended_ct_vec.copy_from_gpu_async(&ct_list.0.d_vec, &keys.stream);
    }
    keys.stream.synchronize();
    ct_list.0.d_vec = extended_ct_vec;
    ct_list.0.lwe_ciphertext_count = LweCiphertextCount(new_num_blocks);
}

pub fn do_sum(keys: &Keys, num_blocks: usize, ct_count: usize, ct_list: &mut CudaLweCiphertextList<u64>) -> CudaLweCiphertextList<u64> {
    let lwe_size = ct_list.lwe_dimension().to_lwe_size();
    let mut res_cuda_vec = CudaVec::new(lwe_size.0 * num_blocks, &keys.stream);
    let bsk = match &keys.server_key.bootstrapping_key {
        Classic(bsk) => {bsk}
        CudaBootstrappingKey::MultiBit(_) => {panic!("Not implemented")}
    };
    unsafe {
        res_cuda_vec.copy_src_range_gpu_to_gpu_async(
            ..lwe_size.0 * num_blocks,
            &ct_list.0.d_vec,
            &keys.stream
        );

        keys.stream.unchecked_sum_ciphertexts_integer_radix_classic_kb_assign_async(
            &mut res_cuda_vec,
            &mut ct_list.0.d_vec,
            &bsk.d_vec,
            &keys.server_key.key_switching_key.d_vec,
            keys.server_key.message_modulus,
            keys.server_key.carry_modulus,
            bsk.glwe_dimension,
            bsk.polynomial_size,
            keys.server_key.key_switching_key
                .output_key_lwe_size()
                .to_lwe_dimension(),
            keys.server_key.key_switching_key.decomposition_level_count(),
            keys.server_key.key_switching_key.decomposition_base_log(),
            bsk.decomp_level_count,
            bsk.decomp_base_log,
            num_blocks as u32,
            ct_count as u32,
        )
    }
    keys.stream.synchronize();
    CudaLweCiphertextList::from_cuda_vec(
        res_cuda_vec,
        LweCiphertextCount(num_blocks),
        keys.server_key.ciphertext_modulus,
    )
}

pub unsafe fn propagate(keys: &Keys, ct_list: &mut CudaLweCiphertextList<u64>, num_blocks: usize) {
    //num_blocks = num_radix_blocks
    let bsk = match &keys.server_key.bootstrapping_key {
        Classic(bsk) => {bsk}
        CudaBootstrappingKey::MultiBit(_) => {panic!("Not implemented")}
    };

    keys.stream.full_propagate_classic_assign_async(
        &mut ct_list.0.d_vec,
        &bsk.d_vec,
        &keys.server_key.key_switching_key.d_vec,
        bsk.input_lwe_dimension(),
        bsk.glwe_dimension(),
        bsk.polynomial_size(),
        keys.server_key.key_switching_key.decomposition_level_count(),
        keys.server_key.key_switching_key.decomposition_base_log(),
        bsk.decomp_level_count(),
        bsk.decomp_base_log(),
        num_blocks as u32,
        keys.server_key.message_modulus,
        keys.server_key.carry_modulus,
    );
}

pub unsafe fn async_compare(keys: &Keys, ct_in: &CudaLweCiphertextList<u64>, scalar: u64, num_blocks: usize) -> CudaBooleanBlock {
    let message_modulus = keys.server_key.message_modulus.0;

    let mut scalar_blocks =
        BlockDecomposer::with_early_stop_at_zero(scalar, message_modulus.ilog2())
            .iter_as::<u64>()
            .collect::<Vec<_>>();

// scalar is obviously bigger if it has non-zero
// blocks  after lhs's last block
    let is_scalar_obviously_bigger = scalar_blocks
        .get(num_blocks..)
        .is_some_and(|sub_slice| sub_slice.iter().any(|&scalar_block| scalar_block != 0));

    if is_scalar_obviously_bigger {
        let value = 0;
        let ct_res: CudaUnsignedRadixCiphertext = keys.server_key.create_trivial_radix(value, 1, &keys.stream);
        return CudaBooleanBlock::from_cuda_radix_ciphertext(ct_res.ciphertext);
    }

    // If we are still here, that means scalar_blocks above
    // num_blocks are 0s, we can remove them
    // as we will handle them separately.
    scalar_blocks.truncate(num_blocks);

    let d_scalar_blocks: CudaVec<u64> = CudaVec::from_cpu_async(&scalar_blocks, &keys.stream);

    let block = CudaLweCiphertextList::new(
        ct_in.lwe_dimension(),
        LweCiphertextCount(1),
        CiphertextModulus::new_native(),
        &keys.stream,
    );

    let ct_info = CudaRadixCiphertextInfo {
        blocks: vec![CudaBlockInfo {
            degree: Degree::new(0),
            message_modulus: keys.server_key.message_modulus,
            carry_modulus: keys.server_key.carry_modulus,
            pbs_order: keys.server_key.pbs_order,
            noise_level: NoiseLevel::NOMINAL,
        }]
    };

    let bsk = match &keys.server_key.bootstrapping_key {
        Classic(bsk) => {bsk}
        CudaBootstrappingKey::MultiBit(_) => {panic!("Not implemented")}
    };

    let mut result = CudaBooleanBlock::from_cuda_radix_ciphertext(CudaRadixCiphertext::new(block, ct_info));

    keys.stream.unchecked_scalar_comparison_integer_radix_classic_kb_async(
        &mut result.as_mut().ciphertext.d_blocks.0.d_vec,
        &ct_in.0.d_vec,
        &d_scalar_blocks,
        &bsk.d_vec,
        &keys.server_key.key_switching_key.d_vec,
        keys.server_key.message_modulus,
        keys.server_key.carry_modulus,
        bsk.glwe_dimension,
        bsk.polynomial_size,
        keys.server_key.key_switching_key
            .input_key_lwe_size()
            .to_lwe_dimension(),
        keys.server_key.key_switching_key
            .output_key_lwe_size()
            .to_lwe_dimension(),
        keys.server_key.key_switching_key.decomposition_level_count(),
        keys.server_key.key_switching_key.decomposition_base_log(),
        bsk.decomp_level_count,
        bsk.decomp_base_log,
        ct_in.lwe_ciphertext_count().0 as u32,
        scalar_blocks.len() as u32,
        ComparisonType::GE,
        false,
    );

    result
}

pub fn do_comparison(keys: &Keys, ct_in: &mut CudaLweCiphertextList<u64>, num_blocks: usize, threshold: u64) -> CudaBooleanBlock {
    let result;
    unsafe {
        propagate(keys, ct_in, num_blocks);
        result = async_compare(keys, ct_in, threshold, num_blocks);
    }
    keys.stream.synchronize();
    result
}