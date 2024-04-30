use crate::core_crypto::algorithms::{allocate_and_generate_new_binary_glwe_secret_key, allocate_and_generate_new_binary_lwe_secret_key, allocate_and_generate_new_lwe_keyswitch_key, par_convert_standard_lwe_multi_bit_bootstrap_key_to_fourier};
use crate::core_crypto::entities::{FourierLweMultiBitBootstrapKey, LweMultiBitBootstrapKeyOwned};
use crate::core_crypto::gpu::{CudaStream};
use crate::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use crate::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use crate::core_crypto::prelude::{FourierLweMultiBitBootstrapKeyOwned, GlweSecretKeyOwned, LweKeyswitchKeyOwned, par_allocate_and_generate_new_lwe_multi_bit_bootstrap_key};
use crate::shortint::{MultiBitPBSParameters};
use crate::shortint::engine::ShortintEngine;


pub fn make_keys_multibit(params: MultiBitPBSParameters, stream: &CudaStream, engine: &mut ShortintEngine) -> (
    (MultiBitPBSParameters, GlweSecretKeyOwned<u64>),
    (LweKeyswitchKeyOwned<u64>, FourierLweMultiBitBootstrapKeyOwned),
    (CudaLweKeyswitchKey<u64>, CudaLweMultiBitBootstrapKey),
    (u64, u64)
) {

    //Generate secret keys
    //Small key
    let lwe_secret_key = allocate_and_generate_new_binary_lwe_secret_key(
        params.lwe_dimension,
        &mut engine.secret_generator,
    );

    // Large key
    let glwe_secret_key = allocate_and_generate_new_binary_glwe_secret_key(
        params.glwe_dimension,
        params.polynomial_size,
        &mut engine.secret_generator,
    );

    //Generate bootstrapping key
    //Generate CPU key
    let bootstrap_key: LweMultiBitBootstrapKeyOwned<u64> =
        par_allocate_and_generate_new_lwe_multi_bit_bootstrap_key(
            &lwe_secret_key,
            &glwe_secret_key,
            params.pbs_base_log,
            params.pbs_level,
            params.grouping_factor,
            params.glwe_noise_distribution,
            params.ciphertext_modulus,
            &mut engine.encryption_generator,
        );

    //Turn into GPU key
    let d_bootstrap_key = CudaLweMultiBitBootstrapKey::from_lwe_multi_bit_bootstrap_key(
        &bootstrap_key,
        stream,
    );

    // Creation of the bootstrapping key in the Fourier domain
    let mut fourier_bsk = FourierLweMultiBitBootstrapKey::new(
        bootstrap_key.input_lwe_dimension(),
        bootstrap_key.glwe_size(),
        bootstrap_key.polynomial_size(),
        bootstrap_key.decomposition_base_log(),
        bootstrap_key.decomposition_level_count(),
        bootstrap_key.grouping_factor(),
    );

    // Conversion to fourier domain
    par_convert_standard_lwe_multi_bit_bootstrap_key_to_fourier(
        &bootstrap_key,
        &mut fourier_bsk,
    );

    // Create key switching key
    //Create CPU key

    // Creation of the key switching key
    let h_key_switching_key = allocate_and_generate_new_lwe_keyswitch_key(
        &glwe_secret_key.as_lwe_secret_key(),
        &lwe_secret_key,
        params.ks_base_log,
        params.ks_level,
        params.lwe_noise_distribution,
        params.ciphertext_modulus,
        &mut engine.encryption_generator,
    );

    //Turn into GPU key
    let d_key_switching_key =
        CudaLweKeyswitchKey::from_lwe_keyswitch_key(&h_key_switching_key, &stream);


    //Scaling factor
    let delta = (1_u64 << 63)
        / (params.message_modulus.0 * params.carry_modulus.0)
        as u64;

    //Overall modulus
    let total_modulus = params.message_modulus.0 * params.carry_modulus.0;

    (
        (params, glwe_secret_key),
        (h_key_switching_key, fourier_bsk),
        (d_key_switching_key, d_bootstrap_key),
        (delta, total_modulus as u64)
    )
}
