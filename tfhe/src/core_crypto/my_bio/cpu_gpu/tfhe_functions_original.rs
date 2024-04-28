use crate::core_crypto::algorithms::{allocate_and_generate_new_binary_glwe_secret_key, allocate_and_generate_new_binary_lwe_secret_key, allocate_and_generate_new_lwe_keyswitch_key};
use crate::core_crypto::gpu::{CudaStream};
use crate::core_crypto::gpu::lwe_bootstrap_key::CudaLweBootstrapKey;
use crate::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use crate::core_crypto::prelude::{FourierLweBootstrapKeyOwned, LweBootstrapKeyOwned, par_allocate_and_generate_new_lwe_bootstrap_key, par_convert_standard_lwe_bootstrap_key_to_fourier};
use crate::integer::gpu::CudaServerKey;
use crate::integer::gpu::server_key::CudaBootstrappingKey;
use crate::shortint::{ClassicPBSParameters, ClientKey, MaxNoiseLevel, ServerKey};
use crate::shortint::ciphertext::MaxDegree;
use crate::shortint::engine::ShortintEngine;
use crate::shortint::server_key::ShortintBootstrappingKey;


pub fn make_keys_original(params: ClassicPBSParameters, stream: &CudaStream, engine: &mut ShortintEngine) -> (
    (ClassicPBSParameters, ClientKey),
    (ServerKey, CudaServerKey)
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

    let client_key = ClientKey {
        glwe_secret_key,
        lwe_secret_key,
        parameters: params.into(),
    };

    //Generate bootstrapping key
    //Generate CPU key
    let bootstrap_key: LweBootstrapKeyOwned<u64> = par_allocate_and_generate_new_lwe_bootstrap_key(
        &client_key.lwe_secret_key,
        &client_key.glwe_secret_key,
        params.pbs_base_log,
        params.pbs_level,
        params.glwe_noise_distribution,
        params.ciphertext_modulus,
        &mut engine.encryption_generator,
    );

    //Turn into GPU key
    let d_bootstrap_key = CudaLweBootstrapKey::from_lwe_bootstrap_key(
        &bootstrap_key,
        stream,
    );

    // Creation of the bootstrapping key in the Fourier domain
    let mut fourier_bsk = FourierLweBootstrapKeyOwned::new(
        bootstrap_key.input_lwe_dimension(),
        bootstrap_key.glwe_size(),
        bootstrap_key.polynomial_size(),
        bootstrap_key.decomposition_base_log(),
        bootstrap_key.decomposition_level_count(),
    );

    // Conversion to fourier domain
    par_convert_standard_lwe_bootstrap_key_to_fourier(
        &bootstrap_key,
        &mut fourier_bsk,
    );

    // Create key switching key
    //Create CPU key
    let h_key_switching_key = allocate_and_generate_new_lwe_keyswitch_key(
        &client_key.glwe_secret_key.as_lwe_secret_key(),
        &client_key.lwe_secret_key,
        params.ks_base_log,
        params.ks_level,
        params.lwe_noise_distribution,
        params.ciphertext_modulus,
        &mut engine.encryption_generator,
    );

    //Turn into GPU key
    let d_key_switching_key =
        CudaLweKeyswitchKey::from_lwe_keyswitch_key(&h_key_switching_key, &stream);

    let max_value = params.message_modulus.0 * params.carry_modulus.0 - 1;

    // The maximum number of operations before we need to clean the carry buffer
    let max_degree = MaxDegree::new(max_value);

    let max_noise_level = MaxNoiseLevel::from_msg_carry_modulus(
        params.message_modulus,
        params.carry_modulus,
    );

    let server_key = ServerKey {
        key_switching_key: h_key_switching_key,
        bootstrapping_key: ShortintBootstrappingKey::Classic(fourier_bsk),
        message_modulus: params.message_modulus,
        carry_modulus: params.carry_modulus,
        max_degree: max_degree,
        max_noise_level: max_noise_level,
        ciphertext_modulus: params.ciphertext_modulus,
        pbs_order: params.encryption_key_choice.into(),
    };

    let cuda_server_key = CudaServerKey {
        key_switching_key: d_key_switching_key,
        bootstrapping_key: CudaBootstrappingKey::Classic(d_bootstrap_key),
        message_modulus: params.message_modulus,
        carry_modulus: params.carry_modulus,
        max_degree,
        max_noise_level,
        ciphertext_modulus: params.ciphertext_modulus,
        pbs_order: params.encryption_key_choice.into(),
    };

    (
        (params, client_key),
        (server_key, cuda_server_key)
    )
}
