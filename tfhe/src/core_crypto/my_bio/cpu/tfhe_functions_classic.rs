use crate::core_crypto::algorithms::{allocate_and_generate_new_binary_glwe_secret_key, allocate_and_generate_new_binary_lwe_secret_key, allocate_and_generate_new_lwe_keyswitch_key};
use crate::core_crypto::prelude::{FourierLweBootstrapKey, LweBootstrapKeyOwned, par_allocate_and_generate_new_lwe_bootstrap_key, par_convert_standard_lwe_bootstrap_key_to_fourier};
use crate::integer::encryption::KnowsMessageModulus;
use crate::shortint;
use crate::shortint::{ClassicPBSParameters};
use crate::shortint::ciphertext::MaxDegree;
use crate::shortint::engine::ShortintEngine;
use crate::shortint::server_key::ShortintBootstrappingKey;


pub fn make_keys_classic(params: ClassicPBSParameters, engine: &mut ShortintEngine) -> (shortint::ClientKey, shortint::ServerKey) {
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
    let bootstrap_key: LweBootstrapKeyOwned<u64> =
        par_allocate_and_generate_new_lwe_bootstrap_key(
            &lwe_secret_key,
            &glwe_secret_key,
            params.pbs_base_log,
            params.pbs_level,
            params.glwe_noise_distribution,
            params.ciphertext_modulus,
            &mut engine.encryption_generator,
        );

    // Creation of the bootstrapping key in the Fourier domain
    let mut fourier_bsk = FourierLweBootstrapKey::new(
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

    //Client key
    let client_key = shortint::ClientKey {
        glwe_secret_key,
        lwe_secret_key,
        parameters: params.into(),
    };

    let max_value = client_key.parameters.message_modulus().0 * client_key.parameters.carry_modulus().0 - 1;
    let max = MaxDegree::new(max_value);

    let server_key = shortint::ServerKey {
        key_switching_key: h_key_switching_key,
        bootstrapping_key: ShortintBootstrappingKey::Classic(fourier_bsk),
        message_modulus: client_key.message_modulus(),
        carry_modulus: client_key.parameters.carry_modulus(),
        max_degree: max,
        max_noise_level: client_key.parameters.max_noise_level(),
        ciphertext_modulus: client_key.parameters.ciphertext_modulus(),
        pbs_order: client_key.parameters.encryption_key_choice().into(),
    };

    (
        client_key, server_key
    )
}
