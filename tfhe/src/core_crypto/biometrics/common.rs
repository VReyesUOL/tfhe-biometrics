use itertools::Itertools;
use crate::boolean::parameters::PolynomialSize;
use crate::core_crypto::algorithms::{allocate_and_encrypt_new_lwe_ciphertext, decrypt_lwe_ciphertext, decrypt_lwe_ciphertext_list, encrypt_glwe_ciphertext_assign};
use crate::core_crypto::entities::{GlweCiphertext, GlweCiphertextList, GlweCiphertextListOwned, GlweCiphertextOwned, GlweSecretKeyOwned, LweCiphertextListOwned, LweCiphertextOwned, LweSecretKey, LweSecretKeyOwned, Plaintext, PlaintextList, PlaintextRef};
use crate::core_crypto::gpu::{CudaDevice, CudaStream};
use crate::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use crate::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::prelude::{ContiguousEntityContainer, ContiguousEntityContainerMut, GlweCiphertextCount, GlweSize, LweCiphertextCount, LweCiphertextList, LweSize, PlaintextCount};
use crate::integer::BooleanBlock;
use crate::integer::encryption::KnowsMessageModulus;
use crate::integer::gpu::ciphertext::boolean_value::CudaBooleanBlock;
use crate::integer::gpu::CudaServerKey;
use crate::shortint;
use crate::shortint::engine::ShortintEngine;
use crate::shortint::{CarryModulus, CiphertextModulus, MessageModulus, ShortintParameterSet};
use crate::shortint::parameters::Degree;
use crate::shortint::server_key::LookupTableOwned;


pub struct Keys {
    pub lwe_secret_key: LweSecretKeyOwned<u64>,
    pub glwe_secret_key: GlweSecretKeyOwned<u64>,
    pub server_key: CudaServerKey,
    pub delta: u64,
    pub total_modulus: usize,
    pub stream: CudaStream,
    pub parameters: ShortintParameterSet,
}

pub fn encode(m: u64, total_mod: u64, delta: u64) -> Plaintext<u64> {
    Plaintext(
        (m % total_mod) * delta
    )
}

pub fn decode(p: &PlaintextRef<u64>, delta: u64) -> u64 {
    let d = p.0;
    //The bit before the message
    let rounding_bit = delta >> 1;

    //compute the rounding bit
    let rounding = (d & rounding_bit) << 1;

    (d.wrapping_add(rounding)) / delta
}

pub fn encrypt(m: u64, engine: &mut ShortintEngine, secret_key: &LweSecretKey<&[u64]>, params: ShortintParameterSet, total_mod: u64, delta: u64) -> LweCiphertextOwned<u64> {
    allocate_and_encrypt_new_lwe_ciphertext(
        &secret_key,
        encode(m, total_mod, delta),
        params.glwe_noise_distribution(),
        params.ciphertext_modulus(),
        &mut engine.encryption_generator,
    )
}

pub fn decrypt_boolean_block_client_key(block: &BooleanBlock, client_key: &shortint::ClientKey) -> bool {
    let delta = (1_u64 << 63)
        / (client_key.parameters.message_modulus().0 * client_key.parameters.carry_modulus().0) as u64;
    let v = decrypt_lwe_ciphertext(
        &client_key.glwe_secret_key.as_lwe_secret_key(),
        &block.0.ct,
    );
    let d = decode(&PlaintextRef(&v.0), delta);
    (d % client_key.message_modulus().0 as u64) != 0

}

pub fn decrypt_cuda_boolean_block_client_key(cuda_block: &CudaBooleanBlock, client_key: &shortint::ClientKey, stream: &CudaStream) -> bool {
    let block = cuda_block.to_boolean_block(&stream);
    let delta = (1_u64 << 63)
        / (client_key.parameters.message_modulus().0 * client_key.parameters.carry_modulus().0) as u64;
    let v = decrypt_lwe_ciphertext(
        &client_key.glwe_secret_key.as_lwe_secret_key(),
        &block.0.ct,
    );
    let d = decode(&PlaintextRef(&v.0), delta);
    (d % client_key.message_modulus().0 as u64) != 0

}

pub fn decrypt_boolean_block(cuda_block: &CudaBooleanBlock, lwe_secret_key: &LweSecretKey<&[u64]>, params: ShortintParameterSet, delta: u64, stream: &CudaStream) -> bool {
    let block = cuda_block.to_boolean_block(&stream);
    let v = decrypt_lwe_ciphertext(
        &lwe_secret_key,
        &block.0.ct,
    );
    let d = decode(&PlaintextRef(&v.0), delta);
    (d % params.message_modulus().0 as u64) != 0
}

pub fn make_context_gpu() -> (CudaStream, ShortintEngine) {
    let gpu_index = 0;
    let device = CudaDevice::new(gpu_index);
    let stream = CudaStream::new_unchecked(device);

    let engine = ShortintEngine::new();
    (stream, engine)
}

pub fn make_lweciphertextlist(ct_count: usize, lwe_size: LweSize, ct_mod: CiphertextModulus) -> LweCiphertextListOwned<u64> {
    LweCiphertextListOwned::new(
        0,
        lwe_size, //lwe_secret_key.lwe_dimension().to_lwe_size(),
        LweCiphertextCount(ct_count),
        ct_mod,
    )
}

pub fn make_cuda_lweciphertextlist(ct_count: usize, lwe_size: LweSize, ct_mod: CiphertextModulus, stream: &CudaStream) -> CudaLweCiphertextList<u64> {
    let ct_list_out = LweCiphertextListOwned::new(
        0,
        lwe_size, //lwe_secret_key.lwe_dimension().to_lwe_size(),
        LweCiphertextCount(ct_count),
        ct_mod,
    );
    CudaLweCiphertextList::from_lwe_ciphertext_list(
        &ct_list_out,
        &stream,
    )
}

pub fn encrypt_ciphertextlist(values: Vec<u64>, mut engine: &mut ShortintEngine, secret_key: &LweSecretKey<&[u64]>, total_mod: u64, delta: u64, params: ShortintParameterSet) -> LweCiphertextListOwned<u64> {
    let raw_list = values.iter().map(|v|{
        encrypt(*v, &mut engine, secret_key, params, total_mod, delta)
    }).collect_vec();

    let mut ct_list = LweCiphertextListOwned::new(
        0,
        secret_key.lwe_dimension().to_lwe_size(),
        LweCiphertextCount(raw_list.len()),
        params.ciphertext_modulus(),
    );
    ct_list.as_mut_view().iter_mut().zip(raw_list).for_each(|(mut v, ct)| {
        v.as_mut().copy_from_slice(ct.into_container().as_mut());
    });
    ct_list
}


pub fn encrypt_cuda_ciphertextlist(values: Vec<u64>, mut engine: &mut ShortintEngine, secret_key: &LweSecretKey<&[u64]>, total_mod: u64, delta: u64, params: ShortintParameterSet, stream: &CudaStream) -> CudaLweCiphertextList<u64> {
    let raw_list = values.iter().map(|v|{
        encrypt(*v, &mut engine, secret_key, params, total_mod, delta)
    }).collect_vec();

    let mut ct_list = LweCiphertextListOwned::new(
        0,
        secret_key.lwe_dimension().to_lwe_size(),
        LweCiphertextCount(raw_list.len()),
        params.ciphertext_modulus(),
    );
    ct_list.as_mut_view().iter_mut().zip(raw_list).for_each(|(mut v, ct)| {
        v.as_mut().copy_from_slice(ct.into_container().as_mut());
    });

    CudaLweCiphertextList::from_lwe_ciphertext_list(
        &ct_list,
        &stream,
    )
}

pub fn decrypt(lwe_ct: &LweCiphertextOwned<u64>, secret_key: &LweSecretKey<&[u64]>, delta: u64) -> u64 {
    let res = decrypt_lwe_ciphertext(secret_key, lwe_ct);
    decode(&PlaintextRef(&res.0), delta)
}

pub fn decrypt_ciphertextlist(lwe_cts: &LweCiphertextList<Vec<u64>>, secret_key: &LweSecretKeyOwned<u64>, delta: u64) -> Vec<u64> {
    let count = lwe_cts.lwe_ciphertext_count().0;

    let mut res_list = PlaintextList::new(
        0,
        PlaintextCount(count),
    );

    decrypt_lwe_ciphertext_list(
        &secret_key,
        &lwe_cts,
        &mut res_list,
    );

    res_list.iter().map(|p| {
        decode(&p, delta)
    }).collect_vec()
}

pub fn decrypt_cuda_ciphertextlist(cts: &CudaLweCiphertextList<u64>, keys: &Keys, delta: u64) -> Vec<u64> {
    let count = cts.lwe_ciphertext_count().0;
    let lwe_cts = cts.to_lwe_ciphertext_list(&keys.stream);

    let mut res_list = PlaintextList::new(
        0,
        PlaintextCount(count),
    );

    decrypt_lwe_ciphertext_list(
        &keys.glwe_secret_key.as_lwe_secret_key(),
        &lwe_cts,
        &mut res_list,
    );

    res_list.iter().map(|p| {
        decode(&p, delta)
    }).collect_vec()
}

pub fn make_indices(max: u64, total: usize, slots: usize, keys: &Keys) -> Vec<CudaVec<u64>> {
    let indices_raw = (0..max).collect_vec();
    let mut result: Vec<CudaVec<u64>> = Vec::with_capacity(total);
    (0..total).for_each(|_| {
        let mut in_indices = unsafe { CudaVec::<u64>::new_async(slots, &keys.stream) };
        keys.stream.synchronize();
        unsafe {
            in_indices.copy_from_cpu_async(indices_raw.as_ref(), &keys.stream);
        }
        keys.stream.synchronize();
        result.push(in_indices);
    });
    result

}

pub fn generate_lookup_tables_individual<F>(fs: Vec<Vec<F>>, client_key: &shortint::ClientKey, params: ShortintParameterSet, engine: &mut ShortintEngine) -> Vec<Vec<LookupTableOwned>>
    where
        F: Fn(u64) -> u64,
{
    let gen_acc = |f| {
        generate_accumulator(
            client_key.glwe_secret_key.glwe_dimension().to_glwe_size(),
            params.polynomial_size(),
            params.message_modulus(),
            params.ciphertext_modulus(),
            params.carry_modulus(),
            f,
        )
    };

    let lut_list = fs.iter().map(|f_vec| {
        f_vec.iter().map(|f| {
            let (mut acc, max_value)  = gen_acc(f);
            encrypt_glwe_ciphertext_assign(
                &client_key.glwe_secret_key,
                &mut acc,
                params.glwe_noise_distribution(),
                &mut engine.encryption_generator,
            );
            LookupTableOwned {
                acc,
                degree: Degree::new(max_value as usize)
            }
        }).collect_vec()
    }).collect_vec();

    lut_list
}

pub fn generate_lookup_tables<F>(fs: Vec<F>, glwe_secret_key: &GlweSecretKeyOwned<u64>, params: ShortintParameterSet, engine: &mut ShortintEngine) -> GlweCiphertextListOwned<u64>
    where
        F: Fn(u64) -> u64,
{
    let gen_acc = |f| {
        generate_accumulator(
            glwe_secret_key.glwe_dimension().to_glwe_size(),
            params.polynomial_size(),
            params.message_modulus(),
            params.ciphertext_modulus(),
            params.carry_modulus(),
            f,
        ).0
    };

    let mut lut_list = GlweCiphertextList::new(
        0,
        params.glwe_dimension().to_glwe_size(),
        params.polynomial_size(),
        GlweCiphertextCount(fs.len()),
        params.ciphertext_modulus(),
    );

    lut_list.iter_mut().zip(fs).for_each(|(mut lut, f)| {
        let mut acc = gen_acc(f);
        encrypt_glwe_ciphertext_assign(
            &glwe_secret_key,
            &mut acc,
            params.glwe_noise_distribution(),
            &mut engine.encryption_generator,
        );
        lut.as_mut().copy_from_slice(acc.as_mut());
    });

    lut_list
}



pub fn generate_cuda_lookup_tables<F>(fs: Vec<F>, glwe_secret_key: &GlweSecretKeyOwned<u64>, params: ShortintParameterSet, engine: &mut ShortintEngine, stream: &CudaStream) -> CudaGlweCiphertextList<u64>
    where
        F: Fn(u64) -> u64,
{
    let gen_acc = |f| {
        generate_accumulator(
            glwe_secret_key.glwe_dimension().to_glwe_size(),
            params.polynomial_size(),
            params.message_modulus(),
            params.ciphertext_modulus(),
            params.carry_modulus(),
            f,
        ).0
    };

    let mut lut_list = GlweCiphertextList::new(
        0,
        params.glwe_dimension().to_glwe_size(),
        params.polynomial_size(),
        GlweCiphertextCount(fs.len()),
        params.ciphertext_modulus(),
    );

    lut_list.iter_mut().zip(fs).for_each(|(mut lut, f)| {
        let mut acc = gen_acc(f);
        encrypt_glwe_ciphertext_assign(
            &glwe_secret_key,
            &mut acc,
            params.glwe_noise_distribution(),
            &mut engine.encryption_generator,
        );
        lut.as_mut().copy_from_slice(acc.as_mut());
    });

    CudaGlweCiphertextList::from_glwe_ciphertext_list(
        &lut_list,
        &stream,
    )
}

pub fn generate_accumulator<F>(
    glwe_size: GlweSize,
    polynomial_size: PolynomialSize,
    message_modulus: MessageModulus,
    ciphertext_modulus: CiphertextModulus,
    carry_modulus: CarryModulus,
    f: F,
) -> (GlweCiphertextOwned<u64>, u64) where
    F: Fn(u64) -> u64,
{
    let mut accumulator = GlweCiphertext::new(
        0,
        glwe_size,
        polynomial_size,
        ciphertext_modulus,
    );

    let mut accumulator_view = accumulator.as_mut_view();

    accumulator_view.get_mut_mask().as_mut().fill(0);

    // Modulus of the msg contained in the msg bits and operations buffer
    let modulus_sup = message_modulus.0 * carry_modulus.0;

    // N/(p/2) = size of each block
    let box_size = polynomial_size.0 / modulus_sup;

    // Value of the shift we multiply our messages by
    let delta = (1_u64 << 63) / (message_modulus.0 * carry_modulus.0) as u64;

    let mut body = accumulator_view.get_mut_body();
    let accumulator_u64 = body.as_mut();

    // Tracking the max value of the function to define the degree later
    let mut max_value = 0;

    for i in 0..modulus_sup {
        let index = i * box_size;
        let f_eval = f(i as u64);
        max_value = max_value.max(f_eval);
        accumulator_u64[index..index + box_size].fill(f_eval * delta);
    }

    let half_box_size = box_size / 2;

    // Negate the first half_box_size coefficients
    for a_i in accumulator_u64[0..half_box_size].iter_mut() {
        *a_i = (*a_i).wrapping_neg();
    }

    // Rotate the accumulator
    accumulator_u64.rotate_left(half_box_size);

    (accumulator, max_value)
}

