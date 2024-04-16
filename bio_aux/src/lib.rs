pub mod io;

use config::*;


pub fn generate_functions(template: &Vec<u8>) -> (Vec<Vec<Box< dyn Fn(u64) -> u64>>>, u64, usize) {
    // read precomputed HELR tables from files
    let tables_path = [DATA_PATH, LOOKUP_TABLES_FOLDER, DATA_SET_NAME, TABLE_PREFIX].join(PATH_SEPARATOR);
    let helr_tables =
        io::read_helr_tables(tables_path.as_str(), NUM_TABLES).unwrap();

    // offset all HELR tables to only have nonnegative entries and save the cumulated offset for all tables
    let (offset_helr_tables, offset) = offset_helr_table(helr_tables);

    // generate a vector of LUT/value vectors for each table
    let mut lut_vecs = Vec::new();
    for (idx, table) in offset_helr_tables.iter().enumerate() {
        let decomposed_helr = decompose_helr_table(&table, BLOCK_LENGTH as usize, NUM_BLOCK);
        let dim = decomposed_helr[0].len() as u64;
        let template_x = template[idx];
        let mut lut_vec = Vec::new();
        for block_num in 0..NUM_BLOCK {
            let copied_table = decomposed_helr.clone();
            let boxed: Box<dyn Fn(u64) -> u64> = Box::new(move |probe_y: u64| -> u64 {
                if probe_y < dim {
                    copied_table[template_x as usize][probe_y as usize][block_num]
                } else {
                    0
                }
            });
            lut_vec.push(boxed);
        }
        lut_vecs.push(lut_vec);
    }

    let threshold = THRESHOLD + offset as u64;
    (lut_vecs, threshold, NUM_BLOCK_SUM)
}

pub fn offset_helr_table(helr_tables:Vec<Vec<Vec<i32>>>) -> (Vec<Vec<Vec<u32>>>, u32 ) {
    let mut offset:u32 = 0;
    let mut tables = Vec::with_capacity(helr_tables.len());
    for helr in helr_tables{
        let local_offset = helr[0][helr[0].len() - 1].abs();
        offset += local_offset as u32;
        let mut new_helr = Vec::with_capacity(helr.len());
        for row in helr {
            let mut new_row = Vec::with_capacity(row.len());
            for entry in row {
                new_row.push((entry  + local_offset) as u32);
            }
            new_helr.push(new_row)
        }
        tables.push(new_helr);
    }
    (tables, offset)
}

pub fn decompose_helr_table(helr: &Vec<Vec<u32>>, block_length: usize, block_count: usize) -> Vec<Vec<Vec<u64>>> {
    let radix = 1 << block_length;
    let mut result = Vec::with_capacity(helr.len());

    for entry in helr {
        let mut decomposed_entry = Vec::with_capacity(entry.len());
        for &value in entry {
            let mut blocks = Vec::with_capacity(block_count);
            let mut remaining_value = value;

            // Decompose the value into blocks
            for _ in 0..block_count {
                let block = (remaining_value % radix) as u64;
                blocks.push(block);
                remaining_value /= radix;
            }

            decomposed_entry.push(blocks);
        }
        result.push(decomposed_entry);
    }
    result
}


pub fn encrypt_feature_vec_radix<F, T: Clone>(feature_vec: Vec<u8>, encrypt: F) -> Vec<Vec<T>>
where
    F: Fn(u64) -> T,
{
    let mut result = Vec::new();
    for feature in feature_vec {
        let mut inner = Vec::with_capacity(NUM_BLOCK);
        let block = encrypt(feature as u64);
        for _ in 1..NUM_BLOCK {
            inner.push(block.clone());
        }
        inner.push(block);
        result.push(inner);
    }
    result
}