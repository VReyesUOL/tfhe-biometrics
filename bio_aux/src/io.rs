
use std::fs;
use config::*;

pub fn read_qbins(path:&str) -> Vec<f64> {
    let mut result = Vec::new();
    let qbins_csv = fs::read_to_string(path);
    match qbins_csv {
        Ok(qbins_csv) => { result = csv_string_to_real_vec(qbins_csv);
        }
        Err(_) => {}
    }
    result
}

fn csv_string_to_real_vec(input_string:String) -> Vec<f64> {
    let mut result:Vec<f64> = Vec::new();
    for str_value in input_string.split(",") {
        let value = str_value.parse::<f64>();
        match value {
            Ok(value) => {result.push(value)}
            Err(_) => {}
        }
    }
    result
}

fn csv_to_vec_square(csv_string:String) -> Option<Vec<Vec<f64>>> {
    let mut output_vec = Vec::new();
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(csv_string.as_bytes());
    for record in reader.records() {
        let record = record.unwrap();
        let mut entry = Vec::new();
        for i in 0.. record.len() {
            entry.push(record[i].parse::<f64>().unwrap());
        }
        output_vec.push(entry);
    }
    Some(output_vec)
}

fn read_dataset(path:&str) -> Option<Vec<(usize,Vec<f64>)>> {
    let dataset_csv_string = fs::read_to_string(path).unwrap();
    let dataset_vec = csv_to_vec_square(dataset_csv_string).unwrap();
    let mut bio_probe_vec = Vec::new();
    for entry in dataset_vec{
        let probe = (entry[0] as usize, entry[1..].to_vec());
        bio_probe_vec.push(probe);
    }
    Some(bio_probe_vec)
}

pub fn read_sample_with_id_in_line_fpd(path:&str, sample_id:usize) -> Option<(usize, Vec<f64>)>{
    let dataset = read_dataset(path).unwrap();
    let new_id = ((sample_id - 1) % dataset.len()) + 1;
    Some(dataset.get(new_id - 1 ).unwrap().clone())
}

fn quantize_feature(raw_feature: f64, qbins: &Vec<f64>) -> usize {
    for bin in 0 .. qbins.len() {
        if raw_feature <= qbins[bin] {
            return bin
        }
    }
    qbins.len()
}

pub fn quantize_feature_vector(raw_feature_vector: Vec<f64>, qbins: &Vec<f64>) -> Vec<usize> {
    let mut quantized_feature_vector = Vec::new();
    for value in raw_feature_vector.iter() {
        quantized_feature_vector.push(quantize_feature(value.clone(), qbins))
    }
    quantized_feature_vector
}

pub fn probe_and_template_generation_radix_prepare(entry_num:usize, template_num: usize) -> (Vec<u8>, Vec<u8>) {
    let qbin_filename = format!("{}{}.csv", DATA_SET_NAME, QBIN_SUFFIX);
    let qbins_path = [DATA_PATH, LOOKUP_TABLES_FOLDER, DATA_SET_NAME, qbin_filename.as_str()].join(PATH_SEPARATOR);
    let bins = read_qbins(qbins_path.as_str());

    let csv_filename = format!("{}.csv", DATA_SET_NAME);
    let csv_path = [DATA_PATH, csv_filename.as_str()].join(PATH_SEPARATOR);

    // Load raw samples of template from file
    let probe_entry =
        read_sample_with_id_in_line_fpd(csv_path.as_str(), entry_num).unwrap();
    let template_entry =
        read_sample_with_id_in_line_fpd(csv_path.as_str(), template_num).unwrap();
    let (_, entry) = probe_entry;
    let (_, template) = template_entry;

    // Quantize
    let probe_vec:Vec<u8> = quantize_feature_vector(entry, &bins).into_iter().map(|x| x as u8).collect();
    let template_vec:Vec<u8> = quantize_feature_vector(template, &bins).into_iter().map(|x| x as u8).collect();

    (probe_vec, template_vec)
}

fn csv_to_helr_table(csv_string:String) -> Option<Vec<Vec<i32>>>{
    let mut data_vec = Vec::new();
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(csv_string.as_bytes());
    for record in reader.records() {
        let record = record.unwrap();
        let mut entry = Vec::new();
        for i in 0.. record.len(){
            entry.push(record[i].parse::<i32>().unwrap());
        }
        data_vec.push(entry);
    }
    Some(data_vec)
}

pub fn read_helr_tables(path:&str, num_tables:usize) -> Option<Vec<Vec<Vec<i32>>>> {
    let mut helr_vec = Vec::new();
    let file_type = "csv";
    for i in 0..num_tables {
        let full_path = format!("{path}{i}.{file_type}");
        let helr_csv_string = fs::read_to_string(full_path).unwrap();
        helr_vec.push(csv_to_helr_table(helr_csv_string).unwrap());
    }
    Some(helr_vec)
}