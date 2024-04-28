pub struct Config {
    pub data_set_name: &'static str,
    pub num_blocks: usize,
    pub num_blocks_sum: usize,
    pub block_length: usize,
    pub num_tables: usize,
    pub threshold: i64,
}

pub const BMDB1: Config = Config {
    data_set_name: "BMDB",
    num_blocks: 6,
    num_blocks_sum: 6,
    block_length: 2,
    num_tables: 36,
    threshold: 14,
};
pub const BMDB2: Config = Config {
    data_set_name: "BMDB",
    num_blocks: 4,
    num_blocks_sum: 4,
    block_length: 3,
    num_tables: 36,
    threshold: 14,
};
pub const PUT: Config = Config {
    data_set_name: "PUT",
    num_blocks: 4,
    num_blocks_sum: 4,
    block_length: 3,
    num_tables: 49,
    threshold: -53,
};

pub const FRGC: Config = Config {
    data_set_name: "FRGC",
    num_blocks: 4,
    num_blocks_sum: 4,
    block_length: 3,
    num_tables: 94,
    threshold: -1,
};

//constant values
pub const TABLE_PREFIX: &str = "HELR";
pub const QBIN_SUFFIX: &str = "_qbins";
pub const PATH_SEPARATOR: &str = "/";
pub const DATA_PATH: &str = "data";
pub const LOOKUP_TABLES_FOLDER: &str = "lookupTables";