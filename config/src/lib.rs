//Param PUT
/*
NUM_BLOCK = 4
BLOCK_LENGTH = 3
NUM_TABLES = 49
 */
//Param BMDB2
/*
NUM_BLOCK = 6
BLOCK_LENGTH = 2
NUM_TABLES = 36
THRESHOLD = 14
 */
//Param BMDB
/*
NUM_BLOCK = 4
BLOCK_LENGTH = 3
NUM_TABLES = 36
THRESHOLD = 14
 */
//Param FRGC
/*
NUM_BLOCK = 4
BLOCK_LENGTH = 3
NUM_TABLES = 94
 */
pub const DATA_SET_NAME: &str = "BMDB";
pub const NUM_BLOCK: usize=4;
pub const NUM_BLOCK_SUM: usize=4;
pub const BLOCK_LENGTH: u8 = 3;
pub const NUM_TABLES: usize= 36;
pub const THRESHOLD: u64 = 14;

//constant values
pub const TABLE_PREFIX: &str = "HELR";
pub const QBIN_SUFFIX: &str = "_qbins";
pub const PATH_SEPARATOR: &str = "/";
pub const DATA_PATH: &str = "data";
pub const LOOKUP_TABLES_FOLDER: &str = "lookupTables";