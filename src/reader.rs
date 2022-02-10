// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Accessing row from raw bytes

use crate::error::{DataFusionError, Result};
use crate::bitmap::{all_valid, get_bit, null_width, NullBitsFormatter};
use crate::{get_offsets, supported};
use arrow::array::*;
use arrow::datatypes::{DataType, Schema};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Read `data` of raw-bytes rows starting at `offsets` out to a record batch
pub fn read_as_batch(
    data: &mut [u8],
    schema: Arc<Schema>,
    offsets: Vec<usize>,
) -> Result<RecordBatch> {
    let row_num = offsets.len();
    let mut output = MutableRecordBatch::new(row_num, schema.clone());
    let mut row = RowReader::new(&schema, data);

    for offset in offsets.iter().take(row_num) {
        row.point_to(*offset);
        read_row(&row, &mut output, &schema);
    }

    output.output().map_err(DataFusionError::ArrowError)
}

macro_rules! get_idx {
    ($NATIVE: ident, $SELF: ident, $IDX: ident, $WIDTH: literal) => {{
        $SELF.assert_index_valid($IDX);
        let offset = $SELF.field_offsets[$IDX];
        let start = $SELF.base_offset + offset;
        let end = start + $WIDTH;
        $NATIVE::from_le_bytes($SELF.data[start..end].try_into().unwrap())
    }};
}

macro_rules! fn_get_idx {
    ($NATIVE: ident, $WIDTH: literal) => {
        paste::item! {
            fn [<get_ $NATIVE>](&self, idx: usize) -> $NATIVE {
                self.assert_index_valid(idx);
                let offset = self.field_offsets[idx];
                let start = self.base_offset + offset;
                let end = start + $WIDTH;
                $NATIVE::from_le_bytes(self.data[start..end].try_into().unwrap())
            }
        }
    };
}

macro_rules! fn_get_idx_opt {
    ($NATIVE: ident) => {
        paste::item! {
            fn [<get_ $NATIVE _opt>](&self, idx: usize) -> Option<$NATIVE> {
                if self.is_valid_at(idx) {
                    Some(self.[<get_ $NATIVE>](idx))
                } else {
                    None
                }
            }
        }
    };
}

/// Read the tuple `data[base_offset..]` we are currently pointing to
pub struct RowReader<'a> {
    /// Raw bytes slice where the tuple stores
    data: &'a [u8],
    /// Start position for the current tuple in the raw bytes slice.
    base_offset: usize,
    /// Total number of fields for each tuple.
    field_count: usize,
    /// The number of bytes used to store null bits for each field.
    null_width: usize,
    /// Starting offset for each fields in the raw bytes.
    /// For fixed length fields, it's where the actual data stores.
    /// For variable length fields, it's a pack of (offset << 32 | length) if we use u64.
    field_offsets: Vec<usize>,
}

impl<'a> std::fmt::Debug for RowReader<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let null_bits = self.null_bits();
        write!(
            f,
            "{:?}",
            NullBitsFormatter::new(null_bits, self.field_count)
        )
    }
}

impl<'a> RowReader<'a> {
    /// new
    pub fn new(schema: &Arc<Schema>, data: &'a [u8]) -> Self {
        assert!(supported(schema));
        let field_count = schema.fields().len();
        let null_width = null_width(field_count);
        let (field_offsets, _) = get_offsets(null_width, schema);
        Self {
            data,
            base_offset: 0,
            field_count,
            null_width,
            field_offsets,
        }
    }

    /// Update this row to point to position `offset` in `base`
    pub fn point_to(&mut self, offset: usize) {
        self.base_offset = offset;
    }

    #[inline]
    fn assert_index_valid(&self, idx: usize) {
        assert!(idx < self.field_count);
    }

    #[inline(always)]
    fn null_bits(&self) -> &[u8] {
        let start = self.base_offset;
        &self.data[start..start + self.null_width]
    }

    #[inline(always)]
    fn all_valid(&self) -> bool {
        let null_bits = self.null_bits();
        all_valid(null_bits, self.field_count)
    }

    fn is_valid_at(&self, idx: usize) -> bool {
        get_bit(self.null_bits(), idx)
    }

    fn get_bool(&self, idx: usize) -> bool {
        self.assert_index_valid(idx);
        let offset = self.field_offsets[idx];
        let value = &self.data[self.base_offset + offset..];
        value[0] != 0
    }

    fn get_u8(&self, idx: usize) -> u8 {
        self.assert_index_valid(idx);
        let offset = self.field_offsets[idx];
        self.data[self.base_offset + offset]
    }

    fn_get_idx!(u16, 2);
    fn_get_idx!(u32, 4);
    fn_get_idx!(u64, 8);
    fn_get_idx!(i8, 1);
    fn_get_idx!(i16, 2);
    fn_get_idx!(i32, 4);
    fn_get_idx!(i64, 8);
    fn_get_idx!(f32, 4);
    fn_get_idx!(f64, 8);

    fn get_date32(&self, idx: usize) -> i32 {
        get_idx!(i32, self, idx, 4)
    }

    fn get_date64(&self, idx: usize) -> i64 {
        get_idx!(i64, self, idx, 8)
    }

    fn get_utf8(&self, idx: usize) -> &str {
        self.assert_index_valid(idx);
        let offset_size = self.get_u64(idx);
        let offset = (offset_size >> 32) as usize;
        let len = (offset_size & 0xffff_ffff) as usize;
        let varlena_offset = self.base_offset + offset;
        let bytes = &self.data[varlena_offset..varlena_offset + len];
        std::str::from_utf8(bytes).unwrap()
    }

    fn get_binary(&self, idx: usize) -> &[u8] {
        self.assert_index_valid(idx);
        let offset_size = self.get_u64(idx);
        let offset = (offset_size >> 32) as usize;
        let len = (offset_size & 0xffff_ffff) as usize;
        let varlena_offset = self.base_offset + offset;
        &self.data[varlena_offset..varlena_offset + len]
    }

    fn_get_idx_opt!(bool);
    fn_get_idx_opt!(u8);
    fn_get_idx_opt!(u16);
    fn_get_idx_opt!(u32);
    fn_get_idx_opt!(u64);
    fn_get_idx_opt!(i8);
    fn_get_idx_opt!(i16);
    fn_get_idx_opt!(i32);
    fn_get_idx_opt!(i64);
    fn_get_idx_opt!(f32);
    fn_get_idx_opt!(f64);

    fn get_date32_opt(&self, idx: usize) -> Option<i32> {
        if self.is_valid_at(idx) {
            Some(self.get_date32(idx))
        } else {
            None
        }
    }

    fn get_date64_opt(&self, idx: usize) -> Option<i64> {
        if self.is_valid_at(idx) {
            Some(self.get_date64(idx))
        } else {
            None
        }
    }

    fn get_utf8_opt(&self, idx: usize) -> Option<&str> {
        if self.is_valid_at(idx) {
            Some(self.get_utf8(idx))
        } else {
            None
        }
    }
}

fn read_row(
    row: &RowReader,
    batch: &mut MutableRecordBatch,
    schema: &Arc<Schema>,
) {
    if row.all_valid() {
        for ((col_idx, to), field) in batch
            .arrays
            .iter_mut()
            .enumerate()
            .zip(schema.fields().iter())
        {
            read_field_null_free(to, field.data_type(), col_idx, row)
        }
    } else {
        for ((col_idx, to), field) in batch
            .arrays
            .iter_mut()
            .enumerate()
            .zip(schema.fields().iter())
        {
            read_field(to, field.data_type(), col_idx, row)
        }
    }
}

macro_rules! fn_read_field {
    ($NATIVE: ident, $ARRAY: ident) => {
        paste::item! {
            extern "C" fn [<read_field_ $NATIVE>](to: &mut Box<dyn ArrayBuilder>, col_idx: usize, row: &RowReader) {
                let to = to
                    .as_any_mut()
                    .downcast_mut::<$ARRAY>()
                    .unwrap();
                to.append_option(row.[<get_ $NATIVE _opt>](col_idx))
                    .map_err(DataFusionError::ArrowError)
                    .unwrap();
            }

            extern "C" fn [<read_field_ $NATIVE _nf>](to: &mut Box<dyn ArrayBuilder>, col_idx: usize, row: &RowReader) {
                let to = to
                    .as_any_mut()
                    .downcast_mut::<$ARRAY>()
                    .unwrap();
                to.append_value(row.[<get_ $NATIVE>](col_idx))
                    .map_err(DataFusionError::ArrowError)
                    .unwrap();
            }
        }
    };
}

fn_read_field!(bool, BooleanBuilder);
fn_read_field!(u8, UInt8Builder);
fn_read_field!(u16, UInt16Builder);
fn_read_field!(u32, UInt32Builder);
fn_read_field!(u64, UInt64Builder);
fn_read_field!(i8, Int8Builder);
fn_read_field!(i16, Int16Builder);
fn_read_field!(i32, Int32Builder);
fn_read_field!(i64, Int64Builder);
fn_read_field!(f32, Float32Builder);
fn_read_field!(f64, Float64Builder);
fn_read_field!(date32, Date32Builder);
fn_read_field!(date64, Date64Builder);
fn_read_field!(utf8, StringBuilder);

fn read_field_binary(to: &mut Box<dyn ArrayBuilder>, col_idx: usize, row: &RowReader) {
    let to = to.as_any_mut().downcast_mut::<BinaryBuilder>().unwrap();
    if row.is_valid_at(col_idx) {
        to.append_value(row.get_binary(col_idx)).unwrap();
    } else {
        to.append_null().unwrap();
    }
}

fn read_field_binary_nf(to: &mut Box<dyn ArrayBuilder>, col_idx: usize, row: &RowReader) {
    let to = to.as_any_mut().downcast_mut::<BinaryBuilder>().unwrap();
    to.append_value(row.get_binary(col_idx))
        .map_err(DataFusionError::ArrowError)
        .unwrap();
}

fn read_field(
    to: &mut Box<dyn ArrayBuilder>,
    dt: &DataType,
    col_idx: usize,
    row: &RowReader,
) {
    use DataType::*;
    match dt {
        Boolean => read_field_bool(to, col_idx, row),
        UInt8 => read_field_u8(to, col_idx, row),
        UInt16 => read_field_u16(to, col_idx, row),
        UInt32 => read_field_u32(to, col_idx, row),
        UInt64 => read_field_u64(to, col_idx, row),
        Int8 => read_field_i8(to, col_idx, row),
        Int16 => read_field_i16(to, col_idx, row),
        Int32 => read_field_i32(to, col_idx, row),
        Int64 => read_field_i64(to, col_idx, row),
        Float32 => read_field_f32(to, col_idx, row),
        Float64 => read_field_f64(to, col_idx, row),
        Date32 => read_field_date32(to, col_idx, row),
        Date64 => read_field_date64(to, col_idx, row),
        Utf8 => read_field_utf8(to, col_idx, row),
        Binary => read_field_binary(to, col_idx, row),
        _ => unimplemented!(),
    }
}

fn read_field_null_free(
    to: &mut Box<dyn ArrayBuilder>,
    dt: &DataType,
    col_idx: usize,
    row: &RowReader,
) {
    use DataType::*;
    match dt {
        Boolean => read_field_bool_nf(to, col_idx, row),
        UInt8 => read_field_u8_nf(to, col_idx, row),
        UInt16 => read_field_u16_nf(to, col_idx, row),
        UInt32 => read_field_u32_nf(to, col_idx, row),
        UInt64 => read_field_u64_nf(to, col_idx, row),
        Int8 => read_field_i8_nf(to, col_idx, row),
        Int16 => read_field_i16_nf(to, col_idx, row),
        Int32 => read_field_i32_nf(to, col_idx, row),
        Int64 => read_field_i64_nf(to, col_idx, row),
        Float32 => read_field_f32_nf(to, col_idx, row),
        Float64 => read_field_f64_nf(to, col_idx, row),
        Date32 => read_field_date32_nf(to, col_idx, row),
        Date64 => read_field_date64_nf(to, col_idx, row),
        Utf8 => read_field_utf8_nf(to, col_idx, row),
        Binary => read_field_binary_nf(to, col_idx, row),
        _ => unimplemented!(),
    }
}

struct MutableRecordBatch {
    arrays: Vec<Box<dyn ArrayBuilder>>,
    schema: Arc<Schema>,
}

impl MutableRecordBatch {
    fn new(target_batch_size: usize, schema: Arc<Schema>) -> Self {
        let arrays = new_arrays(&schema, target_batch_size);
        Self { arrays, schema }
    }

    fn output(&mut self) -> ArrowResult<RecordBatch> {
        let result = make_batch(self.schema.clone(), self.arrays.drain(..).collect());
        result
    }
}

fn new_arrays(schema: &Arc<Schema>, batch_size: usize) -> Vec<Box<dyn ArrayBuilder>> {
    schema
        .fields()
        .iter()
        .map(|field| {
            let dt = field.data_type();
            make_builder(dt, batch_size)
        })
        .collect::<Vec<_>>()
}

fn make_batch(
    schema: Arc<Schema>,
    mut arrays: Vec<Box<dyn ArrayBuilder>>,
) -> ArrowResult<RecordBatch> {
    let columns = arrays.iter_mut().map(|array| array.finish()).collect();
    RecordBatch::try_new(schema, columns)
}
