use rankwise::*;
fn main() {
    println!("available: {:?}", Isa::available());
    let data = vec![vec![0f32; 100]; 600];
    for &n in &[7usize, 31, 33, 100, 512] {
        let ptrs: Vec<*const u8> = data[..n].iter().map(|v| v.as_ptr() as *const u8).collect();
        let st = RawStack { dtype: DType::F32, frames: &ptrs, shape: &[100], strides: &[4] };
        for isa in Isa::available() {
            let o = Options { isa: Some(isa), ..Default::default() };
            println!("n={n:3} {}", describe(&st, &Op::Median, &o).unwrap());
        }
    }
}
