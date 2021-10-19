use mpi::traits::*;
use parallel_tests::sorting::*;

fn main() {
    // 1. Test sample sort
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        println!("Test sorting algorithms: ");
    }
    test_sample_sort(universe);
}
