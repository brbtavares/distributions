use criterion::{criterion_group, criterion_main, Criterion, BatchSize, black_box};
use probability_rs::dist::poisson::Poisson;
use probability_rs::rng::SplitMix64;
use probability_rs::Distribution; // bring trait for .sample

fn bench_poisson_small_lambda(c: &mut Criterion) {
    let pois = Poisson::new(2.5).unwrap();
    c.bench_function("poisson_sample_lambda_2.5", |b| {
        b.iter_batched(
            || SplitMix64::seed_from_u64(123),
            |mut rng| {
                let mut acc = 0i64;
                for _ in 0..1000 {
                    acc ^= pois.sample(&mut rng);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_poisson_large_lambda(c: &mut Criterion) {
    let pois = Poisson::new(250.0).unwrap();
    c.bench_function("poisson_sample_lambda_250", |b| {
        b.iter_batched(
            || SplitMix64::seed_from_u64(456),
            |mut rng| {
                let mut acc = 0i64;
                for _ in 0..1000 {
                    acc ^= pois.sample(&mut rng);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_poisson_very_large_lambda(c: &mut Criterion) {
    let pois = Poisson::new(1000.0).unwrap();
    c.bench_function("poisson_sample_lambda_1000", |b| {
        b.iter_batched(
            || SplitMix64::seed_from_u64(789),
            |mut rng| {
                let mut acc = 0i64;
                for _ in 0..1000 {
                    acc ^= pois.sample(&mut rng);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_poisson_small_lambda, bench_poisson_large_lambda, bench_poisson_very_large_lambda);
criterion_main!(benches);
