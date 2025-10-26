use distributions::dist::{normal, uniform, exponential, bernoulli, poisson, Discrete, Continuous, Moments, Distribution};
use distributions::rng::SplitMix64;

fn main() {
    let normal = normal::Normal::new(0.0, 1.0).unwrap();
    let uniform = uniform::Uniform::new(-1.0, 1.0).unwrap();
    let expo = exponential::Exponential::new(2.0).unwrap();
    let bern = bernoulli::Bernoulli::new(0.4).unwrap();
    let pois = poisson::Poisson::new(3.0).unwrap();

    let mut rng = SplitMix64::seed_from_u64(2024);
    let x_n = normal.sample(&mut rng);
    let x_u = uniform.sample(&mut rng);
    let x_e = expo.sample(&mut rng);
    let x_b = bern.sample(&mut rng);
    let x_p = pois.sample(&mut rng);

    println!("Normal sample: {x_n:.6} pdf(0)={:.6}", normal.pdf(0.0));
    println!("Uniform sample: {x_u:.6} mean={:.3} var={:.3}", uniform.mean(), uniform.variance());
    println!("Exponential sample: {x_e:.6} CDF(1)={:.6}", expo.cdf(1.0));
    println!("Bernoulli sample: {x_b} p=0.4 var={:.3}", bern.variance());
    println!("Poisson sample: {x_p} lambda=3 pmf(3)={:.6}", pois.pmf(3));
}

