use nih_plug::{audio_setup::AudioIOLayout, nih_error};
use rubato::{FftFixedInOut, ResamplerConstructionError};

pub(crate) fn prepare_resampler(
    sample_rate_from: usize,
    sample_rate_to: usize,
    num_of_channels: usize,
    err_message: &str,
) -> Result<FftFixedInOut<f32>, ResamplerConstructionError> {
    FftFixedInOut::<f32>::new(sample_rate_from, sample_rate_to, 256, num_of_channels)
        .inspect_err(|err| nih_error!("{err_message}: {err}"))
}
