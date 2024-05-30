use nih_plug::{audio_setup::AudioIOLayout, nih_error};
use rubato::FftFixedInOut;

pub(crate) fn prepare_resampler(
    sample_rate_from: usize,
    sample_rate_to: usize,
    audio_io_layout: &AudioIOLayout,
    err_message: &str,
) -> FftFixedInOut<f32> {
    FftFixedInOut::<f32>::new(
        sample_rate_from,
        sample_rate_to,
        256,
        audio_io_layout
            .main_input_channels
            .map(|i| i.get() as usize)
            .unwrap_or(2),
    )
    .inspect_err(|err| nih_error!("{err_message}: {err}"))
    .unwrap()
}
