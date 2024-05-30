use nih_plug::prelude::*;
use rubato::{FftFixedInOut, Resampler};
use std::sync::Arc;
use utils::prepare_resampler;

mod utils;

// This is a shortened version of the gain example with most comments removed, check out
// https://github.com/robbert-vdh/nih-plug/blob/master/plugins/examples/gain/src/lib.rs to get
// started

const RESAMPLING_ERROR: &'static str = "Error during resampling";

struct TpLimiter {
    params: Arc<TpLimiterParams>,
    audio_io_layout: AudioIOLayout,
    sample_rate: usize,
    oversampling: usize,
    upsampler: FftFixedInOut<f32>,
    downsampler: FftFixedInOut<f32>,
}

#[derive(Params)]
struct TpLimiterParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    #[id = "gain"]
    pub gain: FloatParam,
}

impl Default for TpLimiter {
    fn default() -> Self {
        let sample_rate = 44100;
        let oversampling = 16;
        let high_sample_rate = sample_rate * oversampling;
        let audio_io_layout = AudioIOLayout::default();
        Self {
            params: Arc::new(TpLimiterParams::default()),
            audio_io_layout,
            sample_rate,
            oversampling,
            upsampler: prepare_resampler(
                sample_rate,
                high_sample_rate,
                &audio_io_layout,
                "Couldn't create default upsampler",
            ),
            downsampler: prepare_resampler(
                high_sample_rate,
                sample_rate,
                &audio_io_layout,
                "Couldn't create default downsampler",
            ),
        }
    }
}

impl Default for TpLimiterParams {
    fn default() -> Self {
        Self {
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            // Because the gain parameter is stored as linear gain instead of storing the value as
            // decibels, we need logarithmic smoothing
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            // There are many predefined formatters we can use here. If the gain was stored as
            // decibels instead of as a linear gain value, we could have also used the
            // `.with_step_size(0.1)` function to get internal rounding.
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
        }
    }
}

impl Plugin for TpLimiter {
    const NAME: &'static str = "True Peak Limiter";
    const VENDOR: &'static str = "Viktor Lazarev";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "taurus101v@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.audio_io_layout = audio_io_layout.clone();
        self.sample_rate = buffer_config.sample_rate as usize;

        let high_sample_rate = self.sample_rate * self.oversampling;
        self.upsampler = prepare_resampler(
            self.sample_rate,
            high_sample_rate,
            &self.audio_io_layout,
            "Culdn't init upsampler",
        );
        self.downsampler = prepare_resampler(
            high_sample_rate,
            self.sample_rate,
            &self.audio_io_layout,
            "Culdn't init downsampler",
        );

        // Resize buffers and perform other potentially expensive initialization operations here.
        // The `reset()` function is always called right after this function. You can remove this
        // function if you do not need it.
        true
    }

    fn reset(&mut self) {
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        match self
            .upsampler
            .process(&buffer.as_slice_immutable(), None)
            .and_then(|wave| self.downsampler.process(&wave, None))
        {
            Err(err) => {
                nih_error!("{RESAMPLING_ERROR}: {err}");
                ProcessStatus::Error(RESAMPLING_ERROR)
            }
            Ok(wave) => {
                for (a, b) in buffer.as_slice().into_iter().zip(wave) {
                    for (a, b) in a.iter_mut().zip(b) {
                        *a = b;
                    }
                }

                ProcessStatus::Normal
            }
        }
    }
}

impl ClapPlugin for TpLimiter {
    const CLAP_ID: &'static str = "com.gentoid-dev.tp-limiter";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("One more True Peak Limiter (CLAP) plugin // WIP");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Limiter,
    ];
}

nih_export_clap!(TpLimiter);
